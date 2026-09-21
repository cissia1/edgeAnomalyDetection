import glob
import os
import numpy as np
import main as eg 

#Metrics
def auc(neg_scores, pos_scores):
    """Area under the ROC curve.
      Equivalently: the probability that a randomly chosen fault frame scores
      higher than a randomly chosen healthy frame. Computed from ranks (the Mann-Whitney U identity), which handles ties
      correctly and needs no threshold sweep.
    """
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([neg_scores, pos_scores]))
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    rank_sum_pos = ranks[n_neg:].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def roc_points(neg_scores, pos_scores, n=200):
   #false-positive rate, true-positive rate
    lo = min(neg_scores.min(), pos_scores.min())
    hi = max(neg_scores.max(), pos_scores.max())
    thresholds = np.linspace(hi, lo, n)
    fpr = np.array([(neg_scores >= t).mean() for t in thresholds])
    tpr = np.array([(pos_scores >= t).mean() for t in thresholds])
    return fpr, tpr

#Audio loading
def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=eg.sample_rate, mono=True)
    return eg.trim_edges(y)

def rms_db(y):
    import librosa
    n_fft = int(eg.sample_rate * eg.frame_length / 1000)
    hop = int(eg.sample_rate * eg.hop_length / 1000)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    return 20.0 * np.log10(rms + 1e-10)

class Recording:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        y = load_audio(path)
        self.features = eg.features_from_signal(y)
        self.rms = rms_db(y)

#These are my 4 scores, with eachm method judged on the same frames.
def score_rms_raw(rec, ref):
    return np.abs(rec.rms - ref["rms_mu"])[1:]

def score_rms(rec, ref):
    centred = rec.rms - rec.rms.mean()
    return np.abs(centred - ref["rms_centred_mu"])[1:]

def score_spectrum(rec, ref, detector):
    u = detector.normalize(rec.features)
    return np.mean(u[1:] ** 2, axis=1)

def score_reservoir(rec, ref, detector):
    return detector.score(rec.features)

def main():
    healthy_path = "recordings/healthy.wav"
    control_path = "recordings/healthy_2.wav"
    fault_paths = sorted(glob.glob("recordings/fault_*.wav"))

    for p in (healthy_path, control_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    if not fault_paths:
        raise SystemExit("no recordings/fault_*.wav files found")

    print(f"training on : {healthy_path}")
    print(f"control     : {control_path}")
    print(f"faults      : {len(fault_paths)} file(s)\n")

    #teach mode on healthy
    healthy = Recording(healthy_path)
    detector = eg.Detector()
    detector.fit(healthy.features)

    #Reference statistics for the RMS baselines
    ref = {
        "rms_mu": healthy.rms.mean(),
        "rms_centred_mu": (healthy.rms - healthy.rms.mean()).mean(),
    }

    #Scoring every recording with every method
    methods = ["rms_raw", "rms", "spectrum", "reservoir"]

 def score_all(rec):
        w = eg.washout
        return {
            "rms_raw":   score_rms_raw(rec, ref)[w:],
            "rms":       score_rms(rec, ref)[w:],
            "spectrum":  score_spectrum(rec, ref, detector)[w:],
            "reservoir": score_reservoir(rec, ref, detector)[w:],
        }

    control = Recording(control_path)
    control_scores = score_all(control)

    cal_err = detector.cal_err[eg.washout:]

    faults = []
    for p in fault_paths:
        rec = Recording(p)
        faults.append((rec.name, score_all(rec)))

    #Table 1: AUC, fault vs the separate healthy session
    print("AUC - each fault vs. healthy_2  (0.5 = useless, 1.0 = perfect)")
    header = f"{'fault':<22}" + "".join(f"{m:>11}" for m in methods)
    print(header)
    print("-" * len(header))

    totals = {m: [] for m in methods}
    for name, sc in faults:
        row = f"{name:<22}"
        for m in methods:
            a = auc(control_scores[m], sc[m])
            totals[m].append(a)
            row += f"{a:>11.3f}"
        print(row)

    print("-" * len(header))
    print(f"{'MEAN':<22}" + "".join(f"{np.mean(totals[m]):>11.3f}" for m in methods))

    #Table 2: operating point at the detectors own threshold
    print(f"\nAt the reservoir's threshold ({detector.thresh:.3f}):")
    fa = 100.0 * (control_scores["reservoir"] > detector.thresh).mean()
    print(f"  false-alarm rate on healthy_2 : {fa:5.1f}% of frames")
    for name, sc in faults:
        det = 100.0 * (sc["reservoir"] > detector.thresh).mean()
        print(f"  detection rate, {name:<20}: {det:5.1f}% of frames")

    #how different are two healthy sessions
    sess = auc(cal_err, control_scores["reservoir"])
    print(f"\nSession effect (held-out healthy vs healthy_2): AUC {sess:.3f}")
    print("  1.00 = the two healthy sessions are perfectly distinguishable")
    print("  0.50 = indistinguishable (ideal - all that is left is the fan)")

    #ROC curves
    try:
        import matplotlib.pyplot as plt
        n = len(faults)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
        for ax, (name, sc) in zip(axes[0], faults):
            for m in methods:
                fpr, tpr = roc_points(control_scores[m], sc[m])
                ax.plot(fpr, tpr, lw=1.4,
                        label=f"{m} ({auc(control_scores[m], sc[m]):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance (0.500)")
            ax.set_title(name)
            ax.set_xlabel("false-positive rate")
            ax.set_ylabel("true-positive rate")
            ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig("roc_curves.png", dpi=120)
        print("\nSaved ROC curves to roc_curves.png")
    except ImportError:
        print("\n(matplotlib not installed - skipping ROC plot)")


if __name__ == "__main__":
    main()
