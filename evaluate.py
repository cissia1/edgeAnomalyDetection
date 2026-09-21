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
