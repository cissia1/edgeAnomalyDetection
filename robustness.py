import glob
import time
import numpy as np
import main as eg
from evaluate import auc, Recording, score_spectrum

CONFIGS = [
    (0.95, 0.30, "old default"),
    (0.95, 0.10, "tuned"),
    (0.99, 0.10, "grid best"),
]
RIDGES = [1e-6, 1e-2, 1.0, 10.0]
SEEDS = range(10)

def mean_fault_auc(healthy, control, faults, rho, leak, ridge, seed):
    eg.spcrtl_radius, eg.leak_rate = rho, leak
    eg.ridge_strength, eg.seed = ridge, seed
    det = eg.Detector()
    det.fit(healthy.features)
    w = eg.washout
    ctrl = det.score(control.features)[w:]
    return np.mean([auc(ctrl, det.score(f.features)[w:]) for f in faults])

def main():
    print("loading recordings...")
    healthy = Recording("recordings/healthy.wav")
    control = Recording("recordings/healthy_2.wav")
    faults = [Recording(p) for p in sorted(glob.glob("recordings/fault_*.wav"))]
    if not faults:
        raise SystemExit("no recordings/fault_*.wav files found")
    w = eg.washout
    print(f"  {len(faults)} fault(s), washout = {w} frames\n")

    #memoryless baseline, no random reservoir
    ref = eg.Detector()
    ref.fit(healthy.features)
    sc = score_spectrum(control, None, ref)[w:]
    spec = np.mean([auc(sc, score_spectrum(f, None, ref)[w:]) for f in faults])
    print(f"memoryless 'spectrum' baseline: {spec:.3f}\n")

    n = len(CONFIGS) * len(RIDGES) * len(SEEDS)
    print(f"running {n} fits ({len(SEEDS)} random reservoirs per row)...\n")
    t0 = time.time()

    header = (f"{'config':<12}{'rho':>5}{'leak':>6}{'ridge':>8} |"
              f"{'mean':>7}{'std':>7}{'min':>7}{'max':>7}  vs baseline")
    print(header)
    print("-" * len(header))

    summary = []
    for rho, leak, label in CONFIGS:
        for ridge in RIDGES:
            aucs = np.array([mean_fault_auc(healthy, control, faults,
                                            rho, leak, ridge, s) for s in SEEDS])
            if aucs.min() > spec:
                verdict = "beats it on EVERY reservoir"
            elif aucs.mean() > spec:
                verdict = "beats it on average only"
            else:
                verdict = "does not beat it"
            summary.append((aucs.mean(), aucs.std(), aucs.min(), label, ridge))
            print(f"{label:<12}{rho:>5.2f}{leak:>6.2f}{ridge:>8g} |"
                  f"{aucs.mean():>7.3f}{aucs.std():>7.3f}"
                  f"{aucs.min():>7.3f}{aucs.max():>7.3f}  {verdict}")
        print()

    print(f"({time.time() - t0:.0f}s)")

    #most reliable row
    best = max(summary, key=lambda r: r[2])
    print(f"\nMost reliable setting: {best[3]}, ridge={best[4]:g}")
    print(f"  mean {best[0]:.3f} +/- {best[1]:.3f}, worst of 10 reservoirs {best[2]:.3f}")
    print(f"  memoryless baseline {spec:.3f}")


if __name__ == "__main__":
    main()
