import glob
import os
import time
import numpy as np
import main as eg
from evaluate import auc, Recording, score_spectrum

#The grid
RHOS   = [0.30, 0.50, 0.70, 0.85, 0.95, 0.99]
LEAKS  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]


def fit_detector(healthy_feat, rho, leak):
    """Build a detector with the given memory parameters. main.py reads rho and alpha from module-level variables, so we set them
    on the module before fitting. 
    """
    eg.spcrtl_radius = rho
    eg.leak_rate = leak
    det = eg.Detector()
    det.fit(healthy_feat)
    return det


def main():
    healthy_path = "recordings/healthy.wav"
    control_path = "recordings/healthy_2.wav"
    fault_paths = sorted(glob.glob("recordings/fault_*.wav"))
    if not fault_paths:
        raise SystemExit("no recordings/fault_*.wav files found")

    #Load every file 1 time
    print("loading recordings...")
    healthy = Recording(healthy_path)
    control = Recording(control_path)
    faults = [Recording(p) for p in fault_paths]
    fault_names = [f.name.replace("fault_", "") for f in faults]
    print(f"  healthy, control, and {len(faults)} fault(s)\n")

    #the memoryless baseline for reference
    ref_det = fit_detector(healthy.features, 0.95, 0.30)
    w = eg.washout
    spec_ctrl = score_spectrum(control, None, ref_det)[w:]
    spec_aucs = [auc(spec_ctrl, score_spectrum(f, None, ref_det)[w:]) for f in faults]
    spec_mean = float(np.mean(spec_aucs))

    #sweep 
    print(f"sweeping {len(RHOS)} x {len(LEAKS)} = {len(RHOS)*len(LEAKS)} settings...")
    t0 = time.time()

    results = np.zeros((len(RHOS), len(LEAKS), len(faults)))
    for i, rho in enumerate(RHOS):
        for j, leak in enumerate(LEAKS):
            det = fit_detector(healthy.features, rho, leak)
            ctrl = det.score(control.features)[w:]
            for k, f in enumerate(faults):
                results[i, j, k] = auc(ctrl, det.score(f.features)[w:])
        print(f"  rho={rho:.2f} done ({time.time()-t0:.0f}s)")

    mean_grid = results.mean(axis=2)

    #grid table
    print("\nMean AUC across all faults")
    print(f"(memoryless 'spectrum' baseline = {spec_mean:.3f} - beat this)\n")
    print("  rho \\ leak " + "".join(f"{l:>8.2f}" for l in LEAKS))
    print("  " + "-" * (11 + 8 * len(LEAKS)))
    for i, rho in enumerate(RHOS):
        row = f"  {rho:>9.2f} "
        for j in range(len(LEAKS)):
            v = mean_grid[i, j]
            mark = "*" if v > spec_mean else " "
            row += f"{v:>7.3f}{mark}"
        print(row)
    print("\n  * = beats the memoryless baseline")

    #optimistic best
    bi, bj = np.unravel_index(np.argmax(mean_grid), mean_grid.shape)
    best_rho, best_leak = RHOS[bi], LEAKS[bj]
    print(f"\nBest cell in the grid : rho={best_rho}, leak={best_leak}, "
          f"mean AUC {mean_grid[bi, bj]:.3f}")
    print("  (optimistic, this cell was chosen because it scored highest)")

    print("\n  per-fault at that setting:")
    for k, name in enumerate(fault_names):
        print(f"    {name:<20} reservoir {results[bi, bj, k]:.3f}"
              f"   spectrum {spec_aucs[k]:.3f}")

    #honest leave one fault out estimate
    honest = []
    for k, name in enumerate(fault_names):
        others = [x for x in range(len(faults)) if x != k]
        tuned = results[:, :, others].mean(axis=2) 
        ti, tj = np.unravel_index(np.argmax(tuned), tuned.shape)
        honest.append(results[ti, tj, k]) 
        print(f"\n  held out {name:<18} -> tuned rho={RHOS[ti]}, leak={LEAKS[tj]}"
              f"  -> AUC {results[ti, tj, k]:.3f}")

    print(f"\nHonest tuned performance (leave-one-fault-out): {np.mean(honest):.3f}")
    print(f"Memoryless spectrum baseline                  : {spec_mean:.3f}")
    print(f"Untuned default (rho=0.95, leak=0.30)         : "
          f"{mean_grid[RHOS.index(0.95), LEAKS.index(0.30)]:.3f}")

    verdict = ("temporal modelling helps" if np.mean(honest) > spec_mean + 0.01
               else "no temporal advantage on these faults")
    print(f"\n=> {verdict}")

    #heatmap
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(LEAKS)), [f"{l:.2f}" for l in LEAKS])
        ax.set_yticks(range(len(RHOS)), [f"{r:.2f}" for r in RHOS])
        ax.set_xlabel("leak rate (alpha)")
        ax.set_ylabel("spectral radius (rho)")
        ax.set_title(f"Mean AUC across faults (spectrum baseline = {spec_mean:.3f})")
        for i in range(len(RHOS)):
            for j in range(len(LEAKS)):
                ax.text(j, i, f"{mean_grid[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        fig.colorbar(im, label="mean AUC")
        plt.tight_layout()
        plt.savefig("sweep_heatmap.png", dpi=120)
        print("\nSaved heatmap to sweep_heatmap.png")
    except ImportError:
        print("\n(matplotlib not installed - skipping heatmap)")


if __name__ == "__main__":
    main()
