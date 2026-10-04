import argparse
import os
import wave
import numpy as np
import main as eg

MODEL = os.path.join("firmware", "esp_detector", "detector_model.npz")
TOLERANCE = 0.001


def load_wav(path):
    """Load a 16-bit WAV exactly the way librosa.load does: int16 / 32768."""
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float32) / 32768.0


def load_detector():
    """Rebuild the detector that export_detector.py put on the chip."""
    if not os.path.exists(MODEL):
        raise SystemExit(f"{MODEL} not found. Run export_detector.py first.")
    m = np.load(MODEL)
    for name in ("leak_rate", "running_frames", "reservoir_size"):
        if float(m[name]) != float(getattr(eg, name)):
            raise SystemExit(
                f"main.py's {name} is {getattr(eg, name)}, but the exported detector "
                f"used {float(m[name]):g}.\nRe-run export_detector.py and re-upload "
                "the firmware.")
    det = eg.Detector()
    det.mu, det.sig = m["mu"], m["sig"]
    det.W_in, det.W, det.W_out = m["W_in"], m["W"], m["W_out"]
    det.thresh = float(m["thresh"])
    return det, int(m["model_id"]), str(m["healthy"])


def compare(name, device, python, scale):
    diff = np.abs(device - python) / scale
    print(f"{name:<10} largest difference {100 * diff.max():.4f}%   "
          f"99th percentile {100 * np.percentile(diff, 99):.4f}%   "
          f"median {100 * np.median(diff):.4f}%")
    return diff.max()


def main():
    ap = argparse.ArgumentParser(description="Compare ESP32 anomaly scores with main.py")
    ap.add_argument("wav", help="recording made by capture.py with esp_detector running")
    args = ap.parse_args()

    det_path = os.path.splitext(args.wav)[0] + "_detector.npz"
    if not os.path.exists(det_path):
        raise SystemExit(f"{det_path} not found. Was esp_detector running "
                         "when you recorded?")
    d = np.load(det_path)
    det, model_id, healthy = load_detector()

    chip_ids = np.unique(d["model_id"])
    if len(chip_ids) != 1 or int(chip_ids[0]) != model_id:
        raise SystemExit(
            f"The chip is running detector 0x{int(chip_ids[0]):08X}, but the last one "
            f"exported is 0x{model_id:08X}.\nUpload the firmware again (it must be "
            "re-uploaded after every export_detector.py run).")
    if not bool(d["restarted"]) or int(d["first_audio_seq"]) != 0:
        raise SystemExit(
            "This recording did not start at the chip's frame 0, so the two "
            "calculations can't be lined up.\nRecord again with the current "
            "capture.py.")

    feat = eg.features_from_signal(load_wav(args.wav))
    py_res = det.score(feat)                                
    py_spec = np.mean(det.normalize(feat)[1:] ** 2, axis=1) 


    seq = d["frame_seq"]
    frames = seq[0] + np.concatenate([[0], np.cumsum(np.diff(seq) % 65536)])
    idx = frames - 1
    ok = idx < len(py_res)
    idx = idx[ok]
    dev_res, dev_spec = d["reservoir"][ok], d["spectrum"][ok]
    flags = d["flags"][ok]
    fault = (flags & 1) > 0
    settled = idx >= eg.washout     

    normal = ~fault if (~fault).any() else np.ones_like(fault)
    res_scale = np.maximum(py_res[idx], np.median(py_res[idx][normal]))
    spec_scale = np.maximum(py_spec[idx], np.median(py_spec[idx][normal]))

    total_us = d["feature_us"] + d["detector_us"]
    print(f"detector on chip   : 0x{model_id:08X}, trained on {healthy}")
    print(f"frames compared    : {len(idx)}  ({fault.sum()} with the fault button held)")
    print("difference from main.py, relative to the typical healthy score:")
    worst = max(compare("  spectrum", dev_spec, py_spec[idx], spec_scale),
                compare("  reservoir", dev_res, py_res[idx], res_scale))
    print(f"compute time       : features {d['feature_us'].mean():.0f} us + detector "
          f"{d['detector_us'].mean():.0f} us = {total_us.mean():.0f} us avg, "
          f"{total_us.max()} us max  (budget 16000 us)")

    alarm = (flags & 2) > 0           
    if fault.any():
        first, last = np.argmax(fault), len(fault) - 1 - np.argmax(fault[::-1])
        pos = np.arange(len(fault))
        parts = (("before the button", settled & (pos < first)),
                 ("button held", settled & fault),
                 ("after release", settled & (pos > last)))
    else:
        parts = (("whole recording", settled),)
    print(f"\nmean scores, main.py / chip      spectrum          reservoir      chip's alarm"
          f"  (threshold {det.thresh:.3f})")
    for label, sel in parts:
        if sel.any():
            print(f"  {label:<20} {py_spec[idx][sel].mean():>9.3f} / {dev_spec[sel].mean():<8.3f}"
                  f" {py_res[idx][sel].mean():>8.3f} / {dev_res[sel].mean():<8.3f}"
                  f" on {100 * alarm[sel].mean():5.1f}% of frames")

    if worst < TOLERANCE:
        print(f"\n=> MATCH: the ESP32 computes the same scores as main.py "
              f"(tolerance {100 * TOLERANCE:g}%)")
    else:
        print(f"\n=> MISMATCH: differences exceed {100 * TOLERANCE:g}%")


if __name__ == "__main__":
    main()
