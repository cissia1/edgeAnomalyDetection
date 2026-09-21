import argparse
import os
import wave
import numpy as np
import main as eg
 
# Largest acceptable difference, in dB. 
TOLERANCE_DB = 0.001
 
def load_wav(path):
    """Load a 16-bit WAV exactly the way librosa.load does: int16 / 32768."""
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(np.float32) / 32768.0
 
def main():
    ap = argparse.ArgumentParser(description="Compare ESP32 features with main.py")
    ap.add_argument("wav", help="recording made by capture.py with esp_features running")
    args = ap.parse_args()
 
    feat_path = os.path.splitext(args.wav)[0] + "_features.npz"
    if not os.path.exists(feat_path):
        raise SystemExit(f"{feat_path} not found. Was esp_features running "
                         "(not esp_stream) when you recorded?")
 
    d = np.load(feat_path)
    audio = load_wav(args.wav)
    reference = eg.features_from_signal(audio) 
 
    # ESP32 frame f starts at audio packet f
    rows = (d["frame_seq"] - int(d["first_audio_seq"])) % 65536
    usable = rows < len(reference)
    device = d["features"][usable]
    python = reference[rows[usable]]
    diff = np.abs(device - python)
 
    band_worst = diff.max(axis=0)
    us = d["compute_us"]
    print(f"frames compared    : {len(device)}")
    print(f"largest difference : {diff.max():.6f} dB")
    print(f"99th percentile    : {np.percentile(diff, 99):.6f} dB")
    print(f"median             : {np.median(diff):.6f} dB")
    print(f"worst band         : band {int(np.argmax(band_worst))} "
          f"({band_worst.max():.6f} dB)")
    print(f"feature values     : {python.min():.1f} to {python.max():.1f} dB")
    print(f"compute time       : avg {us.mean():.0f} us, max {us.max()} us per frame"
          f"  (budget 16000 us)")
 
    if diff.max() < TOLERANCE_DB:
        print(f"\n=> MATCH: the ESP32 computes the same features as main.py "
              f"(tolerance {TOLERANCE_DB} dB)")
    else:
        print(f"\n=> MISMATCH: differences exceed {TOLERANCE_DB} dB")
 
 
if __name__ == "__main__":
    main()
