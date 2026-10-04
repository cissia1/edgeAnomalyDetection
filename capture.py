import argparse
import os
import struct
import time
import wave
import numpy as np
 
MAGIC = 0xA5
AUDIO, FEATURES, DETECTOR, RESTARTED = 0x5A, 0x5B, 0x5C, 0x5D
PACKET_SAMPLES = 256
N_MELS = 32
PACKET_BYTES = {
    AUDIO: 4 + 2 * PACKET_SAMPLES + 2,        
    FEATURES: 4 + 4 * N_MELS + 2 + 2,         
    DETECTOR: 4 + 22 + 2,                     
    RESTARTED: 4 + 2 + 2,                    
}
RESTART_WAIT = 2.0     
SAMPLE_RATE = 16000
BAUD = 921600
 
class PacketParser:
    def __init__(self):
        self.buf = bytearray()
        self.skipped_before_start = 0  
        self.skipped_after_start = 0 
        self.bad_checksums = {kind: 0 for kind in PACKET_BYTES}
        self.started = False
 
    def _length_at(self, i):
        """Packet length if a valid header starts at buf[i], else None."""
        if i + 1 < len(self.buf) and self.buf[i] == MAGIC:
            return PACKET_BYTES.get(self.buf[i + 1])
        return None
 
    def feed(self, data):
        self.buf.extend(data)
        packets = []
        while len(self.buf) >= 2:
            n = self._length_at(0)
            if n is not None:
                if len(self.buf) < n + 2:
                    break                        
                if self._length_at(n) is not None:     
                    kind = self.buf[1]
                    seq = self.buf[2] | (self.buf[3] << 8)
                    payload = bytes(self.buf[4:n - 2])
                    sent_sum = self.buf[n - 2] | (self.buf[n - 1] << 8)
                    del self.buf[:n]
                    self.started = True
                    if sum(payload) & 0xFFFF != sent_sum:
                        self.bad_checksums[kind] += 1 
                        continue
                    packets.append((kind, seq, payload))
                    continue
         
            i = self.buf.find(bytes([MAGIC]), 1)
            skip = i if i > 0 else len(self.buf)
            if self.started:
                self.skipped_after_start += skip
            else:
                self.skipped_before_start += skip
            del self.buf[:skip]
        return packets
 
 
def record(ser, seconds, clock=time.perf_counter, timeout=5.0):
    parser = PacketParser()
    target = int(seconds * SAMPLE_RATE)
    chunks, got = [], 0
    arrival_times, sample_counts = [], []
    dropped, last_seq, first_audio_seq = 0, None, None
    feats, feat_seqs, compute_us = [], [], []
    dropped_feats, last_feat = 0, None
    det_rows, dropped_det, last_det = [], 0, None
    started_waiting = clock()
 
    if hasattr(ser, "write"):
        ser.write(b"R")
    waiting_for_restart, restarted = True, False
 
    while got < target:
        data = ser.read(4096)
        if first_audio_seq is None and clock() - started_waiting > timeout:
            raise SystemExit(
                "No packets received. Check that the firmware is uploaded, the "
                "port is correct, and the Arduino Serial Monitor is closed.")
        if not data:
            continue
        for kind, seq, payload in parser.feed(data):
            if waiting_for_restart:
                if kind == RESTARTED:
                    waiting_for_restart, restarted = False, True
                    continue
                elif clock() - started_waiting > RESTART_WAIT:
                    waiting_for_restart = False
                else:
                    continue
            if kind == RESTARTED:
                continue
            if kind == AUDIO:
                if first_audio_seq is None:
                    first_audio_seq = seq
                if last_seq is not None:
                    dropped += (seq - last_seq - 1) % 65536  
                last_seq = seq
                chunks.append(np.frombuffer(payload, dtype="<i2").copy())
                got += PACKET_SAMPLES
                arrival_times.append(clock())
                sample_counts.append(got)
            elif kind == DETECTOR:
                if last_det is not None:
                    dropped_det += (seq - last_det - 1) % 65536
                last_det = seq
                det_rows.append((seq,) + struct.unpack("<fffIHHBx", payload))
            else:
                if last_feat is not None:
                    dropped_feats += (seq - last_feat - 1) % 65536
                last_feat = seq
                feats.append(np.frombuffer(payload[:4 * N_MELS], dtype="<f4").copy())
                feat_seqs.append(seq)
                compute_us.append(payload[4 * N_MELS] | (payload[4 * N_MELS + 1] << 8))
 
    audio = np.concatenate(chunks)[:target]
    slope = np.polyfit(sample_counts, arrival_times, 1)[0]
    stats = {
        "packets": len(chunks),
        "dropped": dropped,
        "bad_checksums": parser.bad_checksums[AUDIO],
        "corrupted_bytes": parser.skipped_after_start,
        "startup_bytes": parser.skipped_before_start,
        "rate": 1.0 / slope if slope > 0 else float("nan"),
        "feature_frames": len(feats),
        "dropped_features": dropped_feats,
        "bad_feature_checksums": parser.bad_checksums[FEATURES],
        "detector_frames": len(det_rows),
        "dropped_detector": dropped_det,
        "bad_detector_checksums": parser.bad_checksums[DETECTOR],
        "restarted": restarted,
    }
    features = None
    if feats:
        features = {
            "frame_seq": np.array(feat_seqs, dtype=np.int64),
            "features": np.array(feats, dtype=np.float32),
            "compute_us": np.array(compute_us, dtype=np.int64),
            "first_audio_seq": first_audio_seq,
        }
    detector = None
    if det_rows:
        cols = list(zip(*det_rows))
        detector = {
            "frame_seq": np.array(cols[0], dtype=np.int64),
            "spectrum": np.array(cols[1], dtype=np.float32),
            "reservoir": np.array(cols[2], dtype=np.float32),
            "level": np.array(cols[3], dtype=np.float32),
            "model_id": np.array(cols[4], dtype=np.uint32),
            "feature_us": np.array(cols[5], dtype=np.int64),
            "detector_us": np.array(cols[6], dtype=np.int64),
            "flags": np.array(cols[7], dtype=np.uint8),
            "first_audio_seq": first_audio_seq,
            "restarted": restarted,
        }
    return audio, features, detector, stats
 
 
def strongest_tones(audio, n=3, min_gap_hz=20.0, floor_db=-40.0):aa
    x = audio.astype(np.float64)
    x -= x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    spec[freqs < 20.0] = 0.0
    top = spec.max()
    peaks = []
    for _ in range(n):
        k = int(np.argmax(spec))
        rel_db = 20 * np.log10(spec[k] / top + 1e-20)
        if spec[k] <= 0 or rel_db < floor_db:
            break
        peaks.append((freqs[k], rel_db))
        spec[np.abs(freqs - freqs[k]) < min_gap_hz] = 0.0
    return peaks
 
 
def write_wav(path, audio):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype("<i2").tobytes())
 
 
def report(audio, features, detector, stats):
    x = audio.astype(np.float64) / 32768.0
    peak_db = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
    rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    rate_err = abs(stats["rate"] / SAMPLE_RATE - 1.0)
    tones = ", ".join(f"{f:.1f} Hz ({db:+.0f} dB)" for f, db in strongest_tones(audio))
 
    print(f"packets received : {stats['packets']}")
    print(f"dropped packets  : {stats['dropped']}")
    print(f"damaged packets  : {stats['bad_checksums']}")
    print(f"corrupted bytes  : {stats['corrupted_bytes']}"
          f"   (startup bytes skipped: {stats['startup_bytes']}, normal)")
    print(f"sample rate      : {stats['rate']:.1f} Hz  (expected {SAMPLE_RATE})")
    print(f"level            : peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS")
    print(f"strongest tones  : {tones}")
 
    clean = (stats["dropped"] == 0 and stats["bad_checksums"] == 0
             and stats["corrupted_bytes"] == 0 and rate_err < 0.005)
    if features is not None:
        us = features["compute_us"]
        print(f"feature frames   : {stats['feature_frames']}"
              f"  (dropped {stats['dropped_features']},"
              f" damaged {stats['bad_feature_checksums']})")
        print(f"feature compute  : avg {us.mean():.0f} us, max {us.max()} us per frame"
              f"  (budget {1e6 * PACKET_SAMPLES / SAMPLE_RATE:.0f} us)")
        clean = clean and stats["dropped_features"] == 0 \
            and stats["bad_feature_checksums"] == 0
    if detector is not None:
        budget = 1e6 * PACKET_SAMPLES / SAMPLE_RATE
        f_us, d_us = detector["feature_us"], detector["detector_us"]
        total = f_us + d_us
        fault = (detector["flags"] & 1) > 0
        res = detector["reservoir"]
        settled = np.arange(len(res)) >= 125    
        print(f"detector frames  : {stats['detector_frames']}"
              f"  (dropped {stats['dropped_detector']},"
              f" damaged {stats['bad_detector_checksums']})")
        print(f"compute per frame: features {f_us.mean():.0f} us + detector {d_us.mean():.0f} us"
              f" = {total.mean():.0f} us avg, {total.max()} us max"
              f"  ({100 * total.mean() / budget:.1f}% of the {budget:.0f} us budget)")
        print(f"weights held in  : {'RAM' if detector['flags'][0] & 4 else 'flash (no room in RAM)'}")
        print(f"started at frame 0: {'yes' if stats['restarted'] else 'NO - the chip did not restart'}")
        before = settled & ~fault
        if fault.any():
            before &= np.arange(len(res)) < np.argmax(fault)
        line = "reservoir score  :"
        if before.any():
            line += f" mean {res[before].mean():.3f} normally"
        if (settled & fault).any():
            line += (f", {res[settled & fault].mean():.3f} with the fault button held"
                     f" ({(settled & fault).sum()} frames)")
        print(line)
        clean = clean and stats["dropped_detector"] == 0 \
            and stats["bad_detector_checksums"] == 0
    print("\n=> CLEAN recording" if clean else
          "\n=> NOT CLEAN - do not use this recording for evaluation")
    return clean
 
 
def main():
    ap = argparse.ArgumentParser(description="Capture data streamed from the ESP32")
    ap.add_argument("--port", required=True, help="e.g. COM3")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="recordings_esp/test.wav")
    args = ap.parse_args()
 
    import serial
    try:
        ser = serial.Serial(args.port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        raise SystemExit(f"Could not open {args.port}: {e}\n"
                         "If it says 'Access is denied', close the Arduino "
                         "Serial Monitor and try again.")
 
    time.sleep(1.5)     
    ser.reset_input_buffer()
 
    print(f"recording {args.seconds:.0f} s from {args.port}...")
    audio, features, detector, stats = record(ser, args.seconds)
    ser.close()
 
    write_wav(args.out, audio)
    print(f"saved {args.out}")
    if features is not None:
        feat_path = os.path.splitext(args.out)[0] + "_features.npz"
        np.savez(feat_path, **features)
        print(f"saved {feat_path}")
    if detector is not None:
        det_path = os.path.splitext(args.out)[0] + "_detector.npz"
        np.savez(det_path, **detector)
        print(f"saved {det_path}")
    print()
    report(audio, features, detector, stats)
 
 
if __name__ == "__main__":
    main()
