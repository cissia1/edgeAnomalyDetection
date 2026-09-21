import argparse
import os
import time
import wave
import numpy as np

MAGIC = b"\xA5\x5A"
PACKET_SAMPLES = 256
PAYLOAD_END = 4 + 2 * PACKET_SAMPLES
PACKET_BYTES = PAYLOAD_END + 2
SAMPLE_RATE = 16000
BAUD = 921600

class PacketParser:
    def __init__(self):
        self.buf = bytearray()
        self.skipped_before_start = 0
        self.skipped_after_start = 0
        self.bad_checksums = 0 
        self.started = False

    def feed(self, data):
        self.buf.extend(data)
        packets = []
        while len(self.buf) >= PACKET_BYTES + 2:
            aligned = (self.buf[:2] == MAGIC and
                       self.buf[PACKET_BYTES:PACKET_BYTES + 2] == MAGIC)
            if aligned:
                payload = bytes(self.buf[4:PAYLOAD_END])
                sent_sum = self.buf[PAYLOAD_END] | (self.buf[PAYLOAD_END + 1] << 8)
                seq = self.buf[2] | (self.buf[3] << 8)
                del self.buf[:PACKET_BYTES]
                self.started = True
                if sum(payload) & 0xFFFF != sent_sum:
                    self.bad_checksums += 1    
                    continue
                samples = np.frombuffer(payload, dtype="<i2").copy()
                packets.append((seq, samples))
            else:
                i = self.buf.find(MAGIC, 1)
                skip = i if i > 0 else len(self.buf) - 1
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
    n_packets, dropped, last_seq = 0, 0, None
    first_t = None
    arrival_times, sample_counts = [], []
    started_waiting = clock()

    while got < target:
        data = ser.read(4096)
        if first_t is None and clock() - started_waiting > timeout:
            raise SystemExit(
        if not data:
            continue
        for seq, samples in parser.feed(data):
            now = clock()
            if first_t is None:
                first_t = now
            if last_seq is not None:
                dropped += (seq - last_seq - 1) % 65536   
            last_seq = seq
            chunks.append(samples)
            got += len(samples)
            n_packets += 1
            arrival_times.append(now)
            sample_counts.append(got)

    audio = np.concatenate(chunks)[:target]
    slope = np.polyfit(sample_counts, arrival_times, 1)[0]   
    rate = 1.0 / slope if slope > 0 else float("nan")
    stats = {
        "packets": n_packets,
        "dropped": dropped,
        "corrupted_bytes": parser.skipped_after_start,
        "bad_checksums": parser.bad_checksums,
        "startup_bytes": parser.skipped_before_start,
        "rate": rate,
    }
    return audio, stats


def strongest_tones(audio, n=3, min_gap_hz=20.0, floor_db=-40.0):
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


def report(audio, stats):
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
    print("\n=> CLEAN recording" if clean else
          "\n=> NOT CLEAN - do not use this recording for evaluation")
    return clean


def main():
    ap = argparse.ArgumentParser(description="Capture audio streamed from the ESP32")
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
    audio, stats = record(ser, args.seconds)
    ser.close()

    write_wav(args.out, audio)
    print(f"saved {args.out}\n")
    report(audio, stats)


if __name__ == "__main__":
    main()
