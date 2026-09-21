import argparse
import numpy as np

# parameters
sample_rate = 16000
frame_length = 32
hop_length = 16
num_mel = 32

reservoir_size = 200
spcrtl_radius = 0.95
leak_rate = 0.30
input_scale = 0.50
sparsity = 0.10

ridge_strength = 1e-6
washout = 60
smoothing = 5
threshold = 4.0          
seed = 42

def extract_features(wav_path):
    import librosa
    y, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    y = trim_edges(y)
    return features_from_signal(y)


def features_from_signal(y):
    import librosa
    n_fft = int(sample_rate * frame_length / 1000)
    hop = int(sample_rate * hop_length / 1000)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=n_fft,
                                         hop_length=hop, n_mels=num_mel)
    logmel = librosa.power_to_db(mel + 1e-10)
    return logmel.T


def build_reservoir(input_dim, rng):          
    W_in = rng.uniform(-1, 1, size=(reservoir_size, input_dim)) * input_scale
    W = rng.uniform(-1, 1, size=(reservoir_size, reservoir_size))

    drop = rng.uniform(0, 1, size=W.shape) > sparsity
    W[drop] = 0.0
    max_eig = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spcrtl_radius / max_eig
    return W_in, W

def run_reservoir(features, W_in, W):
    x = np.zeros(reservoir_size)
    states = np.empty((features.shape[0], reservoir_size))
    for t in range(features.shape[0]):
        u = features[t]
        x = (1 - leak_rate) * x + leak_rate * np.tanh(W_in @ u + W @ x)
        states[t] = x
    return states

def train_readout(states, features):
    Z = np.hstack([np.ones((states.shape[0], 1)), features, states])
    Z_in = Z[washout:-1]
    Y_next = features[washout + 1:]

    d = Z_in.shape[1]
    A = Z_in.T @ Z_in + ridge_strength * np.eye(d)
    B = Z_in.T @ Y_next
    W_out = np.linalg.solve(A, B)
    return W_out

def anomaly_score(states, features, W_out):
    Z = np.hstack([np.ones((states.shape[0], 1)), features, states])
    pred_next = Z[:-1] @ W_out
    true_next = features[1:]
    return np.mean((pred_next - true_next) ** 2, axis=1)

def moving_average(a, w):
    if w <= 1:
        return a
    return np.convolve(a, np.ones(w) / w, mode="same")

def synthetic_signal(kind, seconds=8.0):
    """Fake 'fan' audio so you can test with no recordings."""
    rng = np.random.default_rng(0 if kind == "healthy" else 1)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    sig = 0.30 * np.sin(2 * np.pi * 120 * t)
    sig += 0.15 * np.sin(2 * np.pi * 240 * t)
    sig += 0.05 * rng.standard_normal(t.shape)
    if kind == "faulty":
        thump = 0.25 * np.sin(2 * np.pi * 23 * t) * (np.sin(2 * np.pi * 23 * t) > 0.8)
        wobble = 0.12 * np.sin(2 * np.pi * 360 * t)
        sig += thump + wobble
    return sig.astype(np.float32)

def trim_edges(y, seconds=1.0):
    n = int(sample_rate * seconds)
    return y[n:-n] if len(y) > 2 * n else y

class Detector:
    """EchoGuard detector. fit() = learn what healthy sounds like. score() = per-frame anomaly score for new audio"""

    def fit(self, healthy_feat):
        # remove overall level
        healthy_feat = healthy_feat - healthy_feat.mean()
        self.mu = healthy_feat.mean(axis=0)
        self.sig = healthy_feat.std(axis=0) + 1e-8
        healthy_feat = (healthy_feat - self.mu) / self.sig

        rng = np.random.default_rng(seed)
        self.W_in, self.W = build_reservoir(healthy_feat.shape[1], rng)

        # train on the first 70%, hold out the last 30% for calibration
        split = int(0.7 * len(healthy_feat))
        train_feat = healthy_feat[:split]
        cal_feat = healthy_feat[split:]

        train_states = run_reservoir(train_feat, self.W_in, self.W)
        self.W_out = train_readout(train_states, train_feat)

        # threshold comes from healthy data only
        self.cal_err = self._score_normalized(cal_feat)
        base = self.cal_err[washout:]
        self.thresh = base.mean() + threshold * base.std()

    def normalize(self, feat):
        """Apply the exact normalization used during fit() to new audio."""
        feat = feat - feat.mean()
        return (feat - self.mu) / self.sig

    def score(self, feat):
        """Per-frame anomaly score for new (un-normalized) features."""
        return self._score_normalized(self.normalize(feat))

    def _score_normalized(self, feat):
        states = run_reservoir(feat, self.W_in, self.W)
        return anomaly_score(states, feat, self.W_out)


def main():
    ap = argparse.ArgumentParser(description="EchoGuard Week 1 offline prototype")
    ap.add_argument("--healthy", help="path to healthy-machine audio")
    ap.add_argument("--test", help="path to test audio (may contain a fault)")
    ap.add_argument("--synthetic", action="store_true",
                    help="ignore files and use built-in fake audio")

    args = ap.parse_args()

    if args.synthetic:
        print("Using synthetic audio (no files needed).")
        healthy_feat = features_from_signal(synthetic_signal("healthy"))
        test_feat = features_from_signal(synthetic_signal("faulty"))
    else:
        if not args.healthy or not args.test:
            ap.error("provide --healthy and --test, or use --synthetic")
        print(f"Loading healthy audio: {args.healthy}")
        healthy_feat = extract_features(args.healthy)
        print(f"Loading test audio:    {args.test}")
        test_feat = extract_features(args.test)

    detector = Detector()
    detector.fit(healthy_feat)
    test_err = detector.score(test_feat)

    cal_err = detector.cal_err
    base = cal_err[washout:]
    thresh = detector.thresh

    h_s = moving_average(cal_err, smoothing)
    t_s = moving_average(test_err, smoothing)       

    #report
    pct_flagged = 100.0 * np.mean(t_s > thresh)
    test_mean = test_err[washout:].mean()
    print("\n--- results ---")
    print(f"healthy mean score : {base.mean():.4f}")
    print(f"test    mean score : {test_mean:.4f}")
    print(f"threshold          : {thresh:.4f}")
    print(f"test frames flagged: {pct_flagged:.1f}%")
    if test_mean > base.mean():
        ratio = test_mean / base.mean()
        verdict = "Clear separation!" if ratio > 1.5 else "Some separation."
        print(f"=> test audio scores {ratio:.1f}x higher than healthy.  {verdict}")

    #plot
    try:
        import matplotlib.pyplot as plt
        dt = hop_length / 1000.0
        plt.figure(figsize=(11, 4))
        plt.plot(np.arange(len(h_s)) * dt, h_s, label="healthy", lw=1.2)
        plt.plot(np.arange(len(t_s)) * dt, t_s, label="test (may have fault)", lw=1.2)
        plt.axhline(thresh, ls="--", color="red", lw=1, label="threshold")
        plt.xlabel("time (s)")
        plt.ylabel("anomaly score (prediction error)")
        plt.title("EchoGuard - anomaly score over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("echoguard_scores.png", dpi=120)
        print("\nSaved plot to echoguard_scores.png")
        plt.show()
    except ImportError:
        print("(matplotlib not installed - skipping plot)")


if __name__ == "__main__":
    main()
