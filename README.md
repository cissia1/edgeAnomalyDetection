# edgeAnomalyDetection
This is a self-supervised machine-fault detector which uses a reservoir-computing (Echo State Network) predictor that learns a
machine's healthy acoustic signature with zero labeled fault data. I am building the DSP and ESN pipeline from scratch in
Python and am validating it on induced fan faults against healthy control.Furthermore, I'm designing the sensing hardware around an ESP32 and INMP441 I2S MEMS microphone, while building a physical test rig with DC fans and mechanically induced faults to generate a labeled validation set. I will be analyzing hardware test results through data visualization to tune signal-processing thresholds and improve
fault-detection sensitivity.

# How it works
Audio is resampled to 16 kHz, trimmed by one second at each end to remove handling noise, and split into 32 ms frames, each converted to a 32 band log-mel spectrum. Each recordings overall level is removed to cancel differences in microphone distance, and the bands are normalized using the healthy recordings statistics. The frames drive a fixed, sparse, randomly wired network of 200 neurons (the reservoir), and a linear readout trained only on healthy audio predicts the next frame from the reservoirs state. The anomaly score is the prediction error: healthy audio is predicted well, and audio from a changed machine is not. The alarm threshold is set from the last 30% of the healthy recording, which the readout never trains on.

# Running it
python -m pip install numpy scipy librosa matplotlib
python main.py --synthetic
python evaluate.py
python robustness.py
main.py holds the detector, evaluate.py scores every fault against a healthy control and compares the reservoir with three simpler baselines, robustness.py repeats that comparison across ten random reservoirs, and sweep.py explores the reservoirs memory settings. The scripts expect recordings/healthy.wav for training, recordings/healthy_2.wav as the healthy control, and any number of recordings/fault_*.wav files. Tested on Python 3.14.

# Week 1 results
The test faults were a partially blocked air intake, a loosened mounting nut, a wire touching the blades, and a RPM lower than intended. Results are reported as AUC, where 0.5 means the detector cant tell faults from healthy audio and 1.0 means it separates them perfectly.

Week 1 produced four findings. First, the faults arent loudness changes: once each recordings overall volume was removed, loudness alone scored at chance (0.49). Second, the reservoir needs minutes of training data. With about 20 seconds of healthy audio, results ranged from 0.46 to 0.97 depending on which random reservoir was drawn. But with four minutes, they held steady around 0.96, varying by only about 0.01 between reservoirs. Third, the reservoir showed no advantage over a simpler detector with no memory, which scored 0.986 on the same data. Fourth, and most important, microphone placement dominated everything. Changing how the phone touched the fan, lying across its full length in one recording and touching only one edge in another, made two recordings of the same healthy fan almost perfectly distinguishable (AUC 0.999), a bigger difference than any fault produced. Because every recording was a separate phone take, these results are consistent with fault detection working but cant prove it. Later I will address this with a microphone mounted permanently to the fan, so healthy and fault audio can be recorded through the same fixed sensor in one sitting. That dataset will also settle whether the reservoirs memory beats the simpler detector.

# Limitations
The dataset is small. I have one fan, four fault types, and a separate phone take for every recording. The alarm threshold, set four standard deviations above the healthy mean, can land too high because prediction errors are skewed.

# Development log
Each fix in Week 1 came from a check that exposed a problem. The threshold was originally set from the readouts error on its own training data, which is optimistically low, so it now comes from held-out healthy audio. Handling noise from starting and stopping each recording caused spikes at both ends, so one second is now trimmed from each end. Microphone distance shifted every frequency band and made a healthy recording look more anomalous than a real fault, so each recordings overall level is now removed before normalizing. The pipeline was moved into a Detector class so the evaluation scripts reuse exactly the same code, with identical output verified before and after the change.

Replaced score ratios with validated AUC and added a memoryless spectral baseline. A parameter sweep initially showed a 0.969 AUC reservoir configuration, but robustness testing across 10 random reservoirs revealed it was kinda lucky. Increasing training audio from one to four minutes reduced reservoir variation from +/-0.19 to +/-0.01. Increased washout from 20 to 60 frames to cover the ~1 second startup transient. Identified inconsistent phone placement as a source of variation. Again I have a fixed MEMS mount planned for later.
