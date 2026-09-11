# edgeAnomalyDetection
This is a self-supervised machine-fault detector which uses a reservoir-computing (Echo State Network) predictor that learns a
machine's healthy acoustic signature with zero labeled fault data. I am building the DSP and ESN pipeline from scratch in
Python and am validating it on induced fan faults against healthy control.Furthermore, I'm designing the sensing hardware around an ESP32 and INMP441 I2S MEMS microphone, while building a physical test rig with DC fans and mechanically induced faults to generate a labeled validation set. I will be analyzing hardware test results through data visualization to tune signal-processing thresholds and improve
fault-detection sensitivity.
