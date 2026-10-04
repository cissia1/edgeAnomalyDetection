#include "mel_filters.h"     
#include "detector_weights.h"   

#if DET_N_IN != EG_N_MELS
#error "detector_weights.h and mel_filters.h disagree. Re-run export_detector.py."
#endif

const uint32_t SAMPLE_RATE    = EG_SAMPLE_RATE;
const int      PACKET_SAMPLES = EG_HOP;     
const int      N_FFT          = EG_N_FFT; 
const int      N_MELS         = EG_N_MELS;   
const int      N_RES          = DET_N_RES;   
const uint32_t PACKET_US      = 1000000UL * PACKET_SAMPLES / SAMPLE_RATE;   
const uint32_t BAUD           = 921600;
const int      LED_PIN        = 2;           
const int      BUTTON_PIN     = 0;         

uint8_t  audioPacket[4 + 2 * PACKET_SAMPLES + 2];
uint8_t  scorePacket[4 + 22 + 2];
uint8_t  restartPacket[4 + 2 + 2] = {0xA5, 0x5D, 0, 0, 'R', 0, 0, 0};
uint16_t seq = 0;
uint32_t nextSendUs;

int16_t history[N_FFT];     
bool    haveOnePacket = false;

float    re[N_FFT], im[N_FFT];
float    window[N_FFT];
float    twRe[N_FFT / 2], twIm[N_FFT / 2];
uint16_t bitRev[N_FFT];
float    melDb[N_MELS];

float    levelAvg   = 0.0f;   
uint32_t framesSeen = 0;
float    x[N_RES];             
float    xNew[N_RES];
float    u[N_MELS];            
float    pred[N_MELS];         
bool     havePred = false;

const float*   wIn  = DET_W_IN;
const float*   wVal = DET_W_VAL;
const float*   wOut = DET_W_OUT_T;
const det_col_t* wCol = DET_W_COL;
bool weightsInRam = false;

float phase1 = 0.0f, phase2 = 0.0f, phase3 = 0.0f;
const float TWO_PI_F = 2.0f * (float)PI;
const float STEP1 = TWO_PI_F * 440.0f  / SAMPLE_RATE;
const float STEP2 = TWO_PI_F * 1000.0f / SAMPLE_RATE;
const float STEP3 = TWO_PI_F * 2500.0f / SAMPLE_RATE;   
uint32_t rngState = 12345;

float noise() {
  rngState ^= rngState << 13;
  rngState ^= rngState >> 17;
  rngState ^= rngState << 5;
  return (float)(int32_t)rngState / 2147483648.0f;
}

void putChecksum(uint8_t* p, int len) {
  uint16_t sum = 0;
  for (int i = 4; i < len - 2; i++) sum += p[i];
  p[len - 2] = sum & 0xFF;
  p[len - 1] = sum >> 8;
}

void fft() {
  for (int i = 0; i < N_FFT; i++) {
    int j = bitRev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (int len = 2; len <= N_FFT; len <<= 1) {
    int half = len >> 1, step = N_FFT / len;
    for (int start = 0; start < N_FFT; start += len) {
      for (int j = 0; j < half; j++) {
        float wr = twRe[j * step], wi = twIm[j * step];
        int a = start + j, b = a + half;
        float tr = re[b] * wr - im[b] * wi;
        float ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;  im[b] = im[a] - ti;
        re[a] += tr;         im[a] += ti;
      }
    }
  }
}

void computeFeatures() {
  for (int n = 0; n < N_FFT; n++) {
    re[n] = (history[n] / 32768.0f) * window[n];   
    im[n] = 0.0f;
  }
  fft();
  for (int k = 0; k <= N_FFT / 2; k++) re[k] = re[k] * re[k] + im[k] * im[k];
  for (int m = 0; m < N_MELS; m++) {
    const float* w = &MEL_WEIGHTS[MEL_OFFSET[m]];
    float sum = 0.0f;
    for (int i = 0; i < MEL_LEN[m]; i++) sum += w[i] * re[MEL_START[m] + i];
    melDb[m] = 10.0f * log10f(sum + 1e-10f);
  }
}

void resetDetector() {
  levelAvg = 0.0f;
  framesSeen = 0;
  memset(x, 0, sizeof(x));
  havePred = false;
}

bool runDetector(float* spectrumScore, float* reservoirScore) {
  float level = 0.0f;
  for (int m = 0; m < N_MELS; m++) level += melDb[m];
  level /= N_MELS;
  framesSeen++;
  uint32_t n = framesSeen < DET_RUNNING_FRAMES ? framesSeen : DET_RUNNING_FRAMES;
  levelAvg += (level - levelAvg) / n;

  float spec = 0.0f, res = 0.0f;
  for (int m = 0; m < N_MELS; m++) {
    float v = (melDb[m] - levelAvg - DET_MU[m]) / DET_SIG[m];
    spec += v * v;
    if (havePred) { float e = pred[m] - v;  res += e * e; }
    u[m] = v;
  }
  *spectrumScore  = spec / N_MELS;
  *reservoirScore = res / N_MELS;

  for (int i = 0; i < N_RES; i++) {
    const float* w = wIn + i * N_MELS;
    float acc = 0.0f;
    for (int j = 0; j < N_MELS; j++) acc += w[j] * u[j];
    for (int k = DET_W_ROW[i]; k < DET_W_ROW[i + 1]; k++) acc += wVal[k] * x[wCol[k]];
    xNew[i] = (1.0f - DET_LEAK) * x[i] + DET_LEAK * tanhf(acc);
  }
  memcpy(x, xNew, sizeof(x));

  for (int j = 0; j < N_MELS; j++) {
    const float* w = wOut + j * DET_N_EXT;
    float acc = w[0];
    for (int m = 0; m < N_MELS; m++) acc += w[1 + m] * u[m];
    for (int i = 0; i < N_RES; i++)  acc += w[1 + N_MELS + i] * x[i];
    pred[j] = acc;
  }

  bool hadPrediction = havePred;
  havePred = true;
  return hadPrediction;
}

void* copyToRam(const void* src, size_t bytes) {
  void* p = malloc(bytes);
  if (p) memcpy(p, src, bytes);
  return p;
}

void put16(uint8_t* p, uint32_t v) { if (v > 65535) v = 65535;  p[0] = v & 0xFF;  p[1] = v >> 8; }

void sendScores(uint16_t frame, float spec, float res, uint32_t featUs, uint32_t detUs, uint8_t flags) {
  uint32_t id = DET_MODEL_ID;
  uint8_t* p = scorePacket;
  p[2] = frame & 0xFF;
  p[3] = frame >> 8;
  memcpy(p + 4,  &spec, 4);          
  memcpy(p + 8,  &res, 4);    
  memcpy(p + 12, &levelAvg, 4);
  memcpy(p + 16, &id, 4);       
  put16(p + 20, featUs);            
  put16(p + 22, detUs);       
  p[24] = flags;                 
  p[25] = 0;
  putChecksum(scorePacket, sizeof(scorePacket));
  Serial.write(scorePacket, sizeof(scorePacket));
}

void setup() {
  Serial.begin(BAUD);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  for (int n = 0; n < N_FFT; n++)
    window[n] = (float)(0.5 - 0.5 * cos(2.0 * PI * n / N_FFT));

  for (int k = 0; k < N_FFT / 2; k++) {
    twRe[k] = (float)cos(2.0 * PI * k / N_FFT);
    twIm[k] = (float)-sin(2.0 * PI * k / N_FFT);
  }

  int bits = 0;
  while ((1 << bits) < N_FFT) bits++;
  for (int i = 0; i < N_FFT; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++)
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    bitRev[i] = r;
  }
  
  void* a = copyToRam(DET_W_IN,    sizeof(DET_W_IN));
  void* b = copyToRam(DET_W_VAL,   sizeof(DET_W_VAL));
  void* c = copyToRam(DET_W_OUT_T, sizeof(DET_W_OUT_T));
  void* d = copyToRam(DET_W_COL,   sizeof(DET_W_COL));
  if (a && b && c && d) {
    wIn = (const float*)a;  wVal = (const float*)b;  wOut = (const float*)c;
    wCol = (const det_col_t*)d;
    weightsInRam = true;
  } else {
    free(a);  free(b);  free(c);  free(d);
  }

  resetDetector();
  audioPacket[0] = 0xA5;  audioPacket[1] = 0x5A;
  scorePacket[0] = 0xA5;  scorePacket[1] = 0x5C;
  nextSendUs = micros();
}

void loop() {
  bool restart = false;
  while (Serial.available()) if (Serial.read() == 'R') restart = true;
  if (restart) {
    seq = 0;
    haveOnePacket = false;
    resetDetector();
    putChecksum(restartPacket, sizeof(restartPacket));
    Serial.write(restartPacket, sizeof(restartPacket));    
    nextSendUs = micros();
  }

  while ((int32_t)(micros() - nextSendUs) < 0) { }
  nextSendUs += PACKET_US;
  
  bool fault = digitalRead(BUTTON_PIN) == LOW;      
  memmove(history, history + PACKET_SAMPLES, PACKET_SAMPLES * sizeof(int16_t));  
  for (int i = 0; i < PACKET_SAMPLES; i++) {
    float s = 0.30f * sinf(phase1) + 0.15f * sinf(phase2) + 0.05f * noise();
    if (fault) s += 0.08f * sinf(phase3);
    phase1 += STEP1;  if (phase1 >= TWO_PI_F) phase1 -= TWO_PI_F;
    phase2 += STEP2;  if (phase2 >= TWO_PI_F) phase2 -= TWO_PI_F;
    phase3 += STEP3;  if (phase3 >= TWO_PI_F) phase3 -= TWO_PI_F;

    int16_t v = (int16_t)(s * 32767.0f);
    history[PACKET_SAMPLES + i] = v;
    audioPacket[4 + 2 * i]     = v & 0xFF;
    audioPacket[4 + 2 * i + 1] = (v >> 8) & 0xFF;
  }
  audioPacket[2] = seq & 0xFF;
  audioPacket[3] = seq >> 8;
  putChecksum(audioPacket, sizeof(audioPacket));
  Serial.write(audioPacket, sizeof(audioPacket));

  if (haveOnePacket) {
    uint32_t t0 = micros();
    computeFeatures();
    uint32_t t1 = micros();
    float spec, res;
    bool haveScore = runDetector(&spec, &res);
    uint32_t t2 = micros();

    if (haveScore) {
      bool alarm = framesSeen > DET_WASHOUT && res > DET_THRESHOLD;
      digitalWrite(LED_PIN, alarm ? HIGH : LOW);
      uint8_t flags = (fault ? 1 : 0) | (alarm ? 2 : 0) | (weightsInRam ? 4 : 0);
      sendScores(seq - 1, spec, res, t1 - t0, t2 - t1, flags);   
    }
  }
  haveOnePacket = true;
  seq++;
}
