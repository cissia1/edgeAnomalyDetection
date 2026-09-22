#include "mel_filters.h"

const uint32_t SAMPLE_RATE    = EG_SAMPLE_RATE;
const int      PACKET_SAMPLES = EG_HOP; 
const int      N_FFT          = EG_N_FFT;
const int      N_MELS         = EG_N_MELS; 
const uint32_t PACKET_US      = 1000000UL * PACKET_SAMPLES / SAMPLE_RATE;   
const uint32_t BAUD           = 921600;

uint8_t  audioPacket[4 + 2 * PACKET_SAMPLES + 2];
uint8_t  featPacket[4 + 4 * N_MELS + 2 + 2];
uint16_t seq = 0;
uint32_t nextSendUs;

int16_t history[N_FFT];
bool    haveOnePacket = false;

float    re[N_FFT], im[N_FFT];
float    window[N_FFT];
float    twRe[N_FFT / 2], twIm[N_FFT / 2];
uint16_t bitRev[N_FFT];
float    melDb[N_MELS];

float phase1 = 0.0f, phase2 = 0.0f;
const float TWO_PI_F = 2.0f * (float)PI;
const float STEP1 = TWO_PI_F * 440.0f  / SAMPLE_RATE;
const float STEP2 = TWO_PI_F * 1000.0f / SAMPLE_RATE;
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

void setup() {
  Serial.begin(BAUD);

  // Periodic Hann window, identical to librosa's: 0.5 - 0.5 cos(2 pi n / N).
  // Computed in double precision, then stored as float, as numpy does.
  for (int n = 0; n < N_FFT; n++)
    window[n] = (float)(0.5 - 0.5 * cos(2.0 * PI * n / N_FFT));

  // FFT "twiddle factors": the complex rotations e^(-2 pi i k / N)
  for (int k = 0; k < N_FFT / 2; k++) {
    twRe[k] = (float)cos(2.0 * PI * k / N_FFT);
    twIm[k] = (float)-sin(2.0 * PI * k / N_FFT);
  }

  // bit-reversal table: for 512 points, index i's 9 bits reversed
  int bits = 0;
  while ((1 << bits) < N_FFT) bits++;
  for (int i = 0; i < N_FFT; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++)
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    bitRev[i] = r;
  }

  audioPacket[0] = 0xA5;  audioPacket[1] = 0x5A;
  featPacket[0]  = 0xA5;  featPacket[1]  = 0x5B;
  nextSendUs = micros();
}

void loop() {
  while ((int32_t)(micros() - nextSendUs) < 0) { }
  nextSendUs += PACKET_US;

  memmove(history, history + PACKET_SAMPLES, PACKET_SAMPLES * sizeof(int16_t));  
  for (int i = 0; i < PACKET_SAMPLES; i++) {
    float s = 0.30f * sinf(phase1) + 0.15f * sinf(phase2) + 0.05f * noise();
    phase1 += STEP1;  if (phase1 >= TWO_PI_F) phase1 -= TWO_PI_F;
    phase2 += STEP2;  if (phase2 >= TWO_PI_F) phase2 -= TWO_PI_F;

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
    uint32_t elapsed = micros() - t0;

    uint16_t frame = seq - 1;  
    featPacket[2] = frame & 0xFF;
    featPacket[3] = frame >> 8;
    memcpy(&featPacket[4], melDb, 4 * N_MELS);  
    uint16_t us = elapsed > 65535 ? 65535 : elapsed;
    featPacket[4 + 4 * N_MELS]     = us & 0xFF;
    featPacket[4 + 4 * N_MELS + 1] = us >> 8;
    putChecksum(featPacket, sizeof(featPacket));
    Serial.write(featPacket, sizeof(featPacket));
  }
  haveOnePacket = true;
  seq++;
}
