const uint32_t SAMPLE_RATE    = 16000;
const int PACKET_SAMPLES       = 256;
const uint32_t PACKET_US      = 1000000UL * PACKET_SAMPLES / SAMPLE_RATE;
const uint32_t BAUD           = 921600;  

uint8_t  packet[4 + 2 * PACKET_SAMPLES + 2];
uint16_t seq = 0;
uint32_t nextSendUs;

float phase1 = 0.0f, phase2 = 0.0f;
const float TWO_PI_F = 2.0f * (float)PI;
const float STEP1 = TWO_PI_F * 440.0f  / SAMPLE_RATE;
const float STEP2 = TWO_PI_F * 1000.0f / SAMPLE_RATE;

void setup() {
  Serial.begin(BAUD);
  packet[0] = 0xA5;
  packet[1] = 0x5A;
  nextSendUs = micros();
}

void loop() {
  while ((int32_t)(micros() - nextSendUs) < 0) { }
  nextSendUs += PACKET_US;

  packet[2] = seq & 0xFF; 
  packet[3] = seq >> 8;

  for (int i = 0; i < PACKET_SAMPLES; i++) {
    float s = 0.30f * sinf(phase1) + 0.15f * sinf(phase2);
    phase1 += STEP1;  if (phase1 >= TWO_PI_F) phase1 -= TWO_PI_F;
    phase2 += STEP2;  if (phase2 >= TWO_PI_F) phase2 -= TWO_PI_F;

    int16_t v = (int16_t)(s * 32767.0f);
    packet[4 + 2 * i]     = v & 0xFF; 
    packet[4 + 2 * i + 1] = (v >> 8) & 0xFF;
  }

  uint16_t sum = 0;
  for (int i = 4; i < 4 + 2 * PACKET_SAMPLES; i++) sum += packet[i];
  packet[4 + 2 * PACKET_SAMPLES]     = sum & 0xFF;
  packet[4 + 2 * PACKET_SAMPLES + 1] = sum >> 8;

  Serial.write(packet, sizeof(packet));
  seq++;                                
}
