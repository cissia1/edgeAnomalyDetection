const int LED_PIN = 2; 

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  delay(500);
  Serial.println("EchoGuard: ESP32 is alive");
}

void loop() {
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  Serial.print("uptime (ms): ");
  Serial.println(millis());
  delay(1000);
}
