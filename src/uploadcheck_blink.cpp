#include <Arduino.h>

constexpr uint8_t kStatusLedPin = 14;

void setup() {
  pinMode(kStatusLedPin, OUTPUT);
}

void loop() {
  digitalWrite(kStatusLedPin, HIGH);
  delay(200);
  digitalWrite(kStatusLedPin, LOW);
  delay(200);
}
