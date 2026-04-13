/*
  Infrastructure Guardian — Arduino Sketch
  =========================================
  Sensors:
    • Piezoelectric  → pin A0  (ANALOG)  — normalized 0.0000-0.9999
    • Vibration SW420→ pin 2   (DIGITAL) — 0=still, 1=vibrating
    • Flame Sensor   → pin 4   (DIGITAL) — 0=fire!, 1=clear

  4-State Logic (V=Vibration, F=Flame):
    V=0, F=0  → GOOD       (all clear)
    V=1, F=1  → EARTHQUAKE (vibration only)
    V=0, F=1  → FIRE       (fire only)
    V=1, F=0  → EXTREME    (both)

  Piezo Health:
    >= 0.80  → GOOD
    0.60-0.79 → PROBLEM
    < 0.60   → CRITICAL

  Output format (every 500ms):
    Piezo:0.4523,Vibr:1,Fire:0
*/

const int PIN_PIEZO = A0;
const int PIN_VIBR  = 2;
const int PIN_FIRE  = 4;
const int LED_PIN   = 13;

void setup() {
  Serial.begin(9600);
  pinMode(PIN_VIBR, INPUT);
  pinMode(PIN_FIRE, INPUT);
  pinMode(LED_PIN, OUTPUT);

  // Discard first few ADC readings for stability
  for (int i = 0; i < 5; i++) {
    analogRead(PIN_PIEZO);
    delay(20);
  }
}

void loop() {
  // Piezo: analog → normalize
  int piezoRaw    = constrain(analogRead(PIN_PIEZO), 0, 1023);
  float piezoNorm = 1.0 - (piezoRaw / 1023.0);

  // Vibration: digital
  int vibrState = digitalRead(PIN_VIBR);  // 0=no vib, 1=vibrating

  // Flame: digital
  int fireState = digitalRead(PIN_FIRE);
  fireState = !fireState;

  // LED on when any alert
  digitalWrite(LED_PIN, (vibrState == 1 || fireState == 0) ? HIGH : LOW);

  // Send to Python
  Serial.print("Piezo:"); Serial.print(piezoNorm, 4);
  Serial.print(",Vibr:"); Serial.print(vibrState);
  Serial.print(",Fire:"); Serial.println(fireState);

  delay(500);
}
