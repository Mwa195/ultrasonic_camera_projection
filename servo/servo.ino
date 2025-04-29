#include <Servo.h>

Servo myServo;

#define trigPin 9
#define echoPin 10

int angle = 0;
float distance = 0, duration = 0;

void setup() {
  // put your setup code here, to run once:
  myServo.attach(11);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
  myServo.write(90);
  delay(3000);
}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 30; i <= 150; i += 5) {    myServo.write(i);
    angle = i;
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH);            // Measure the pulse duration
    distance = (duration * 0.0343) / 2.0;          // Convert time to distance in cm
    Serial.print(angle);
    Serial.print(",");
    Serial.println(distance);
    delay(100);
  }
  for (int i = 150; i >= 30; i -= 5) {
    myServo.write(i);
    angle = i;
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH);            // Measure the pulse duration
    distance = (duration * 0.0343) / 2.0;          // Convert time to distance in cm
    Serial.print(angle);
    Serial.print(",");
    Serial.println(distance);
    delay(100);
  }
}
