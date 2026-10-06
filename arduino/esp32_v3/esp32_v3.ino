#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <BH1750.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "secrets.h"

#define SDA_PIN 21
#define SCL_PIN 22
#define FSR_PIN 35
#define SPEAKER_PIN 18

const char* SSID          = WIFI_SSID;
const char* PASSWORD      = WIFI_PASSWORD;
const char* PYTHON_SERVER = SERVER_URL;

WebServer server(80);
unsigned long alertUntil = 0;
bool alertHigh = false;

BH1750 lightMeter;
MAX30105 particleSensor;
SemaphoreHandle_t i2c_mtx;

#define FINGER_IR_MIN 15000   // IR below this = no finger. With the 16384nA range a finger reads ~27k; empty sensor reads far lower
volatile long irRaw = 0;
// ── SpO2 (ratio-of-ratios from red + IR) ────────────────────────────
// R = (AC_red/DC_red) / (AC_ir/DC_ir);  SpO2 ≈ -45.060*R^2 + 30.354*R + 94.845 (Maxim reference fit).
// This is an uncalibrated estimate from a hobby sensor, not a medical measurement.
#define SPO2_WINDOW 300          // samples per estimate (~3 s at 100 samples/s)
float dcRed = 0, dcIr = 0;       // slow baselines (exponential moving average)
double sumSqRed = 0, sumSqIr = 0;
int spo2Samples = 0;
float spo2Hist[4]; byte spo2Count = 0, spo2Spot = 0;
volatile long irAC = 0;        // pulse amplitude (max-min over ~2s): needs to be >~100 for beat detection
long irWinMin = 999999, irWinMax = 0; unsigned long irWinStart = 0;       // latest IR reading, sent for debugging
float heartRate=0, spo2=0, lux=0;
int fsr=0;
float ax=0,ay=0,az=0,gx=0,gy=0,gz=0,temp=0;

bool maxOK=false, mpuOK=false, bhOK=false;
unsigned long lastSend=0;

const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
byte rateCount = 0;   // how many real beats are in rates[] (so empty slots don't drag the average down)
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;

void readPoxTask(void * parameter) {
  for(;;) {
    if(maxOK) {
      if(xSemaphoreTake(i2c_mtx, portMAX_DELAY)) {
        long irValue = particleSensor.getIR();
        irRaw = irValue;
        if (irValue >= FINGER_IR_MIN) {
          float r = particleSensor.getRed(), ir = irValue;
          if (dcIr == 0) { dcRed = r; dcIr = ir; }
          dcRed += (r - dcRed) * 0.02f;  dcIr += (ir - dcIr) * 0.02f;
          sumSqRed += (r - dcRed) * (r - dcRed);
          sumSqIr  += (ir - dcIr) * (ir - dcIr);
          if (++spo2Samples >= SPO2_WINDOW) {
            float acRed = sqrt(sumSqRed / spo2Samples), acIr = sqrt(sumSqIr / spo2Samples);
            float R = (acRed / dcRed) / (acIr / dcIr + 1e-6f);
            // accept only when a pulse is being tracked and the ratio is physically plausible
            if (heartRate > 0 && R > 0.3f && R < 1.5f) {
              float est = -45.060f * R * R + 30.354f * R + 94.845f;
              if (est > 100) est = 100;
              if (est >= 80) {
                spo2Hist[spo2Spot++] = est; spo2Spot %= 4; if (spo2Count < 4) spo2Count++;
                float sum = 0; for (byte i = 0; i < spo2Count; i++) sum += spo2Hist[i];
                spo2 = sum / spo2Count;      // average of the last few windows
              }
            }
            sumSqRed = sumSqIr = 0; spo2Samples = 0;
          }
        } else {
          spo2 = 0; spo2Count = 0; spo2Samples = 0; dcIr = 0; sumSqRed = sumSqIr = 0;   // finger off
        }
        if (irValue < irWinMin) irWinMin = irValue;
        if (irValue > irWinMax) irWinMax = irValue;
        if (millis() - irWinStart > 2000) { irAC = irWinMax - irWinMin; irWinMin = 999999; irWinMax = 0; irWinStart = millis(); }
        if (checkForBeat(irValue) == true) {
          long delta = millis() - lastBeat;
          lastBeat = millis();
          beatsPerMinute = 60 / (delta / 1000.0);
          if (beatsPerMinute < 180 && beatsPerMinute > 40) {   // plausible resting/active range only
            rates[rateSpot++] = (byte)beatsPerMinute;
            rateSpot %= RATE_SIZE;
            if (rateCount < RATE_SIZE) rateCount++;
            beatAvg = 0;
            for (byte x = 0 ; x < rateCount ; x++) beatAvg += rates[x];
            beatAvg /= rateCount;
            // show a value only after 3+ beats, so one noisy beat isn't reported
            heartRate = (rateCount >= 3) ? beatAvg : 0;
          }
        }
        if (millis() - lastBeat > 6000) { heartRate = 0; rateCount = 0; spo2 = 0; spo2Count = 0; }   // no beat for 6 s -> stale
        if (irValue < FINGER_IR_MIN) {
          heartRate = 0;
          rateCount = 0; rateSpot = 0;   // finger removed: start fresh next time
        }
        xSemaphoreGive(i2c_mtx);
      }
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

bool readMPU(){
  if(xSemaphoreTake(i2c_mtx, portMAX_DELAY)) {
    Wire.beginTransmission(0x68);
    if(Wire.endTransmission()!=0) { 
      xSemaphoreGive(i2c_mtx); 
      return false; 
    }
    Wire.beginTransmission(0x68);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(0x68,14,true);
    if(Wire.available()==14) {
      ax = (Wire.read()<<8|Wire.read())/16384.0;
      ay = (Wire.read()<<8|Wire.read())/16384.0;
      az = (Wire.read()<<8|Wire.read())/16384.0;
      temp = (Wire.read()<<8|Wire.read())/340.0+36.53;
      gx = (Wire.read()<<8|Wire.read())/131.0;
      gy = (Wire.read()<<8|Wire.read())/131.0;
      gz = (Wire.read()<<8|Wire.read())/131.0;
    }
    xSemaphoreGive(i2c_mtx);
    return true;
  }
  return false;
}

void sendData(){
  if(WiFi.status()!=WL_CONNECTED) return;
  HTTPClient http;
  http.setTimeout(2000);   // don't stall the loop (and the buzzer web server) if the server is down
  http.begin(String(PYTHON_SERVER)+"/sensor-data");
  http.addHeader("Content-Type","application/json");
  
  StaticJsonDocument<512> doc;
  // null (not 0) when a reading is unavailable so the server treats it as "no data"
  if(heartRate>0) doc["heart_rate"]=heartRate; else doc["heart_rate"]=nullptr;
  if(spo2>0) doc["spo2"]=spo2; else doc["spo2"]=nullptr;
  doc["lux"]=lux;
  doc["fsr_raw"]=fsr;
  doc["fsr_voltage"]=fsr/4095.0*3.3;
  doc["ip"]=WiFi.localIP().toString();
  doc["ir"]=irRaw;
  doc["ir_ac"]=irAC;
  doc["accel_x"]=ax;
  doc["accel_y"]=ay;
  doc["accel_z"]=az;
  doc["gyro_x"]=gx;
  doc["gyro_y"]=gy;
  doc["gyro_z"]=gz;
  doc["temperature"]=temp;
  
  String body;
  serializeJson(doc,body);
  int code = http.POST(body);
  Serial.printf("POST /sensor-data -> %d  IR=%ld AC=%ld HR=%.0f SpO2=%.1f\n", code, irRaw, irAC, heartRate, spo2);   // 200 = OK, negative = can't reach server
  http.end();
}

void handlePing(){ server.send(200, "text/plain", "pong"); }

void handleAlert(){
  StaticJsonDocument<256> doc;
  if(deserializeJson(doc, server.arg("plain"))){
    server.send(400, "text/plain", "bad json");
    return;
  }
  const char* sev = doc["severity"] | "Medium";
  alertHigh = (strcmp(sev, "High") == 0);
  alertUntil = millis() + (alertHigh ? 3000 : 1500);
  server.send(200, "text/plain", "ok");
}

void setup(){
  Serial.begin(115200);
  delay(1000);
  Serial.println("Booting System...");
  
  pinMode(SPEAKER_PIN, OUTPUT);
  noTone(SPEAKER_PIN);

  i2c_mtx = xSemaphoreCreateMutex();

  Wire.begin(SDA_PIN,SCL_PIN);
  Wire.setClock(100000);

  if(lightMeter.begin()){
    bhOK=true;
  }

  Wire.beginTransmission(0x68);
  if(Wire.endTransmission()==0){
    mpuOK=true;
  }

  if (particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    maxOK=true;
    particleSensor.setup(0x1F, 4, 2, 400, 411, 16384);   // 400Hz/avg4 = 100 samples/s to match checkForBeat's tuning; LED ~6mA + 16384nA avoids saturating
  }

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);

  // List what the ESP32 can actually see (it only supports 2.4 GHz networks)
  Serial.println("Scanning for Wi-Fi networks...");
  int n = WiFi.scanNetworks();
  bool found = false;
  for (int i = 0; i < n; i++) {
    Serial.printf("  %2d: %-32s ch%-2d %4d dBm %s\n", i + 1, WiFi.SSID(i).c_str(),
                  WiFi.channel(i), WiFi.RSSI(i), WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "open" : "secured");
    if (WiFi.SSID(i) == SSID) found = true;
  }
  Serial.printf(found ? "Target network \"%s\" FOUND\n"
                      : "Target network \"%s\" NOT FOUND (5 GHz only? out of range? hidden? wrong name?)\n", SSID);

  // Connect, retrying every 15 s, and say why it failed
  while (WiFi.status() != WL_CONNECTED) {
    Serial.printf("Connecting to \"%s\" ...\n", SSID);
    WiFi.disconnect(true);
    delay(200);
    WiFi.begin(SSID, PASSWORD);
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
      delay(500);
      Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() != WL_CONNECTED) {
      int st = WiFi.status();
      Serial.printf("Failed, status=%d (%s)\n", st,
        st == WL_NO_SSID_AVAIL ? "network not found" :
        st == WL_CONNECT_FAILED ? "wrong password / auth rejected" :
        st == WL_DISCONNECTED ? "disconnected / still trying" : "other");
    }
  }
  Serial.println("Wi-Fi connected");

  Serial.printf("Sensors: BH1750=%d MPU6050=%d MAX30105=%d\n", bhOK, mpuOK, maxOK);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  server.on("/ping", HTTP_GET, handlePing);
  server.on("/alert", HTTP_POST, handleAlert);
  server.begin();

  xTaskCreatePinnedToCore(readPoxTask, "PoxTask", 10000, NULL, 1, NULL, 0);
}

void loop(){
  if(bhOK) {
    if(xSemaphoreTake(i2c_mtx, portMAX_DELAY)) {
      float l=lightMeter.readLightLevel();
      if(l>0) lux=l;
      xSemaphoreGive(i2c_mtx);
    }
  }
  
  if(mpuOK) readMPU();
  
  fsr=analogRead(FSR_PIN);
  
  // Buzzer sounds only for server alerts: continuous tone = High, beeping = Medium
  server.handleClient();
  if (millis() < alertUntil) {
    if (alertHigh || (millis() / 250) % 2 == 0) tone(SPEAKER_PIN, 1000);
    else noTone(SPEAKER_PIN);
  } else {
    noTone(SPEAKER_PIN);
  }
  
  if(millis()-lastSend>3000){
    sendData();
    lastSend=millis();
  }
  
  delay(10);
}