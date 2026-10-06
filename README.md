# FocusAura — IoT Health Monitor v3

An ESP32 with sensors plus a webcam feed into a Flask dashboard that estimates
**stress** and **productivity**, stores history in MySQL, and sends alerts back to the device buzzer.

## Architecture
- **ESP32** (`arduino/esp32_v3`): MAX30105 heart rate, BH1750 light, MPU6050 motion, FSR pressure.
  POSTs readings to `/sensor-data` every 3 s and exposes `GET /ping` and `POST /alert`.
- **Server** (`python/`): Flask on port 5001. `analyzer.py` (mood + posture), `calculator.py`
  (stress/productivity/alerts), `database.py` (MySQL).
- **Dashboard**: http://localhost:5001 — the browser captures a webcam frame every second and POSTs it to `/analyze`.

## Setup
1. **Secrets** (never committed):
   ```bash
   cp .env.example .env                                         # DB credentials
   cp arduino/esp32_v3/secrets.example.h arduino/esp32_v3/secrets.h   # Wi-Fi + server URL
   ```
2. **Install + database**
   ```bash
   bash install.sh
   ```
3. **Run**: `bash run.sh`, then open http://localhost:5001
4. **Firmware**: open `arduino/esp32_v3/esp32_v3.ino` in Arduino IDE, flash, and note the IP printed over serial.
   The server auto-learns the ESP32 address from its first reading; the Connect box is a fallback.

Configuration (env vars): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `FLASK_DEBUG` (default 0), `FLASK_HOST` (default 0.0.0.0).

## Detection
- **Mood**: HSEmotion (AffectNet-trained EfficientNet, ONNX) on the face nearest the centre of the frame, averaged over the last 5 frames.
  The ~16 MB model is downloaded once to `~/.hsemotion/`. DeepFace's emotion model was dropped as primary: in live webcam tests it
  returned a saturated 100% fear/disgust for a neutral face. DeepFace and an OpenCV Haar heuristic remain as fallbacks.
- **Posture**: MediaPipe Pose landmarks; each must have visibility > 0.5 *and* lie inside the frame. Spine lean is checked only when the hips are visible.
- **Calibration** (recommended): press **Calibrate** on the dashboard while sitting upright. Your baseline is saved to
  `python/calibration.json`, and later frames are judged by drift from it (head sinking toward shoulders, sitting lower than usual,
  too close / leaning back). Without a baseline, generic thresholds are used. `DELETE /calibrate` clears it.
- Check `/api/system-info` to see which models are active.

## Scoring
- **Stress** = weighted HR 30%, SpO2 20%, mood 25%, posture 15%, FSR 10%. Missing inputs (`null`/0) are skipped and weights renormalised.
- **Productivity** = posture + mood + HR stability + low stress + low FSR (0–100).
- **Alerts**: HR > 110, SpO2 < 95, stress ≥ 40, poor posture; 30 s cooldown per type. Buzzer: continuous = High, beeping = Medium.
- Stress/productivity rows are persisted at most every 5 s.

## Known limitations
- **SpO2 is an uncalibrated estimate** (red/IR ratio-of-ratios on a MAX30105, averaged over ~3 s windows, shown only while a pulse is tracked). Indicative only, not a medical measurement; values outside 70–100% are discarded.
- Posture thresholds are heuristic; a front-facing webcam cannot measure forward-head posture.
- Stress weights are not calibrated per user.

## API
`/api/state`, `/api/system-info`, `/sensor-data`, `/analyze`, `/api/history`, `/api/mood-data`,
`/api/posture-data`, `/api/stress-data`, `/api/productivity-data`, `/api/alerts`,
`/api/alerts/ack/<id>`, `/api/hourly`, `/api/hourly-stress`, `/api/mood-dist`, `/api/stats`,
`/api/totals`, `/api/export/<table>?hours=24`, `/calibrate` (POST start / DELETE clear), `/api/calibration`

## Dependency note
Do **not** `pip install onnx` or `hsemotion-onnx`: they pull numpy 2 and protobuf 6+, which break TensorFlow 2.15 and MediaPipe. Use `onnxruntime` only.

## Tests
```bash
pip install pytest && pytest tests
```
Live webcam check (guided, run in your own terminal): `.venv/bin/python tests/live_posture_check.py`
