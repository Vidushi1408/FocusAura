"""
app.py — IoT Health Monitor v3
Improved mood (DeepFace) + posture (MediaPipe)
Running on port 5001
"""
from flask import Flask, render_template, request, jsonify
import base64, numpy as np, cv2, requests, threading, time
from analyzer import (analyze_frame, DEEPFACE_OK, MEDIAPIPE_OK, HSE_OK,
                      start_calibration, clear_calibration, calibration_status)
from calculator import calc_stress, calc_productivity, should_alert
import database as db
from config import DEBUG, HOST

app  = Flask(__name__)
PORT = 5001

esp32_ip        = None
last_alert_time = {}
COOLDOWN        = 30
STALE_AFTER     = 15     # seconds before camera/sensor readings stop counting
_last_seen      = {'camera': 0, 'sensor': 0}
PERSIST_EVERY   = 5      # seconds between stress/productivity DB rows
_last_persist   = 0

state = dict(
    heart_rate=None, spo2=None, lux=None, fsr_raw=None, fsr_voltage=None,
    mood=None, mood_scores={}, mood_metrics={},
    posture_status=None, posture_score=None, posture_issues=[],
    posture_metrics={},
    stress_score=None, stress_level=None,
    productivity_score=None, productivity_label=None,
    face_detected=False, pose_detected=False,
    esp32_ip=None,
    mood_method='none', posture_method='none', ir=None, ir_ac=None,
    deepface_available=DEEPFACE_OK,
    mediapipe_available=MEDIAPIPE_OK
)

# ── Helpers ───────────────────────────────────────────────────────
def _send_esp32(payload):
    if not esp32_ip: return False
    try:
        r = requests.post(f'http://{esp32_ip}/alert', json=payload, timeout=2)
        return r.status_code == 200
    except: return False

def _fire_alerts(mood, posture, stress, hr, spo2):
    for a in should_alert(hr=hr, spo2=spo2, stress=stress, posture=posture):
        t = a['type']
        now = time.time()
        if now - last_alert_time.get(t, 0) < COOLDOWN: continue
        last_alert_time[t] = now
        sent = _send_esp32({'mood': mood, 'posture': posture,
                            'stress': stress, 'severity': a['severity']})
        db.ins_alert({**a, 'mood': mood, 'posture': posture,
                      'stress': stress, 'hr': hr, 'spo2': spo2, 'sent': sent})

def _fresh(kind):
    return time.time() - _last_seen[kind] < STALE_AFTER

def _recalc():
    # Ignore readings from a source that has gone quiet (camera stopped, ESP32 unplugged)
    cam, sen = _fresh('camera'), _fresh('sensor')
    g = lambda k, ok: state.get(k) if ok else None
    st = calc_stress(
        heart_rate=g('heart_rate', sen), spo2=g('spo2', sen),
        mood=g('mood', cam), posture_status=g('posture_status', cam),
        posture_score=g('posture_score', cam), fsr_raw=g('fsr_raw', sen)
    )
    pr = calc_productivity(
        posture_status=g('posture_status', cam),
        posture_score=g('posture_score', cam),
        mood=g('mood', cam), heart_rate=g('heart_rate', sen),
        stress_score=st['score'], fsr_raw=g('fsr_raw', sen)
    )
    state.update(stress_score=st['score'], stress_level=st['level'],
                 productivity_score=pr['score'], productivity_label=pr['label'])
    global _last_persist
    now = time.time()
    if now - _last_persist < PERSIST_EVERY:
        return
    _last_persist = now
    try:
        db.ins_stress(st)
        db.ins_productivity(pr)
    except Exception as e:
        print(f'[DB] persist failed: {e}')
    threading.Thread(target=_fire_alerts, daemon=True, args=(
        state.get('mood'), state.get('posture_status'),
        st['score'], state.get('heart_rate'), state.get('spo2')
    )).start()

# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/set-esp32', methods=['POST'])
def set_esp32():
    global esp32_ip
    ip = request.json.get('ip', '').strip()
    try:
        if requests.get(f'http://{ip}/ping', timeout=2).text == 'pong':
            esp32_ip = ip; state['esp32_ip'] = ip
            return jsonify({'status': 'connected', 'ip': ip})
    except: pass
    return jsonify({'status': 'unreachable'}), 400

@app.route('/sensor-data', methods=['GET', 'POST'])
def sensor_data():
    if request.method == 'POST':
        d = request.json or {}
        global esp32_ip
        # Auto-learn the device address from the IP the ESP32 reports about itself
        # (not request.remote_addr, which would also pick up curl tests / other hosts).
        if not esp32_ip and d.get('ip'):
            esp32_ip = d['ip']; state['esp32_ip'] = esp32_ip
        d.setdefault('ip', '')
        _last_seen['sensor'] = time.time()
        if d.get('spo2') is not None and not 70 <= d['spo2'] <= 100:
            d['spo2'] = None      # outside the physically plausible range = noise
        state.update(heart_rate=d.get('heart_rate'), spo2=d.get('spo2'),
                     lux=d.get('lux'), fsr_raw=d.get('fsr_raw'),
                     fsr_voltage=d.get('fsr_voltage'), ir=d.get('ir'), ir_ac=d.get('ir_ac'))
        try: db.ins_sensor(d)
        except: pass
        _recalc()
        return jsonify({'status': 'ok'})
    return jsonify({k: state.get(k) for k in
                    ['heart_rate', 'spo2', 'lux', 'fsr_raw', 'fsr_voltage']})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        raw = request.json.get('image', '')
        if ',' in raw: raw = raw.split(',')[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return jsonify({'error': 'bad image'}), 400

        res = analyze_frame(frame)
        _last_seen['camera'] = time.time()

        state.update(
            mood=res.get('mood'), mood_scores=res.get('mood_scores', {}),
            mood_metrics=res.get('mood_metrics', {}),
            posture_status=res.get('posture_status'),
            posture_score=res.get('posture_score'),
            posture_issues=res.get('posture_issues', []),
            posture_metrics=res.get('posture_metrics', {}),
            face_detected=res.get('face_detected', False),
            pose_detected=res.get('pose_detected', False),
            mood_method=res.get('mood_method', 'none'),
            posture_method=res.get('posture_method', 'none')
        )

        try:
            db.ins_mood(res)
            db.ins_posture(res)
        except: pass

        _recalc()

        res['stress_score']       = state.get('stress_score')
        res['stress_level']       = state.get('stress_level')
        res['productivity_score'] = state.get('productivity_score')
        res['productivity_label'] = state.get('productivity_label')
        res['esp32_connected']    = esp32_ip is not None
        res['deepface_active']    = DEEPFACE_OK
        res['mediapipe_active']   = MEDIAPIPE_OK
        return jsonify(res)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/calibrate', methods=['POST', 'DELETE'])
def calibrate():
    if request.method == 'DELETE':
        clear_calibration()
    else:
        start_calibration()
    return jsonify(calibration_status())

@app.route('/api/calibration')
def api_calibration(): return jsonify(calibration_status())

# ── API ───────────────────────────────────────────────────────────
@app.route('/api/state')
def api_state():
    # sensor_online: the ESP32 posted recently; the dashboard uses it to tell
    # "finger removed" (online, no reading -> show 0) from "device offline" (show --)
    return jsonify({**state, 'sensor_online': _fresh('sensor')})

@app.route('/api/system-info')
def api_system():
    return jsonify({
        'deepface': DEEPFACE_OK,
        'mediapipe': MEDIAPIPE_OK,
        'port': PORT,
        'hsemotion': HSE_OK,
        'mood_method': 'HSEmotion (AffectNet)' if HSE_OK else ('DeepFace' if DEEPFACE_OK else 'OpenCV Haar cascade'),
        'posture_method': 'MediaPipe Pose (33 landmarks)' if MEDIAPIPE_OK else 'OpenCV face position'
    })

@app.route('/api/mood-data')
def api_mood(): return jsonify(db.hist_mood(int(request.args.get('n', 60))))

@app.route('/api/posture-data')
def api_posture(): return jsonify(db.hist_posture(int(request.args.get('n', 60))))

@app.route('/api/stress-data')
def api_stress(): return jsonify(db.hist_stress(int(request.args.get('n', 60))))

@app.route('/api/productivity-data')
def api_prod(): return jsonify(db.hist_productivity(int(request.args.get('n', 60))))

@app.route('/api/history')
def api_history(): return jsonify(db.hist_sensor(int(request.args.get('n', 60))))

@app.route('/api/alerts')
def api_alerts(): return jsonify(db.hist_alerts(int(request.args.get('n', 30))))

@app.route('/api/alerts/ack/<int:aid>', methods=['POST'])
def ack(aid): db.ack_alert(aid); return jsonify({'ok': True})

@app.route('/api/hourly')
def api_hourly(): return jsonify(db.hourly_sensor(int(request.args.get('h', 24))))

@app.route('/api/hourly-stress')
def api_hourly_stress(): return jsonify(db.hourly_stress(int(request.args.get('h', 24))))

@app.route('/api/mood-dist')
def api_mood_dist(): return jsonify(db.mood_dist())

@app.route('/api/stats')
def api_stats(): return jsonify(db.stats_summary())

@app.route('/api/totals')
def api_totals(): return jsonify(db.dashboard_totals())

@app.route('/api/export/<table>')
def api_export(table):
    hours = int(request.args.get('hours', 24))
    csv_data = db.export_csv(table, hours)
    from flask import Response
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={table}_{hours}h.csv'}
    )

if __name__ == '__main__':
    print('\n' + '='*55)
    print(f'  IoT Health Monitor v3')
    print(f'  Mood detection  : {"HSEmotion (AffectNet)" if HSE_OK else ("DeepFace" if DEEPFACE_OK else "OpenCV Haar (fallback)")}')
    print(f'  Posture detect  : {"MediaPipe Pose 33-pt" if MEDIAPIPE_OK else "OpenCV face (fallback)"}')
    print(f'  Running at      : http://localhost:{PORT}')
    print('='*55 + '\n')
    app.run(debug=DEBUG, host=HOST, port=PORT)
