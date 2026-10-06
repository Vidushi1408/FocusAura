"""
analyzer.py — Improved Mood + Posture Detection
================================================
Mood    : DeepFace (AffectNet / VGG-Face2 backend)
          Falls back to OpenCV Haar if DeepFace unavailable
Posture : MediaPipe Pose (33 body landmarks — Google pre-trained)
          Checks head tilt, neck angle, shoulder level, spine lean
"""

import os
import cv2
import numpy as np
import math
import warnings
from collections import deque
warnings.filterwarnings("ignore")

# ── Try importing DeepFace ────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    print("[Analyzer] DeepFace loaded — using deep learning emotion model")
except ImportError:
    DEEPFACE_OK = False
    print("[Analyzer] DeepFace not found — falling back to OpenCV Haar cascade")

# ── Try importing MediaPipe ───────────────────────────────────────
try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose
    _pose = _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,          # 0=lite 1=full 2=heavy — use 2 for best accuracy
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_OK = True
    print("[Analyzer] MediaPipe Pose loaded — using 33-landmark model")
except Exception as e:
    MEDIAPIPE_OK = False
    print(f"[Analyzer] MediaPipe not available: {e} — using face-based posture")

# ── HSEmotion ONNX (AffectNet-trained EfficientNet) — primary mood model ──────
# Replaces DeepFace's FER model, which returned saturated 100% labels (fear /
# disgust) for a neutral face in live webcam tests.
HSE_MODEL = os.path.expanduser('~/.hsemotion/enet_b0_8_best_vgaf.onnx')
HSE_URL = ('https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/'
           'models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx?raw=true')
try:
    import onnxruntime as ort
    if not os.path.isfile(HSE_MODEL):
        import urllib.request
        os.makedirs(os.path.dirname(HSE_MODEL), exist_ok=True)
        print("[Analyzer] Downloading HSEmotion model (~16MB)...")
        urllib.request.urlretrieve(HSE_URL, HSE_MODEL)
    _hse = ort.InferenceSession(HSE_MODEL, providers=['CPUExecutionProvider'])
    _face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    HSE_OK = True
    print("[Analyzer] HSEmotion loaded — using AffectNet EfficientNet for mood")
except Exception as e:
    HSE_OK = False
    print(f"[Analyzer] HSEmotion not available: {e}")

# ── OpenCV fallback cascade ───────────────────────────────────────
_face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
_eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Emotion label map — DeepFace uses these exact keys
EMOTION_MAP = {
    'happy':    'happy',
    'sad':      'sad',
    'angry':    'stressed',
    'disgust':  'stressed',
    'fear':     'stressed',
    'surprise': 'surprised',
    'neutral':  'neutral'
}

MOOD_LABELS = ['happy', 'sad', 'stressed', 'surprised', 'neutral']

# Rolling window of recent mood scores — stops the label flickering frame to frame
_mood_history = deque(maxlen=5)

def _smooth_mood(scores):
    _mood_history.append(scores)
    avg = {k: sum(h.get(k, 0) for h in _mood_history) / len(_mood_history)
           for k in MOOD_LABELS}
    return max(avg, key=avg.get), {k: round(v, 1) for k, v in avg.items()}


# ══════════════════════════════════════════════════════════════════
# MOOD DETECTION
# ══════════════════════════════════════════════════════════════════

_HSE_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_HSE_MAP = {'anger': 'stressed', 'disgust': 'stressed', 'fear': 'stressed',
            'contempt': 'neutral', 'happy': 'happy', 'neutral': 'neutral',
            'sad': 'sad', 'surprise': 'surprised'}


def _mood_hse(frame):
    """Mood from the face nearest the centre of the frame."""
    try:
        h, w = frame.shape[:2]
        dets = _face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).detections or []
        if not dets:
            return None, {}, {'method': 'hsemotion', 'error': 'no face'}, False
        # prefer the central face, ignoring much smaller background faces
        boxes = [d.location_data.relative_bounding_box for d in dets]
        biggest = max(b.width * b.height for b in boxes)
        b = min((b for b in boxes if b.width * b.height >= 0.4 * biggest),
                key=lambda b: abs(b.xmin + b.width / 2 - 0.5))
        x, y, bw, bh = b.xmin * w, b.ymin * h, b.width * w, b.height * h
        x0, y0 = int(max(0, x - bw * .15)), int(max(0, y - bh * .15))
        x1, y1 = int(min(w, x + bw * 1.15)), int(min(h, y + bh * 1.15))
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        if crop.size == 0:
            return None, {}, {'method': 'hsemotion', 'error': 'bad crop'}, False

        x_in = cv2.resize(crop, (224, 224)) / 255.0
        x_in = (x_in - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        x_in = x_in.transpose(2, 0, 1).astype('float32')[None]
        lg = _hse.run(None, {'input': x_in})[0][0]
        p = np.exp(lg - lg.max()); p /= p.sum()

        scores = {k: 0.0 for k in MOOD_LABELS}
        for name, prob in zip(_HSE_CLASSES, p):
            scores[_HSE_MAP[name]] += float(prob) * 100
        scores = {k: round(v, 1) for k, v in scores.items()}
        mood, scores = _smooth_mood(scores)
        metrics = {'method': 'hsemotion',
                   'raw': {n: round(float(v) * 100, 1) for n, v in zip(_HSE_CLASSES, p)},
                   'face_x': int(x), 'face_y': int(y), 'face_w': int(bw), 'face_h': int(bh),
                   'faces_in_frame': len(dets)}
        return mood, scores, metrics, True
    except Exception as e:
        return None, {}, {'error': str(e), 'method': 'hsemotion_failed'}, False


def _mood_deepface(frame):
    """Use DeepFace with AffectNet model for best accuracy."""
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',   # fast and reliable
            silent=True
        )
        # DeepFace returns one result per face. With several people in frame, pick the
        # one closest to the image centre (the person at the desk), ignoring tiny
        # background faces, instead of blindly taking the first.
        faces = results if isinstance(results, list) else [results]
        fh, fw = frame.shape[:2]
        def _area(f): return f.get('region', {}).get('w', 0) * f.get('region', {}).get('h', 0)
        def _off(f):
            rg = f.get('region', {})
            return abs(rg.get('x', 0) + rg.get('w', 0) / 2 - fw / 2)
        biggest = max(_area(f) for f in faces)
        r = min((f for f in faces if _area(f) >= 0.4 * biggest), key=_off)
        raw_emotions = r.get('emotion', {})
        region = r.get('region', {})

        # enforce_detection=False returns the whole frame as the "face" when none
        # is found — treat that as no face rather than a bogus emotion.
        if (region.get('w', 0) >= 0.95 * fw and region.get('h', 0) >= 0.95 * fh) \
                or not region.get('w'):
            return None, {}, {'method': 'deepface', 'error': 'no face'}, False

        # Remap DeepFace labels → our 5 labels
        scores = {k: 0.0 for k in MOOD_LABELS}
        for df_label, pct in raw_emotions.items():
            our_label = EMOTION_MAP.get(df_label.lower(), 'neutral')
            scores[our_label] = scores.get(our_label, 0.0) + pct

        # Normalise to 100%
        total = sum(scores.values()) + 1e-6
        scores = {k: round(v / total * 100, 1) for k, v in scores.items()}
        mood, scores = _smooth_mood(scores)

        metrics = {
            'method': 'deepface',
            'raw': {k: round(v, 1) for k, v in raw_emotions.items()},
            'face_x': region.get('x', 0),
            'face_y': region.get('y', 0),
            'face_w': region.get('w', 0),
            'face_h': region.get('h', 0)
        }
        return mood, scores, metrics, True

    except Exception as e:
        return None, {}, {'error': str(e), 'method': 'deepface_failed'}, False


def _mood_opencv_fallback(frame):
    """OpenCV Haar cascade fallback when DeepFace is unavailable."""
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape

    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if not len(faces):
        return None, {}, {'method': 'opencv_haar', 'error': 'no face'}, False

    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    fg = gray[fy:fy + fh, fx:fx + fw]

    eyes = _eye_cascade.detectMultiScale(fg, 1.1, 5, minSize=(18, 18))
    smiles = _smile_cascade.detectMultiScale(fg, 1.7, 22, minSize=(25, 15))
    smile = len(smiles) > 0
    ec = len(eyes)

    bright = float(np.mean(fg))
    cont = float(np.std(fg))
    brow = fg[:fh // 4, :]
    bed = float(np.sum(cv2.Canny(brow, 50, 150) > 0)) / (brow.size + 1e-6)

    sc = {}
    sc['happy']     = min(1.0, (0.6 if smile else 0) + (0.2 if ec >= 2 else 0) + max(0, (bright - 100) / 200) * 0.2)
    sc['surprised'] = min(1.0, ((0.3 if ec >= 2 else 0) + max(0, (float(np.mean(fg[fh//2:,:])) - float(np.mean(fg[:fh//2,:]))) / 60) * 0.5) * 0.8)
    sc['sad']       = min(1.0, (max(0, (120 - bright) / 120) * 0.5 + (0.3 if ec < 2 else 0) + (0 if smile else 0.2)) * 0.7)
    sc['stressed']  = min(1.0, (bed * 4 + (0.3 if ec >= 1 and not smile else 0) + max(0, (cont - 40) / 80) * 0.3) * 0.9)
    sc['neutral']   = max(0, 0.5 - max(sc.values()) * 0.8)

    total = sum(sc.values()) + 1e-6
    sc = {k: round(v / total * 100, 1) for k, v in sc.items()}
    mood, sc = _smooth_mood(sc)
    return mood, sc, {'method': 'opencv_haar', 'brightness': round(bright, 1),
                      'eyes': int(ec), 'smile': smile}, True


def detect_mood(frame):
    if HSE_OK:
        mood, scores, metrics, ok = _mood_hse(frame)
        if ok:
            return mood, scores, metrics
        if metrics.get('error') == 'no face':
            return None, {}, metrics      # don't invent a mood from a faceless frame
    if DEEPFACE_OK:
        mood, scores, metrics, ok = _mood_deepface(frame)
        if ok:
            return mood, scores, metrics
        # DeepFace failed (no face found etc.) — try fallback
    mood, scores, metrics, ok = _mood_opencv_fallback(frame)
    return mood, scores, metrics


# ══════════════════════════════════════════════════════════════════
# POSTURE DETECTION — MediaPipe 33-landmark model
# ══════════════════════════════════════════════════════════════════

def _angle_3pts(a, b, c):
    """Angle at point b formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _pt(lm, idx, w, h):
    p = lm[idx]
    return [p.x * w, p.y * h, p.visibility]


# ── Personal calibration ──────────────────────────────────────────
# Absolute thresholds can't tell a slouch from a person who just sits close or
# low in frame (live test: a slumped pose still scored 100). Calibration stores
# the user's own upright metrics; later frames are judged by drift from them.
import json, threading
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')
CALIB_KEYS = ['eye_to_shoulder_pct', 'nose_y_pct', 'shoulder_w_pct',
              'shoulder_tilt_pct', 'eye_tilt_pct', 'neck_angle_deg']
_calib_lock = threading.Lock()
_calib_samples = None      # list while collecting, else None
_calib_target = 0

def _load_baseline():
    try:
        with open(CALIB_FILE) as f:
            return json.load(f)
    except Exception:
        return None

BASELINE = _load_baseline()

def start_calibration(n=8):
    global _calib_samples, _calib_target
    with _calib_lock:
        _calib_samples, _calib_target = [], n

def clear_calibration():
    global BASELINE, _calib_samples
    with _calib_lock:
        BASELINE, _calib_samples = None, None
    try: os.remove(CALIB_FILE)
    except FileNotFoundError: pass

def calibration_status():
    with _calib_lock:
        return {'calibrated': BASELINE is not None,
                'collecting': _calib_samples is not None,
                'progress': len(_calib_samples) if _calib_samples is not None else 0,
                'target': _calib_target,
                'baseline': BASELINE}

def _calib_feed(metrics, issues):
    """Collect upright samples; finish and persist when enough are in."""
    global BASELINE, _calib_samples
    with _calib_lock:
        if _calib_samples is None or 'eye_to_shoulder_pct' not in metrics:
            return
        # only accept frames the user is actually sitting upright in
        if any('tilt' in i or 'leaning' in i for i in issues):
            return
        _calib_samples.append({k: metrics[k] for k in CALIB_KEYS if k in metrics})
        if len(_calib_samples) >= _calib_target:
            base = {k: round(float(np.median([x[k] for x in _calib_samples if k in x])), 2)
                    for k in CALIB_KEYS if any(k in x for x in _calib_samples)}
            BASELINE, _calib_samples = base, None
            with open(CALIB_FILE, 'w') as f:
                json.dump(base, f, indent=2)


def _posture_mediapipe(frame):
    """
    Full 33-landmark posture analysis.
    Checks: neck angle, shoulder tilt, spine lean, head forward position.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = _pose.process(rgb)

    if not res.pose_landmarks:
        return None, None, [], {}, False

    lm = res.pose_landmarks.landmark
    PL = _mp_pose.PoseLandmark

    issues = []
    metrics = {}

    # Extract key points [x_px, y_px, visibility]
    nose       = _pt(lm, PL.NOSE, w, h)
    l_ear      = _pt(lm, PL.LEFT_EAR, w, h)
    r_ear      = _pt(lm, PL.RIGHT_EAR, w, h)
    l_shoulder = _pt(lm, PL.LEFT_SHOULDER, w, h)
    r_shoulder = _pt(lm, PL.RIGHT_SHOULDER, w, h)
    l_hip      = _pt(lm, PL.LEFT_HIP, w, h)
    r_hip      = _pt(lm, PL.RIGHT_HIP, w, h)
    l_eye      = _pt(lm, PL.LEFT_EYE, w, h)
    r_eye      = _pt(lm, PL.RIGHT_EYE, w, h)

    # Midpoints
    ear_mid  = [(l_ear[0] + r_ear[0]) / 2,  (l_ear[1] + r_ear[1]) / 2]
    sh_mid   = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
    hip_mid  = [(l_hip[0] + r_hip[0]) / 2,  (l_hip[1] + r_hip[1]) / 2]
    eye_mid  = [(l_eye[0] + r_eye[0]) / 2,  (l_eye[1] + r_eye[1]) / 2]

    VIS = 0.5
    # MediaPipe reports high visibility even for points it extrapolates outside the
    # frame, so a landmark only counts if it is also inside the image.
    def seen(*pts):
        return all(p[2] >= VIS and 0 <= p[0] <= w and 0 <= p[1] <= h for p in pts)

    if not seen(l_shoulder, r_shoulder):
        return None, None, [], {}, False   # can't judge posture without shoulders

    penalty = 0

    # ── 1. Neck / head lean: craniovertebral-style angle, ear→shoulder vs vertical
    if seen(l_ear, r_ear):
        dx = ear_mid[0] - sh_mid[0]
        dy = sh_mid[1] - ear_mid[1]
        neck_angle = math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))
        metrics['neck_angle_deg'] = round(neck_angle, 1)
        if neck_angle > 25:
            issues.append(f'Head leaning ({neck_angle:.0f}° off vertical)')
            penalty += 25

    # ── 2. Shoulder tilt ─────────────────────────────────────────
    sh_width = abs(l_shoulder[0] - r_shoulder[0]) + 1e-6
    sh_tilt_pct = abs(l_shoulder[1] - r_shoulder[1]) / sh_width * 100
    metrics['shoulder_tilt_pct'] = round(sh_tilt_pct, 1)
    if sh_tilt_pct > 12:
        issues.append(f'Shoulders uneven ({sh_tilt_pct:.0f}%)')
        penalty += 20

    # ── 3. Spine lean (only when hips are actually in frame) ─────
    if seen(l_hip, r_hip):
        sdx = sh_mid[0] - hip_mid[0]
        sdy = sh_mid[1] - hip_mid[1]
        spine_angle = math.degrees(math.atan2(abs(sdx), abs(sdy) + 1e-6))
        metrics['spine_lean_deg'] = round(spine_angle, 1)
        if spine_angle > 15:
            issues.append(f'Slouching / leaning ({spine_angle:.0f}°)')
            penalty += 30

    # ── 4. Head tilt left/right ───────────────────────────────────
    if seen(l_eye, r_eye):
        eye_w = abs(l_eye[0] - r_eye[0]) + 1e-6
        eye_tilt = abs(l_eye[1] - r_eye[1]) / eye_w * 100
        metrics['eye_tilt_pct'] = round(eye_tilt, 1)
        if eye_tilt > 15:
            issues.append(f'Head tilted sideways ({eye_tilt:.0f}%)')
            penalty += 15

        # ── 5. Head position relative to the shoulders ───────────
        eye_sh_ratio = (sh_mid[1] - eye_mid[1]) / sh_width * 100
        metrics['eye_to_shoulder_pct'] = round(eye_sh_ratio, 1)
        base_es = (BASELINE or {}).get('eye_to_shoulder_pct')
        if base_es:
            if eye_sh_ratio < base_es * 0.85:
                issues.append('Head sinking toward shoulders — slouching or looking down')
                penalty += 25
        elif eye_sh_ratio < 55:   # uncalibrated: upright 64-67, looking down 48
            issues.append('Head low / looking down — raise your screen')
            penalty += 25

    if seen(nose):
        metrics['nose_offset_pct'] = round(abs(nose[0] - sh_mid[0]) / sh_width * 100, 1)

    # ── Position metrics (used for calibration drift) ─────────────
    metrics['shoulder_w_pct'] = round(sh_width / w * 100, 1)
    if seen(nose):
        metrics['nose_y_pct'] = round(nose[1] / h * 100, 1)
    if BASELINE:
        b = BASELINE
        if 'nose_y_pct' in metrics and 'nose_y_pct' in b and metrics['nose_y_pct'] - b['nose_y_pct'] > 6:
            issues.append('Sitting lower than your usual posture')
            penalty += 25
        bw_ = b.get('shoulder_w_pct')
        if bw_:
            if metrics['shoulder_w_pct'] > bw_ * 1.25:
                issues.append('Too close to the screen')
                penalty += 15
            elif metrics['shoulder_w_pct'] < bw_ * 0.75:
                issues.append('Leaning back / far from the screen')
                penalty += 10

    # ── Score: severity-weighted ──────────────────────────────────
    score = max(0, 100 - penalty)
    status = 'Good' if score >= 80 else ('Fair' if score >= 60 else 'Poor')
    metrics['method'] = 'mediapipe_33pts'

    return status, score, issues, metrics, True


def _posture_opencv_fallback(frame):
    """Face-position based posture when MediaPipe is unavailable."""
    h, w = frame.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if not len(faces):
        return None, None, [], {}, False

    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    issues = []
    metrics = {}

    off = abs((fx + fw // 2) - w // 2) / (w // 2)
    metrics['head_center_offset'] = round(off * 100, 1)
    if off > 0.25:
        issues.append('Head tilted sideways')

    tr = fy / (h + 1e-6)
    metrics['face_top_pct'] = round(tr * 100, 1)
    if tr > 0.45:
        issues.append('Head too low — possible slouching')

    sz = (fw * fh) / (w * h)
    metrics['face_size_pct'] = round(sz * 100, 2)
    if sz < 0.015:
        issues.append('Too far from camera')
    elif sz > 0.18:
        issues.append('Too close to screen')

    if fw / (fh + 1e-6) < 0.65:
        issues.append('Head turned sideways')

    sy = fy + fh
    if sy < h:
        sr = gray[sy:min(sy + fh // 2, h), :]
        lv = float(np.mean(sr[:, :w // 2]))
        rv = float(np.mean(sr[:, w // 2:]))
        sym = abs(lv - rv) / (max(lv, rv) + 1e-6)
        metrics['shoulder_symmetry'] = round(sym * 100, 1)
        if sym > 0.12:
            issues.append('Shoulders appear uneven')

    metrics['method'] = 'opencv_face'
    score = max(0, 100 - len(issues) * 20)
    status = 'Good' if score >= 80 else ('Fair' if score >= 60 else 'Poor')
    return status, score, issues, metrics, True


def detect_posture(frame):
    if MEDIAPIPE_OK:
        status, score, issues, metrics, ok = _posture_mediapipe(frame)
        if ok:
            return status, score, issues, metrics
    status, score, issues, metrics, ok = _posture_opencv_fallback(frame)
    return status, score, issues, metrics


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def analyze_frame(frame):
    result = dict(
        mood=None, mood_scores={}, mood_metrics={},
        posture_status=None, posture_score=None,
        posture_issues=[], posture_metrics={},
        face_detected=False, pose_detected=False,
        mood_method='none', posture_method='none'
    )

    # Mood
    mood, scores, mm = detect_mood(frame)
    if mood:
        result.update(
            face_detected=True, mood=mood,
            mood_scores=scores, mood_metrics=mm,
            mood_method=mm.get('method', 'unknown')
        )

    # Posture
    status, score, issues, pm = detect_posture(frame)
    if status and pm.get('method') == 'mediapipe_33pts':
        _calib_feed(pm, issues)
    if status:
        result.update(
            pose_detected=True, posture_status=status,
            posture_score=score, posture_issues=issues,
            posture_metrics=pm,
            posture_method=pm.get('method', 'unknown')
        )

    return result
