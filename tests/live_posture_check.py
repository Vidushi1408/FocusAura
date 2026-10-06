"""
Guided live posture check using the real webcam.

Run from the project root in your own Terminal (macOS will ask for camera access):
    .venv/bin/python tests/live_posture_check.py

For each pose you get a countdown, then 5 frames are analysed and averaged.
Calibrates on an upright pose first, then judges every pose against that baseline.
Frames are saved to tests/live_frames/ and a summary table is printed.
"""
import os, sys, time
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
import analyzer as A

POSES = [
    ('upright',     'Sit up straight, shoulders level, looking at the screen'),
    ('slouch',      'Slouch / hunch forward, let your shoulders drop'),
    ('head tilt',   'Tilt your head to one side (ear toward shoulder)'),
    ('lean',        'Lean your whole upper body to one side'),
    ('shoulder up', 'Raise ONE shoulder higher than the other'),
    ('looking down','Look down as if at a phone in your lap'),
    ('too close',   'Lean in close to the camera'),
    ('away',        'Step out of frame (no one visible)'),
]
OUT = os.path.join(ROOT, 'tests', 'live_frames')
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit('Cannot open camera — allow camera access for Terminal in '
             'System Settings > Privacy & Security > Camera, then rerun.')
for _ in range(20):          # let auto-exposure settle
    cap.read(); time.sleep(0.05)

# ── Calibrate on the user's upright posture first ──────────────────
print('\n=== CALIBRATION: sit upright, shoulders level, look at the screen and hold still')
for sec in (4, 3, 2, 1):
    print(f'   starting in {sec}...', end='\r', flush=True); time.sleep(1)
A.start_calibration(8)
for _ in range(40):
    ok, frame = cap.read()
    if ok: A.analyze_frame(frame)
    if not A.calibration_status()['collecting']: break
    time.sleep(0.2)
st = A.calibration_status()
print('   baseline saved:' if st['calibrated'] else '   calibration FAILED (no upright pose detected):', st['baseline'])

rows = []
for name, instr in POSES:
    print(f'\n=== {name.upper()}: {instr}')
    for s in (4, 3, 2, 1):
        print(f'   capturing in {s}...', end='\r', flush=True); time.sleep(1)
    res = []
    for i in range(5):
        ok, frame = cap.read()
        if not ok: continue
        if i == 2: cv2.imwrite(os.path.join(OUT, f'{name.replace(" ", "_")}.jpg'), frame)
        r = A.analyze_frame(frame)
        res.append(r); time.sleep(0.3)
    scores = [r['posture_score'] for r in res if r['posture_score'] is not None]
    issues = sorted({i for r in res for i in r['posture_issues']})
    moods = [r['mood'] for r in res if r['mood']]
    rows.append((name, scores, issues, moods, res[-1]['posture_metrics'] if res else {}))
    print(f"   posture scores {scores or 'no pose'}  issues {issues or '-'}  mood {moods[-1] if moods else '-'}")

cap.release()
print('\n' + '=' * 70)
print(f"{'pose':14}{'score':>7}  {'status':8}issues")
for name, scores, issues, moods, m in rows:
    avg = round(sum(scores) / len(scores)) if scores else None
    st = '-' if avg is None else ('Good' if avg >= 80 else 'Fair' if avg >= 60 else 'Poor')
    print(f"{name:14}{str(avg):>7}  {st:8}{'; '.join(issues) or '-'}")
print(f'\nFrames saved in {OUT}')
