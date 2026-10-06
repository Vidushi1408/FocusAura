"""Posture tests against a real photo (matplotlib's public-domain grace_hopper.jpg)."""
import os, sys, glob
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
cv2 = pytest.importorskip('cv2')
analyzer = pytest.importorskip('analyzer')
if not analyzer.MEDIAPIPE_OK:
    pytest.skip('MediaPipe unavailable', allow_module_level=True)

import matplotlib
IMG = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'sample_data', 'grace_hopper.jpg')
src = cv2.imread(IMG)

@pytest.fixture(autouse=True)
def no_saved_calibration(monkeypatch):
    # Tests must not depend on a developer's real python/calibration.json
    monkeypatch.setattr(analyzer, 'BASELINE', None)


def rot(img, deg):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

def test_upright_is_good():
    st, sc, issues, m, ok = analyzer._posture_mediapipe(src)
    assert ok and st == 'Good' and sc == 100 and not issues

def test_scale_invariant():
    small = cv2.copyMakeBorder(cv2.resize(src, None, fx=.5, fy=.5), 150, 150, 200, 200, cv2.BORDER_REPLICATE)
    assert analyzer._posture_mediapipe(small)[0] == 'Good'

def test_tilt_lowers_score():
    st, sc, issues, m, ok = analyzer._posture_mediapipe(rot(src, 35))
    assert ok and st == 'Poor' and issues

def test_missing_shoulders_not_judged():
    # shoulders are outside the frame (MediaPipe extrapolates them with high visibility)
    head_only = src[:src.shape[0] // 2, :]
    assert analyzer._posture_mediapipe(head_only)[4] is False

def test_blank_frame_no_pose():
    assert analyzer._posture_mediapipe(src * 0)[4] is False


def test_calibration_flags_drift_from_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzer, 'CALIB_FILE', str(tmp_path / 'c.json'))
    analyzer.clear_calibration()
    analyzer.start_calibration(3)
    for _ in range(3):
        analyzer.analyze_frame(src)
    st = analyzer.calibration_status()
    assert st['calibrated'] and not st['collecting']
    # same photo, shrunk toward the bottom of the frame = sitting lower / further away
    h, w = src.shape[:2]
    shifted = cv2.copyMakeBorder(cv2.resize(src, None, fx=.7, fy=.7), 200, 0, 100, 100, cv2.BORDER_REPLICATE)
    issues = analyzer.analyze_frame(shifted)['posture_issues']
    assert any('lower' in i or 'far' in i for i in issues)
    analyzer.clear_calibration()
