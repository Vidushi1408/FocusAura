import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from calculator import calc_stress, calc_productivity, should_alert


def test_no_data_is_zero_stress():
    assert calc_stress()['score'] == 0

def test_zero_hr_is_no_data():
    assert calc_stress(heart_rate=0)['hr_f'] == 0

def test_missing_inputs_renormalise():
    # only mood present -> score equals mood factor * 100
    assert calc_stress(mood='stressed')['score'] == 90.0

def test_high_stress_inputs():
    r = calc_stress(heart_rate=140, spo2=88, mood='stressed', posture_score=0, fsr_raw=4095)
    assert r['level'] == 'High' and r['score'] >= 90

def test_level_boundaries():
    assert calc_stress(mood='happy')['level'] == 'Low'
    assert calc_stress(mood='sad')['level'] == 'High'

def test_productivity_range():
    r = calc_productivity(posture_score=100, mood='happy', heart_rate=70, stress_score=0, fsr_raw=0)
    assert r['score'] == 100 and r['label'] == 'High'

def test_alerts():
    types = {a['type'] for a in should_alert(hr=120, spo2=90, stress=70, posture='Poor')}
    assert types == {'high_hr', 'low_spo2', 'high_stress', 'bad_posture'}
    assert should_alert(hr=70, spo2=98, stress=10, posture='Good') == []
