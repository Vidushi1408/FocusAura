"""
Stress and Productivity Calculator
===================================
Stress  (0–100): High HR + bad posture + sad/stressed mood + low SpO2 + high FSR = high stress
Productivity (0–100): Good posture + happy/neutral mood + stable HR + low stress + low FSR = high
"""

def _has(v):
    """A reading counts only if present and non-zero (0 = sensor not reading)."""
    return v is not None and v > 0


def calc_stress(heart_rate=None, spo2=None, mood=None,
                posture_status=None, posture_score=None, fsr_raw=None):
    """Weighted stress score. Missing inputs are dropped and weights renormalised."""
    factors, parts = {}, []   # parts: (weight, factor)

    if _has(heart_rate):
        hr = heart_rate
        if hr > 100:   hr_f = min(1.0, (hr-100)/40)
        elif hr < 55:  hr_f = 0.3
        else:          hr_f = max(0, (hr-75)/25)*0.5
        parts.append((0.30, hr_f))
    else:
        hr_f = 0.0
    factors['hr_f'] = round(hr_f, 3)

    if _has(spo2):
        if spo2 < 90:    spo2_f = 1.0
        elif spo2 < 95:  spo2_f = (95-spo2)/5 * 0.8
        else:            spo2_f = 0.0
        parts.append((0.20, spo2_f))
    else:
        spo2_f = 0.0
    factors['spo2_f'] = round(spo2_f, 3)

    mood_map = {'stressed': 0.9, 'sad': 0.7, 'surprised': 0.4,
                'neutral': 0.2, 'happy': 0.0}
    if mood in mood_map:
        mood_f = mood_map[mood]
        parts.append((0.25, mood_f))
    else:
        mood_f = 0.0
    factors['mood_f'] = mood_f

    if posture_score is not None:
        posture_f = max(0, (80 - posture_score) / 80)
        parts.append((0.15, posture_f))
    else:
        posture_f = 0.0
    factors['posture_f'] = round(posture_f, 3)

    if fsr_raw is not None:
        fsr_f = min(1.0, fsr_raw / 4095)
        parts.append((0.10, fsr_f))
    else:
        fsr_f = 0.0
    factors['fsr_f'] = round(fsr_f, 3)

    total_w = sum(w for w, _ in parts)
    score = (sum(w * f for w, f in parts) / total_w * 100) if total_w else 0.0
    score = round(min(100, max(0, score)), 1)

    if score >= 65:   level = 'High'
    elif score >= 35: level = 'Medium'
    else:             level = 'Low'

    return {'score': score, 'level': level, **factors}


def calc_productivity(posture_status=None, posture_score=None, mood=None,
                      heart_rate=None, stress_score=None, fsr_raw=None):
    contribs = {}

    # Posture contribution (0–25)
    ps = 50 if posture_score is None else posture_score
    posture_c = (ps / 100) * 25
    contribs['posture_c'] = round(posture_c, 1)

    # Mood contribution (0–25)
    mood_map = {'happy': 25, 'neutral': 20, 'surprised': 12,
                'sad': 8, 'stressed': 5}
    mood_c = mood_map.get(mood or 'neutral', 15)
    contribs['mood_c'] = mood_c

    # Heart rate stability (0–20) — near 70 is ideal
    hr = heart_rate if _has(heart_rate) else 70
    hr_c = max(0, 20 - abs(hr - 70) * 0.4)
    contribs['hr_c'] = round(hr_c, 1)

    # Low stress = high productivity (0–20)
    ss = 30 if stress_score is None else stress_score
    stress_c = max(0, (100 - ss) / 100 * 20)
    contribs['stress_c'] = round(stress_c, 1)

    # Low FSR = not fidgeting (0–10)
    fsr = fsr_raw or 0
    fsr_c = max(0, 10 - (fsr / 4095) * 10)
    contribs['fsr_c'] = round(fsr_c, 1)

    score = round(min(100, posture_c + mood_c + hr_c + stress_c + fsr_c), 1)

    if score >= 70:   label = 'High'
    elif score >= 45: label = 'Moderate'
    else:             label = 'Low'

    return {'score': score, 'label': label, **contribs}


def should_alert(hr=None, spo2=None, stress=None, posture=None):
    alerts = []
    if hr and hr > 110:
        alerts.append({'type':'high_hr','severity':'High',
                       'message':f'Heart rate critically high: {hr:.0f} BPM'})
    if spo2 and spo2 < 92:
        alerts.append({'type':'low_spo2','severity':'High',
                       'message':f'SpO2 dangerously low: {spo2:.1f}%'})
    elif spo2 and spo2 < 95:
        alerts.append({'type':'low_spo2','severity':'Medium',
                       'message':f'SpO2 below normal: {spo2:.1f}%'})
    if stress and stress >= 65:
        alerts.append({'type':'high_stress','severity':'High',
                       'message':f'High stress detected: {stress:.0f}/100'})
    elif stress and stress >= 40:
        alerts.append({'type':'medium_stress','severity':'Medium',
                       'message':f'Elevated stress: {stress:.0f}/100'})
    if posture == 'Poor':
        alerts.append({'type':'bad_posture','severity':'Medium',
                       'message':'Poor posture detected — please sit up straight'})
    return alerts
