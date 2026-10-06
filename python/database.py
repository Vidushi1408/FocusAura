import mysql.connector
from mysql.connector import Error
import json, csv, io

from config import DB

def conn(): return mysql.connector.connect(**DB)

def _q(sql,params=(),one=False,many=False):
    try:
        c=conn();cur=c.cursor(dictionary=True)
        cur.execute(sql,params)
        if many:
            rows=cur.fetchall()
            for r in rows:
                for k,v in r.items():
                    if hasattr(v,'isoformat'):r[k]=str(v)
            cur.close();c.close();return rows
        if one:
            r=cur.fetchone()
            if r:
                for k,v in r.items():
                    if hasattr(v,'isoformat'):r[k]=str(v)
            cur.close();c.close();return r
        lid=cur.lastrowid;cur.close();c.close();return lid
    except Error as e:
        print(f'[DB] {e}');return None

def ins_sensor(d):
    return _q("INSERT INTO sensor_data(heart_rate,spo2,lux,fsr_raw,fsr_voltage,esp32_ip,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mpu_temp)"
              "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (d.get('heart_rate'),d.get('spo2'),d.get('lux'),d.get('fsr_raw'),d.get('fsr_voltage'),d.get('ip',''),
         d.get('accel_x'),d.get('accel_y'),d.get('accel_z'),d.get('gyro_x'),d.get('gyro_y'),d.get('gyro_z'),d.get('temperature')))

def ins_mood(d):
    sc=d.get('mood_scores',{})
    return _q("INSERT INTO mood_data(mood,score_happy,score_sad,score_stressed,score_surprised,score_neutral,face_detected,mood_method)VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
        (d.get('mood'),sc.get('happy',0),sc.get('sad',0),sc.get('stressed',0),sc.get('surprised',0),sc.get('neutral',0),int(d.get('face_detected',False)),d.get('mood_method')))

def ins_posture(d):
    pm=d.get('posture_metrics',{})
    return _q("INSERT INTO posture_data(posture_status,posture_score,neck_angle,spine_lean,shoulder_tilt,eye_tilt,issues,pose_detected,posture_method)"
              "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (d.get('posture_status'),d.get('posture_score'),
         pm.get('neck_angle_deg'),pm.get('spine_lean_deg'),
         pm.get('shoulder_tilt_pct') if pm.get('shoulder_tilt_pct') is not None else pm.get('shoulder_symmetry'),
         pm.get('eye_tilt_pct'),
         json.dumps(d.get('posture_issues',[])),int(d.get('pose_detected',False)),d.get('posture_method')))

def ins_stress(d):
    return _q("INSERT INTO stress_data(stress_score,stress_level,hr_factor,spo2_factor,mood_factor,posture_factor,fsr_factor)VALUES(%s,%s,%s,%s,%s,%s,%s)",
        (d.get('score'),d.get('level'),d.get('hr_f'),d.get('spo2_f'),d.get('mood_f'),d.get('posture_f'),d.get('fsr_f')))

def ins_productivity(d):
    return _q("INSERT INTO productivity_data(productivity_score,productivity_label,posture_contrib,mood_contrib,hr_contrib,stress_contrib,fsr_contrib)VALUES(%s,%s,%s,%s,%s,%s,%s)",
        (d.get('score'),d.get('label'),d.get('posture_c'),d.get('mood_c'),d.get('hr_c'),d.get('stress_c'),d.get('fsr_c')))

def ins_alert(d):
    return _q("INSERT INTO alerts(alert_type,severity,message,mood,posture,stress_score,heart_rate,spo2,sent_to_esp32)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (d.get('type'),d.get('severity'),d.get('message'),d.get('mood'),d.get('posture'),d.get('stress'),d.get('hr'),d.get('spo2'),int(d.get('sent',False))))

def hist_sensor(n=60):  return _q("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def hist_mood(n=60):    return _q("SELECT * FROM mood_data ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def hist_posture(n=60): return _q("SELECT * FROM posture_data ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def hist_stress(n=60):  return _q("SELECT * FROM stress_data ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def hist_productivity(n=60): return _q("SELECT * FROM productivity_data ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def hist_alerts(n=30):  return _q("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT %s",(n,),many=True)
def latest_sensor():    return _q("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1",one=True)
def mood_dist():
    return _q("SELECT mood,COUNT(*) as cnt FROM mood_data WHERE timestamp>=NOW()-INTERVAL 1 HOUR GROUP BY mood",many=True)
def hourly_sensor(h=24):
    return _q("SELECT DATE_FORMAT(timestamp,'%%H:%%i') as t,ROUND(AVG(heart_rate),1) hr,ROUND(AVG(spo2),1) sp,ROUND(AVG(lux),0) lx,ROUND(AVG(fsr_raw),0) fsr FROM sensor_data WHERE timestamp>=NOW()-INTERVAL %s HOUR GROUP BY DATE_FORMAT(timestamp,'%%Y-%%m-%%d %%H:%%i') ORDER BY timestamp",(h,),many=True)
def hourly_stress(h=24):
    return _q("SELECT DATE_FORMAT(timestamp,'%%H:%%i') as t,ROUND(AVG(stress_score),1) ss FROM stress_data WHERE timestamp>=NOW()-INTERVAL %s HOUR GROUP BY DATE_FORMAT(timestamp,'%%Y-%%m-%%d %%H:%%i') ORDER BY timestamp",(h,),many=True)
def ack_alert(aid): _q("UPDATE alerts SET acknowledged=1 WHERE id=%s",(aid,))

def stats_summary():
    s  = _q("SELECT COUNT(*) cnt,ROUND(AVG(heart_rate),1) hr,ROUND(AVG(spo2),1) sp,ROUND(AVG(lux),0) lx FROM sensor_data WHERE timestamp>=NOW()-INTERVAL 1 HOUR",one=True)
    a  = _q("SELECT COUNT(*) cnt FROM alerts WHERE timestamp>=NOW()-INTERVAL 1 HOUR AND acknowledged=0",one=True)
    m  = _q("SELECT mood,COUNT(*) cnt FROM mood_data WHERE timestamp>=NOW()-INTERVAL 1 HOUR GROUP BY mood ORDER BY cnt DESC LIMIT 1",one=True)
    st = _q("SELECT ROUND(AVG(stress_score),1) avg FROM stress_data WHERE timestamp>=NOW()-INTERVAL 1 HOUR",one=True)
    pr = _q("SELECT ROUND(AVG(productivity_score),1) avg FROM productivity_data WHERE timestamp>=NOW()-INTERVAL 1 HOUR",one=True)
    return {'sensor':s,'unread_alerts':(a or {}).get('cnt',0),'top_mood':(m or {}).get('mood'),
            'avg_stress':(st or {}).get('avg'),'avg_productivity':(pr or {}).get('avg')}

def export_csv(table='sensor',hours=24):
    MAP = {
        'sensor':      ('sensor_data',       'timestamp,heart_rate,spo2,lux,fsr_raw,fsr_voltage'),
        'mood':        ('mood_data',          'timestamp,mood,score_happy,score_sad,score_stressed,score_surprised,score_neutral'),
        'posture':     ('posture_data',       'timestamp,posture_status,posture_score,issues,neck_angle,spine_lean,shoulder_tilt,eye_tilt,posture_method'),
        'stress':      ('stress_data',        'timestamp,stress_score,stress_level,hr_factor,mood_factor,posture_factor'),
        'productivity':('productivity_data',  'timestamp,productivity_score,productivity_label,posture_contrib,mood_contrib'),
        'alerts':      ('alerts',             'timestamp,alert_type,severity,message,mood,posture,stress_score,sent_to_esp32'),
    }
    if table not in MAP: return ''
    tbl,cols=MAP[table]; col_list=cols.split(',')
    rows=_q(f"SELECT {cols} FROM {tbl} WHERE timestamp>=NOW()-INTERVAL %s HOUR ORDER BY timestamp DESC",(hours,),many=True)
    if not rows: return 'timestamp\n(no data)\n'
    out=io.StringIO()
    w=csv.DictWriter(out,fieldnames=col_list,extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
    return out.getvalue()

def dashboard_totals():
    def v(sql): return (_q(sql,one=True) or {})
    return {
        'total_readings':   v("SELECT COUNT(*) cnt FROM sensor_data WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('cnt',0),
        'total_alerts':     v("SELECT COUNT(*) cnt FROM alerts WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('cnt',0),
        'avg_stress':       v("SELECT ROUND(AVG(stress_score),1) v FROM stress_data WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('v',0),
        'avg_productivity': v("SELECT ROUND(AVG(productivity_score),1) v FROM productivity_data WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('v',0),
        'avg_hr':           v("SELECT ROUND(AVG(heart_rate),1) v FROM sensor_data WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('v',0),
        'avg_spo2':         v("SELECT ROUND(AVG(spo2),1) v FROM sensor_data WHERE timestamp>=NOW()-INTERVAL 24 HOUR").get('v',0),
    }
