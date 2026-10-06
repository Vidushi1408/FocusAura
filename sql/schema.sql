-- ================================================================
--  IoT Health Monitor — MySQL Schema
--  Run: mysql -u root -p < schema.sql
-- ================================================================
CREATE DATABASE IF NOT EXISTS iot_health_monitor;
USE iot_health_monitor;

CREATE TABLE IF NOT EXISTS sensor_data (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    heart_rate  FLOAT,
    spo2        FLOAT,
    lux         FLOAT,
    fsr_raw     INT,
    fsr_voltage FLOAT,
    esp32_ip    VARCHAR(20),
    accel_x FLOAT, accel_y FLOAT, accel_z FLOAT,
    gyro_x  FLOAT, gyro_y  FLOAT, gyro_z  FLOAT,
    mpu_temp FLOAT
);

CREATE TABLE IF NOT EXISTS mood_data (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    mood            VARCHAR(20),
    score_happy     FLOAT DEFAULT 0,
    score_sad       FLOAT DEFAULT 0,
    score_stressed  FLOAT DEFAULT 0,
    score_surprised FLOAT DEFAULT 0,
    score_neutral   FLOAT DEFAULT 0,
    face_detected   TINYINT(1) DEFAULT 0,
    mood_method     VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS posture_data (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    posture_status  VARCHAR(20),
    posture_score   INT,
    neck_angle      FLOAT,
    spine_lean      FLOAT,
    shoulder_tilt   FLOAT,
    eye_tilt        FLOAT,
    issues          TEXT,
    pose_detected   TINYINT(1) DEFAULT 0,
    posture_method  VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS stress_data (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    stress_score  FLOAT,
    stress_level  VARCHAR(20),
    hr_factor     FLOAT,
    spo2_factor   FLOAT,
    mood_factor   FLOAT,
    posture_factor FLOAT,
    fsr_factor    FLOAT
);

CREATE TABLE IF NOT EXISTS productivity_data (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
    productivity_score  FLOAT,
    productivity_label  VARCHAR(20),
    posture_contrib     FLOAT,
    mood_contrib        FLOAT,
    hr_contrib          FLOAT,
    stress_contrib      FLOAT,
    fsr_contrib         FLOAT
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    alert_type    VARCHAR(50),
    severity      VARCHAR(20),
    message       TEXT,
    mood          VARCHAR(20),
    posture       VARCHAR(20),
    stress_score  FLOAT,
    heart_rate    FLOAT,
    spo2          FLOAT,
    sent_to_esp32 TINYINT(1) DEFAULT 0,
    acknowledged  TINYINT(1) DEFAULT 0
);

-- Indexes for fast history queries
CREATE INDEX idx_sensor_ts    ON sensor_data(timestamp);
CREATE INDEX idx_mood_ts      ON mood_data(timestamp);
CREATE INDEX idx_posture_ts   ON posture_data(timestamp);
CREATE INDEX idx_stress_ts    ON stress_data(timestamp);
CREATE INDEX idx_prod_ts      ON productivity_data(timestamp);
CREATE INDEX idx_alerts_ts    ON alerts(timestamp);

SELECT 'Schema created successfully' AS status;
