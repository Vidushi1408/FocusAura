#!/bin/bash
# ================================================================
# IoT Health Monitor v3 — Startup Script (Mac)
# Usage: bash run.sh
# ================================================================

echo ""
echo "========================================"
echo "  IoT Health Monitor v3"
echo "  http://localhost:5001"
echo "========================================"
echo ""

cd "$(dirname "$0")/python"

# Load DB credentials from ../.env (copy .env.example first)
[ -f ../.env ] && set -a && . ../.env && set +a

# Check Python 3
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] Python 3 not found. Install from python.org"
  exit 1
fi

# Check if packages installed
python3 -c "import flask" 2>/dev/null || {
  echo "[INFO] Installing Python packages..."
  pip3 install -r requirements.txt
}

# Check MySQL is running
if ! mysql -u "${DB_USER:-root}" -p"$DB_PASSWORD" -e "USE iot_health_monitor;" 2>/dev/null; then
  echo "[INFO] Creating database..."
  mysql -u "${DB_USER:-root}" -p"$DB_PASSWORD" < ../sql/schema.sql 2>/dev/null || {
    echo "[WARN] Could not connect to MySQL — app will run without DB"
  }
fi

echo "[INFO] Starting Flask server on port 5001..."
echo "[INFO] Press Ctrl+C to stop"
echo ""
python3 app.py
