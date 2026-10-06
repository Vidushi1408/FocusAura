#!/bin/bash
# ================================================================
# IoT Health Monitor v3 — First-time Install Script (Mac)
# Usage: bash install.sh
# ================================================================

echo ""
echo "========================================"
echo "  IoT Health Monitor v3 — Install"
echo "========================================"
echo ""

# 1. Python packages
echo "[1/3] Installing Python packages..."
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  pip3 install -r python/requirements-macos.txt
else
  pip3 install -r python/requirements.txt
fi
echo ""

# 2. MySQL database
[ -f .env ] && set -a && . ./.env && set +a
echo "[2/3] Setting up MySQL database..."
echo "      Enter MySQL root password when prompted:"
mysql -u "${DB_USER:-root}" -p < sql/schema.sql
echo ""

# 3. Verify
echo "[3/3] Verifying installation..."
python3 -c "
import sys
packages = ['flask','cv2','numpy','requests','mysql.connector']
ok = True
for p in packages:
    try: __import__(p); print(f'  [OK] {p}')
    except: print(f'  [MISSING] {p}'); ok = False

try: from deepface import DeepFace; print('  [OK] deepface (AI mood model)')
except: print('  [WARN] deepface not installed — will use OpenCV fallback')

try: import mediapipe; print('  [OK] mediapipe (pose detection)')
except: print('  [WARN] mediapipe not installed — will use face-based posture')

print()
print('Install complete!' if ok else 'Some packages missing — run: pip3 install -r python/requirements-macos.txt')
"
echo ""
echo "Run the app with:  bash run.sh"
echo "Then open:         http://localhost:5001"
echo ""
