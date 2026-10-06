"""config.py — environment-based settings (reads ../.env if present)."""
import os

def _load_env():
    path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_env()

DB = dict(host=os.getenv('DB_HOST', 'localhost'),
          port=int(os.getenv('DB_PORT', '3306')),
          database=os.getenv('DB_NAME', 'iot_health_monitor'),
          user=os.getenv('DB_USER', 'root'),
          password=os.getenv('DB_PASSWORD', ''),
          autocommit=True)
DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'
HOST  = os.getenv('FLASK_HOST', '0.0.0.0')
