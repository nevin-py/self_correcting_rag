#!/bin/sh
set -eu

echo "[entrypoint] Waiting for database..."
python - <<'PY'
import os, sys, time
from urllib.parse import urlparse

url = os.environ.get("DATABASE_URL", "")
if not url:
    print("DATABASE_URL is required", file=sys.stderr)
    sys.exit(1)

# asyncpg DSN → host/port for TCP wait
parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://", 1))
host = parsed.hostname or "db"
port = parsed.port or 5432

import socket
deadline = time.time() + int(os.environ.get("DB_WAIT_SECONDS", "60"))
while True:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"[entrypoint] Database reachable at {host}:{port}")
            break
    except OSError:
        if time.time() > deadline:
            print(f"[entrypoint] Timed out waiting for {host}:{port}", file=sys.stderr)
            sys.exit(1)
        time.sleep(1)
PY

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "[entrypoint] Running alembic upgrade head..."
  alembic upgrade head
fi

WORKERS="${UVICORN_WORKERS:-2}"
HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"

echo "[entrypoint] Starting uvicorn workers=${WORKERS} ${HOST}:${PORT}"
exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers \
  --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-*}"
