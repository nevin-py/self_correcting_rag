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

# asyncpg DSN → host/port for TCP wait (skip Cloud SQL unix sockets)
from urllib.parse import parse_qs

parsed = urlparse(url.replace("postgresql+asyncpg://", "postgresql://", 1))
qs_host = (parse_qs(parsed.query).get("host") or [None])[0]
host = parsed.hostname or qs_host
if not host or str(host).startswith("/"):
    print("[entrypoint] Skipping TCP wait (unix socket / Cloud SQL)")
else:
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
# Cloud Run injects PORT (usually 8080). Docker Compose uses 8000.
PORT="${PORT:-${UVICORN_PORT:-8000}}"

echo "[entrypoint] Starting uvicorn workers=${WORKERS} ${HOST}:${PORT}"
exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers \
  --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-*}"
