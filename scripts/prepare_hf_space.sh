#!/usr/bin/env bash
# Stage the repo for a Hugging Face Spaces (Docker SDK) deployment.
#
# HF Spaces build from a single git repo whose ROOT contains `Dockerfile`
# and a README.md with Space metadata. This script assembles that tree under
# deploy/hf-space/ from the current working copy (never copies .env / secrets —
# those are set in the Space's settings UI).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=deploy/hf-space
rm -rf "$STAGE"
mkdir -p "$STAGE"

# App code + migrations + deps
cp -r app "$STAGE/app"
rm -rf "$STAGE/app"/**/__pycache__ 2>/dev/null || find "$STAGE/app" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp -r alembic "$STAGE/alembic"
find "$STAGE/alembic" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cp alembic.ini requirements.txt "$STAGE/"

# Dockerfile — HF Space variant: entrypoint at repo root, listen on $PORT
# (HF routes to the port declared as app_port in README.md).
cat > "$STAGE/Dockerfile" <<'EOF'
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app

USER appuser

# HF Spaces: app_port (README.md) must match this default.
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV UVICORN_WORKERS=1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
EOF

# Entrypoint (same as docker/entrypoint.sh — PORT env respected for HF routing)
cp docker/entrypoint.sh "$STAGE/entrypoint.sh"

# Space metadata
cat > "$STAGE/README.md" <<'EOF'
---
title: Self-Correcting RAG API
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

Self-Correcting RAG backend (FastAPI). All state lives in an external
Postgres (Supabase, pgvector) — the container itself is stateless.
Configure via Space **Settings → Variables and secrets** (see deploy/README.md).
EOF

echo "Staged Space at $STAGE/"
find "$STAGE" -maxdepth 1 -mindepth 1 | sort
