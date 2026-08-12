#!/usr/bin/env bash
# Backup Postgres + Chroma volumes used by docker compose (prod or local).
# Usage:
#   ./docker/backup.sh                 # uses docker-compose.prod.yml project
#   COMPOSE_FILE=docker/docker-compose.yml ./docker/backup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.prod.yml}"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/$STAMP"
mkdir -p "$OUT"

cd "$ROOT"

echo "[backup] Writing to $OUT"

# Postgres dump via running db container
DB_SERVICE="${DB_SERVICE:-db}"
POSTGRES_USER="${POSTGRES_USER:-ariva}"
POSTGRES_DB="${POSTGRES_DB:-self_correcting_rag}"

if docker compose -f "$COMPOSE_FILE" ps --status running -q "$DB_SERVICE" >/dev/null 2>&1; then
  docker compose -f "$COMPOSE_FILE" exec -T "$DB_SERVICE" \
    pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$OUT/postgres.sql.gz"
  echo "[backup] postgres.sql.gz ok"
else
  echo "[backup] WARNING: db service not running — skipped Postgres dump" >&2
fi

# Chroma volume: copy from named volume
CHROMA_VOLUME="${CHROMA_VOLUME:-}"
if [ -z "$CHROMA_VOLUME" ]; then
  # Infer from compose project name (directory of compose file parent)
  PROJECT="$(basename "$(dirname "$COMPOSE_FILE")")"
  # Compose v2 project name is usually the parent folder of the compose file's context
  # Prefer explicit listing
  CHROMA_VOLUME="$(docker volume ls -q | grep -E 'chroma_data$' | head -n1 || true)"
fi

if [ -n "$CHROMA_VOLUME" ]; then
  docker run --rm \
    -v "$CHROMA_VOLUME":/data:ro \
    -v "$OUT":/backup \
    alpine:3.20 \
    tar czf /backup/chroma_data.tar.gz -C /data .
  echo "[backup] chroma_data.tar.gz from volume $CHROMA_VOLUME"
else
  echo "[backup] WARNING: chroma_data volume not found — skipped" >&2
fi

# Retention (keep last N)
KEEP="${BACKUP_KEEP:-14}"
ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -rf

echo "[backup] Done: $OUT"
ls -lh "$OUT"
