#!/bin/bash
# Restart the full stack: stop → wipe data → rebuild → start → migrate → show logs
# Usage: ./scripts/restart.sh [--keep-data]

set -e

COMPOSE="docker compose -f docker/docker-compose.yml"
KEEP_DATA=false

if [[ "$1" == "--keep-data" ]]; then
    KEEP_DATA=true
fi

echo "═══════════════════════════════════════════════"
echo "  Self-Correcting RAG — Full Restart"
echo "═══════════════════════════════════════════════"

# 1. Stop
echo ""
echo "▶ Stopping containers..."
$COMPOSE down 2>/dev/null || true

# 2. Wipe data (optional)
if [ "$KEEP_DATA" = false ]; then
    echo "▶ Wiping database and ChromaDB volumes..."
    $COMPOSE down -v 2>/dev/null || true
else
    echo "▶ Keeping existing data volumes"
fi

# 3. Rebuild and start
echo "▶ Building and starting containers..."
$COMPOSE up -d --build

# 4. Wait for DB health
echo "▶ Waiting for database to be ready..."
for i in $(seq 1 30); do
    if $COMPOSE exec -T db pg_isready -U ariva -d self_correcting_rag -q 2>/dev/null; then
        echo "  ✓ Database ready"
        break
    fi
    sleep 1
done

# 5. Run migrations
echo "▶ Running Alembic migrations..."
$COMPOSE exec -T api alembic upgrade head 2>&1 || echo "  ⚠ Migration skipped (tables may already exist)"

# 6. Show status
echo ""
echo "═══════════════════════════════════════════════"
echo "  Services"
echo "═══════════════════════════════════════════════"
$COMPOSE ps

echo ""
echo "═══════════════════════════════════════════════"
echo "  URLs"
echo "═══════════════════════════════════════════════"
echo "  Frontend:    http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  Swagger:     http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"
echo "  Metrics:     http://localhost:8000/metrics"
echo ""
echo "▶ Tailing logs (Ctrl+C to stop)..."
echo ""

# 7. Tail logs (with timestamps)
$COMPOSE logs -f --timestamps --tail=50
