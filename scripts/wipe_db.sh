#!/bin/bash
# Drop all tables and re-run migrations (keeps the DB running)
# Usage: ./scripts/wipe_db.sh

COMPOSE="docker compose -f docker/docker-compose.yml"

echo "▶ Dropping all tables..."
$COMPOSE exec -T api python -c "
import asyncio
from app.core.database import engine, Base
async def drop():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print('  ✓ All tables dropped')
asyncio.run(drop())
"

echo "▶ Re-running migrations..."
$COMPOSE exec -T api alembic upgrade head

echo "✓ Database reset complete"
