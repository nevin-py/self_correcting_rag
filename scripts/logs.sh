#!/bin/bash
# View logs for a specific service or all services
# Usage: ./scripts/logs.sh [api|db|chromadb] [--tail N]

COMPOSE="docker compose -f docker/docker-compose.yml"
SERVICE="${1:-all}"
TAIL="${2:---tail=100}"

if [ "$SERVICE" = "all" ]; then
    $COMPOSE logs -f --timestamps $TAIL
elif [ "$SERVICE" = "api" ]; then
    $COMPOSE logs -f --timestamps $TAIL api
elif [ "$SERVICE" = "db" ]; then
    $COMPOSE logs -f --timestamps $TAIL db
elif [ "$SERVICE" = "chromadb" ]; then
    $COMPOSE logs -f --timestamps $TAIL chromadb
else
    echo "Usage: $0 [api|db|chromadb] [--tail N]"
fi
