# Local development commands for the whole stack.
# Run `make help` to list targets.

COMPOSE := docker compose -f docker/docker-compose.yml --env-file .env
PY      := .venv/bin/python
UVICORN := .venv/bin/uvicorn

.PHONY: help up down restart db searxng migrate api web stack logs wipe test eval lint typecheck

help:
	@echo "Local dev targets:"
	@echo "  make up         start Postgres + SearXNG (Docker)"
	@echo "  make migrate    apply Alembic migrations"
	@echo "  make api        run FastAPI on :8000 (reload)"
	@echo "  make web        run Next.js on :3000 (dev)"
	@echo "  make stack      full backend in Docker (db + searxng + api)"
	@echo "  make logs       tail API logs (Docker)"
	@echo "  make down       stop Docker services (keeps data)"
	@echo "  make wipe       stop and DELETE all data volumes"
	@echo "  make test       run backend test suite"
	@echo "  make eval       run golden-set citation evals"

up:
	$(COMPOSE) up -d db searxng
	@$(MAKE) --no-print-directory wait-db

wait-db:
	@until .venv/bin/python -c "import asyncio, asyncpg; from app.core.config import settings; asyncio.run(asyncpg.connect(settings.DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')))" 2>/dev/null; do sleep 1; done
	@echo "Postgres is ready."

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart db searxng

wipe:
	$(COMPOSE) down -v
	@echo "All data volumes removed."

migrate:
	$(PY) -m alembic upgrade head
	-$(PY) scripts/migrate_chroma_collections.py

api:
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd frontend && npm run dev

stack:
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f api

test:

eval-live:
	@echo "Usage: make eval-live MODELS='modelA,modelB' (judge: gpt-oss-120b)"
	$(PY) -m evals.harness --models "$(MODELS)"

test:

eval:
	$(PY) -m evals.run_eval

lint:
	cd frontend && npm run lint
