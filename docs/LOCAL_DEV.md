# Local development commands

Run the **backend** (FastAPI) and **frontend** (Next.js) on your machine.

```text
Terminal A          Terminal B           Terminal C (optional)
Postgres+SearXNG    uvicorn API          next dev
(docker)            :8000                :3000
```

**Shortcut:** every command below is wrapped in a Makefile at the repo root —
`make up`, `make migrate`, `make api`, `make web`, `make test`, `make wipe`.
Run `make help` for the full list. The manual commands follow.

---

## 0. One-time setup

```bash
cd /path/to/self_correcting_rag

# Python
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Env
cp .env.example .env
# Edit .env — at least:
#   SECRET_KEY          (openssl rand -hex 32)
#   POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB / SEARXNG_SECRET
#   NOMIC_API_KEY / TAVILY_API_KEY
#   optional: GROQ_KEY, OPENROUTER_API_KEY, GOOGLE_AI_API_KEY
#   ENVIRONMENT=development
#   CORS_ORIGINS=http://localhost:3000

# Frontend
cd frontend
npm install
cd ..
```

For **hybrid** mode (API on host, DB in Docker), set in `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://ariva:YOUR_PASSWORD@localhost:5433/self_correcting_rag
SEARXNG_URL=http://localhost:8899
```

(`POSTGRES_*` must match the user/password/db in `DATABASE_URL`.)

---

## Option A — Recommended: Docker DB + local API + local frontend

### A1. Start Postgres + SearXNG

```bash
cd /path/to/self_correcting_rag

# Needs POSTGRES_PASSWORD and SEARXNG_SECRET in .env
docker compose -f docker/docker-compose.yml --env-file .env up -d db searxng
```

Check:

```bash
docker compose -f docker/docker-compose.yml ps
# Postgres → localhost:5433
# SearXNG  → localhost:8899
```

### A2. Migrate database

```bash
source .venv/bin/activate
alembic upgrade head
```

### A3. Run backend (terminal 1)

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### A4. Run frontend (terminal 2)

```bash
cd frontend
# optional: echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > .env.local
npm run dev
```

- UI: http://localhost:3000  

Default API base is already `http://localhost:8000` if `NEXT_PUBLIC_API_URL` is unset.

---

## Option B — Full backend in Docker + local frontend

Runs API inside Docker (migrations on start) plus DB + SearXNG.

```bash
# Backend stack
docker compose -f docker/docker-compose.yml --env-file .env up -d --build

# Frontend (host)
cd frontend
npm run dev
```

- API: http://localhost:8000  
- UI: http://localhost:3000  

Logs:

```bash
docker compose -f docker/docker-compose.yml logs -f api
```

Stop backend:

```bash
docker compose -f docker/docker-compose.yml down
```

---

## Useful commands

### Backend

```bash
# Activate venv
source .venv/bin/activate

# Run API with reload
uvicorn app.main:app --reload --port 8000

# Migrations
alembic upgrade head
alembic current
alembic history

# Tests
pytest -q
```

### Frontend

```bash
cd frontend
npm run dev      # http://localhost:3000
npm run build
npm run start    # production-mode local serve
npm run lint
```

### Docker helpers

```bash
# Only DB + SearXNG
docker compose -f docker/docker-compose.yml --env-file .env up -d db searxng

# All backend services
docker compose -f docker/docker-compose.yml --env-file .env up -d --build

# Stop / remove containers (keep volumes)
docker compose -f docker/docker-compose.yml down

# Stop and wipe DB + Chroma volumes (destructive)
docker compose -f docker/docker-compose.yml down -v
```

---

## Auth note (local OTP)

With empty `SMTP_*` (non-production), registration still works without mail:
the 6-digit code is returned in the API response as `debug_otp` (visible in
browser devtools / Swagger) **and** logged by the API. Enter it on
`/verify-email`. In production with SMTP configured, no code is ever echoed.

---

## Makefile shortcuts

```bash
make up         # Postgres + SearXNG up, waits until DB accepts connections
make migrate    # alembic upgrade head
make api        # uvicorn --reload on :8000
make web        # next dev on :3000
make stack      # whole backend in Docker instead of local api
make logs       # tail api logs
make down       # stop containers, keep data
make wipe       # stop and DELETE data volumes
```

---

## Quick copy-paste (Option A)

```bash
# --- once ---
cd /path/to/self_correcting_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit secrets + DATABASE_URL on :5433
cd frontend && npm install && cd ..

# --- every session ---
docker compose -f docker/docker-compose.yml --env-file .env up -d db searxng

# terminal 1
source .venv/bin/activate
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```

Open http://localhost:3000 → register → verify with code from API logs → chat.
