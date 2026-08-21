# Self-Correcting RAG

**Author:** Nevin Sunil Oommen

An agentic retrieval-augmented generation system with built-in hallucination detection and self-repair. When the LLM's answer isn't grounded in evidence, the system automatically re-plans, retrieves additional context (from vector DB, Wikipedia, or Tavily web search), and regenerates — looping until the answer passes a factual verification step.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Initial Retrieval   │  ← ChromaDB vector search (top-k chunks)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│      Planner        │  ← LLM decides: evidence sufficient?
└────┬───────────┬────┘
     │           │
  sufficient   not enough
     │           │
     │           ▼
     │  ┌─────────────────┐
     │  │  Search Tools    │  ← Wikipedia + Tavily + Vector re-retrieval
     │  └────────┬────────┘
     │           │
     │           ▼
     │     (back to Planner)
     │
     ▼
┌─────────────────────┐
│  Answer Generation   │  ← LLM generates response from evidence
└─────────┬───────────┘
          ▼
┌─────────────────────────┐
│  Hallucination Checker   │  ← LLM verifies claims against evidence
└────┬───────────────┬────┘
     │               │
  factual        hallucinated
     │               │
     ▼               ▼
   [END]       (back to Planner)
```

## Tech Stack

- **FastAPI** — async web framework
- **LangGraph** — stateful agent orchestration
- **Groq (LLaMA 3)** — fast LLM inference for planning, generation, and verification
- **ChromaDB** — local vector storage for document embeddings
- **Nomic Embed** — text embedding API
- **Tavily** — AI-optimized web search
- **PostgreSQL + SQLAlchemy** — user auth, chat history, observability logging
- **Docker** — containerized deployment

## Supported File Types

PDF, TXT, Markdown, HTML, CSV, Excel, JSON, Python/JS source, images (JPG/PNG/WEBP via Groq Vision OCR).

## Getting Started

### 1. Clone and set up

```bash
git clone https://github.com/nevin-py/self_correcting_rag.git
cd self_correcting_rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys and database URL
```

### 3. Set up PostgreSQL

```bash
createdb self_correcting_rag
.venv/bin/alembic upgrade head
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

**Full local command cheat sheet** (backend + frontend): [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md).

### Production Docker

See [docker/PRODUCTION.md](docker/PRODUCTION.md). Short version:

```bash
# Fill .env: SECRET_KEY, POSTGRES_*, SEARXNG_SECRET, CORS_ORIGINS, DOMAIN, SMTP_*, API keys
docker compose -f docker/docker-compose.prod.yml --env-file .env up -d --build
./docker/backup.sh   # Postgres + Chroma volume snapshot
```

**Google Cloud Run (Always Free only, no Cloud SQL):** see [docs/DEPLOY_CLOUD_RUN.md](docs/DEPLOY_CLOUD_RUN.md).

**Split deploy (Oracle Cloud API + Vercel frontend):** see [docs/DEPLOY_ORACLE_VERCEL.md](docs/DEPLOY_ORACLE_VERCEL.md).

**Production:** keep personal LLM keys out of git and Docker images; use placeholders in `.env.example`; set `ENVIRONMENT=production` (disables SQL echo, validates CORS/`SECRET_KEY`); lock CORS via `CORS_ORIGINS`. Caddy terminates TLS. Migrations run on API container start.
## API Endpoints (in progress)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create unverified user + send email OTP |
| POST | `/auth/verify-email` | Verify OTP and issue JWT |
| POST | `/auth/login` | Get JWT (requires verified email) |
| POST | `/auth/forgot-password` / `/auth/reset-password` | Password reset via OTP |
| GET/PUT/DELETE | `/settings/providers` | Per-user encrypted LLM keys + models |
| GET | `/memory/chunks` / `/memory/stats` | Browse Chroma chunks for the user |
| POST | `/documents/upload_file` | Upload a document for ingestion |
| POST | `/agent/chats/{id}/query` | Send a query to the self-correcting RAG |

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing + Fernet key derivation (`openssl rand -hex 32`) |
| `DATABASE_URL` | PostgreSQL async connection string |
| `SMTP_*` | SMTP for email OTP (verify + password reset) |
| `CORS_ORIGINS` | Comma-separated frontend origins (required in production) |
| `GROQ_KEY` / `GOOGLE_AI_API_KEY` / `OPENROUTER_API_KEY` | Optional server LLM defaults (users can set their own in Settings) |
| `NOMIC_API_KEY` | Nomic API key for embeddings (system-only) |
| `TAVILY_API_KEY` | Tavily API key for web search (system-only) |

**Production:** keep personal LLM keys out of git and Docker images; use placeholders in `.env.example`; set `ENVIRONMENT=production` (validates CORS/`SECRET_KEY`, disables SQL echo). See [docker/PRODUCTION.md](docker/PRODUCTION.md).

## Project Structure

```
self_correcting_rag/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── config.py        # Pydantic settings from .env
│   │   ├── database.py      # Async SQLAlchemy engine & session
│   │   ├── security.py      # JWT + password hashing
│   │   ├── email.py         # SMTP mailer
│   │   ├── secrets.py       # Fernet encrypt/mask for user API keys
│   │   └── usage.py         # Chat/query/tavily/ingest counters
│   ├── auth/
│   ├── settings/            # Per-user provider key API
│   ├── memory/              # Chroma chunk list/stats
│   ├── documents/
│   └── agent/
├── docker/
├── requirements.txt
├── .env.example
└── .gitignore
```

## License

Copyright © Nevin Sunil Oommen. Released under the MIT License.
