# Self-Correcting RAG

**Author:** Nevin Sunil Oommen

An agentic retrieval-augmented generation system with built-in hallucination detection and self-repair. When the LLM's answer isn't grounded in evidence, the system automatically re-plans, retrieves additional context (from vector DB, Wikipedia, or Tavily web search), and regenerates — looping until the answer passes a factual verification step.

## Architecture

```
User Query
    │
    ▼
classify_and_plan ──► conversational_response ──► END      (small talk / meta)
    │
    ├──► ask_clarification ─────────────────────────► END   (genuinely ambiguous)
    │
    ▼
gather_evidence  ◄── Chroma hybrid (per-chat) + web (SearXNG → Wikipedia → Tavily),
    │                  FlashRank cross-encoder rerank, token-budgeted context
    ▼
generate_answer ◄── cited generation from assembled evidence only ([E1..En])
    │
    ▼
verify_answer ◄── deterministic citation validation + support gate (local MiniLM),
    │             then one structured LLM judge verdict
    │
    ├─ unsupported but fixable ──► repair loop: gather ► generate ► verify (bounded)
    ├─ contradicted / unresolvable ──► answer + Caveats
    └─ clean ──────────────────────► answered
```

## Tech Stack
- **FastAPI** — async web framework (SSE + WebSocket streaming)
- **LangGraph** — stateful agent orchestration (`classify_and_plan → gather_evidence → generate_answer → verify_answer`, bounded repair loop)
- **OpenRouter / Google AI / Groq** — pluggable LLM providers with per-role models, fallback chains, and per-user encrypted keys; free-tier friendly defaults
- **ChromaDB** — per-chat vector storage + BM25 hybrid retrieval over uploaded documents
- **Nomic Embed** — text embedding API for ingestion
- **FlashRank** — local cross-encoder reranking of assembled evidence
- **SearXNG (self-hosted) → Wikipedia → Tavily** — layered web search with per-user daily budgets
- **MiniLM support gate** — local-embedding entailment check demoting cited claims their evidence doesn't support (`app/agent/support.py`)
- **PostgreSQL + SQLAlchemy + Alembic** — user auth (OTP), chat history, per-LLM-call tracing
- **Next.js frontend** — streaming chat UI with pipeline tracker and provenance panels
- **Docker** — containerized deployment (Caddy TLS, Cloud Run / Oracle+Vercel guides)

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

## API Endpoints

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
