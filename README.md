# Self-Correcting RAG

**Author:** Nevin Sunil Oommen

An agentic retrieval-augmented generation system with built-in hallucination detection and self-repair. When the LLM's answer isn't grounded in evidence, the system automatically re-plans, retrieves additional context (from the vector store, Wikipedia, or Tavily web search), and regenerates — looping until the answer passes a factual verification step. Every claim is either cited and verified or explicitly flagged as a caveat; nothing is silently asserted.

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
gather_evidence  ◄── pgvector+BM25 hybrid (owner-scoped) + web
    │                  (SearXNG → Wikipedia → Tavily w/ key rotation),
    │                  FlashRank cross-encoder rerank, token-budgeted context
    ▼
generate_answer ◄── cited generation from assembled evidence only ([E1..En]),
    │               tokens streamed live via astream_events
    ▼
verify_answer ◄── deterministic citation validation + MiniLM support gate,
    │             then one structured LLM judge verdict
    │
    ├─ unsupported but fixable ──► repair loop: gather ► generate ► verify (bounded)
    ├─ contradicted / unresolvable ──► answer + Caveats
    └─ clean ──────────────────────► answered
```

## Tech Stack
- **FastAPI** — async web framework; SSE streaming with **true token streaming** (`astream_events(v2)` token deltas, `answer_reset` on repair passes, live pipeline status + provenance events)
- **LangGraph** — stateful agent orchestration (`classify_and_plan → gather_evidence → generate_answer → verify_answer`, bounded repair loop, guard counters on every loop)
- **OpenRouter / Google AI / Groq / any OpenAI-compatible endpoint** — pluggable providers with custom base URLs (Together, Fireworks, DeepSeek, xAI, Mistral, Ollama local), per-role models, fallback chains, reasoning-effort control, and per-user encrypted keys; free-tier friendly defaults
- **PostgreSQL + pgvector** — stateless backend: chunks, vectors (Nomic `nomic-embed-text-v1.5`, 768 dims), users, chats, traces and usage all in one database (ChromaDB was removed because local-disk persistence broke containerized deploys)
- **Hybrid retrieval** — pgvector cosine `<=>` + BM25, **parent-child chunking** (precise child chunks, 4× parent context re-injected at search time), file-hash dedupe
- **FlashRank** — local ONNX cross-encoder reranking of assembled evidence
- **SearXNG (self-hosted) → Wikipedia → Tavily** — layered web search with Tavily key auto-rotation, full-page enrichment of top results (lxml, 2MB memory cap), per-user daily budgets
- **MiniLM support gate** — local fastembed ONNX encoder demoting cited claims their evidence doesn't semantically support (`app/agent/support.py`); citation-id resolution is not entailment
- **LLM call tracing** — every invocation recorded fire-and-forget to `llm_call_traces` (model, node, latency, tokens, outcome), optional Langfuse export
- **PostgreSQL + SQLAlchemy + Alembic** — user auth (OTP, rotating refresh tokens with reuse detection), chat history, usage quotas computed in SQL (DB-clock windows)
- **HMAC-signed document links** — expiring, constant-time-verified URLs for serving stored originals
- **Next.js 16 + React 19 + Zustand frontend** — streaming chat UI with pipeline tracker, provenance panels with signed citation links, memory browser, per-user provider settings
- **Docker** — containerized stateless deployment (Caddy TLS; Cloud Run, Render blueprint, Oracle+Vercel, Hugging Face Space)

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

### 3. Set up PostgreSQL (pgvector)

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
./docker/backup.sh   # Postgres snapshot
```

**Google Cloud Run (Always Free only, no Cloud SQL):** see [docs/DEPLOY_CLOUD_RUN.md](docs/DEPLOY_CLOUD_RUN.md).

**Render (free web service, Supabase-backed):** the repo ships a Blueprint — `render.yaml`.

**Split deploy (Oracle Cloud API + Vercel frontend):** see [docs/DEPLOY_ORACLE_VERCEL.md](docs/DEPLOY_ORACLE_VERCEL.md).

**Hugging Face Space:** see [deploy/hf-space](deploy/hf-space).

**Production:** keep personal LLM keys out of git and Docker images; use placeholders in `.env.example`; set `ENVIRONMENT=production` (disables SQL echo, validates CORS/`SECRET_KEY` — fail-fast boot); lock CORS via `CORS_ORIGINS`. Caddy terminates TLS. Migrations run on API container start (`RUN_MIGRATIONS=true`).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create unverified user + send email OTP |
| POST | `/auth/verify-email` | Verify OTP and issue JWT |
| POST | `/auth/login` | Get JWT (requires verified email) |
| POST | `/auth/forgot-password` / `/auth/reset-password` | Password reset via OTP |
| GET/PUT/DELETE | `/settings/providers` | Per-user encrypted LLM keys + models (any provider, custom base URLs) |
| GET | `/memory/chunks` / `/memory/stats` | Browse stored chunks for the user |
| POST | `/documents/upload_file` | Upload a document for ingestion |
| GET | `/documents/{id}/file` | Original download via HMAC-signed expiring link |
| POST | `/agent/chats/{id}/query` | Send a query to the self-correcting RAG |
| POST | `/agent/chats/{id}/query_stream` | SSE: pipeline status, live tokens, provenance |
| GET | `/agent/llm-traces` | Per-LLM-call tracing records |

## Evaluation Results

All numbers are reproducible via committed tooling; raw runs live in `evals/results/`.

| Instrument | Result |
|---|---|
| **RAGAS** (12 live cases, MiMo judge, 0 NaN) | faithfulness **0.809** · answer_relevancy **0.919** · context_precision **0.809** · answer_correctness 0.46 |
| **TruthfulQA** (adversarial, n=30, free model, $0) | faithfulness **0.632** |
| **Repair A/B** (paired runs, repair on/off) | **+0.42 avg judge score, +12pp verified-claim share, caveat rate roughly halved**; under degraded retrieval, repair rescues collapsed no-repair runs (1.25 → 3.83 /5) |
| **Golden-set citation eval** (20 cases) | zero fabricated citations across ~40 recorded runs |
| **Cost / latency** | ≈ **$0.0008/query**, 13–30s end-to-end on the paid primary |
| **Tests** | **144 offline tests** (LLMs faked, no network in CI) + live-key integration suite |

```bash
pytest tests/ -q                      # offline suite
python -m evals.run_eval              # golden-set citation checks
python -m evals.ragas_eval            # RAGAS metrics (~$0.04/run)
python -m evals.harness --models <model> --ab-repair --repeat 2   # self-correction lift
```

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing + HMAC link signing (`openssl rand -hex 32`) |
| `ENCRYPTION_KEY` | Dedicated Fernet key for per-user LLM keys |
| `DATABASE_URL` | PostgreSQL async connection string (pgvector) |
| `SMTP_*` | SMTP for email OTP (verify + password reset) |
| `CORS_ORIGINS` | Comma-separated frontend origins (required in production) |
| `GROQ_KEY` / `GOOGLE_AI_API_KEY` / `OPENROUTER_API_KEY` | Optional server LLM defaults (users can set their own in Settings) |
| `NOMIC_API_KEY` | Nomic API key for embeddings (system-only) |
| `TAVILY_API_KEY` / `TAVILY_API_KEY_BACKUP` | Tavily keys with auto-rotation (system-only) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional LLM trace export |

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
│   │   ├── limiter.py       # slowapi rate limiting
│   │   └── usage.py         # Chat/query/tavily/ingest counters (SQL windows)
│   ├── auth/                # JWT + refresh rotation + OTP
│   ├── settings/            # Per-user provider key API (any provider)
│   ├── memory/              # Chunk browsing/stats
│   ├── documents/           # Ingestion, pgvector store, signed links, clients
│   ├── agent/               # LangGraph nodes, streaming, support gate, evals
│   └── observability/       # llm_call_traces + optional Langfuse export
├── evals/                   # Golden set, harness, RAGAS, results/
├── frontend/                # Next.js 16 + React 19 streaming chat UI
├── docker/                  # Dockerfile, compose, Caddy, backup
├── render.yaml              # Render blueprint (free tier)
├── cloudbuild.yaml          # Cloud Build → Cloud Run
├── requirements.txt
├── .env.example
└── .gitignore
```

## License

Copyright © Nevin Sunil Oommen. Released under the MIT License.
