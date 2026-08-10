# Self-Correcting RAG

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
# Run alembic migrations (coming soon)
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

## API Endpoints (in progress)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Create a new user account |
| POST | `/auth/login` | Get JWT access token |
| POST | `/documents/upload_file` | Upload a document for ingestion |
| POST | `/agent/query` | Send a query to the self-correcting RAG |

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | JWT signing key (`openssl rand -hex 32`) |
| `DATABASE_URL` | PostgreSQL async connection string |
| `GROQ_KEY` | Groq API key for LLM inference |
| `NOMIC_API_KEY` | Nomic API key for embeddings |
| `TAVILY_API_KEY` | Tavily API key for web search |

## Project Structure

```
self_correcting_rag/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── config.py        # Pydantic settings from .env
│   │   ├── database.py      # Async SQLAlchemy engine & session
│   │   └── security.py      # JWT + password hashing
│   ├── auth/
│   │   ├── models.py        # User SQLAlchemy model
│   │   ├── schemas.py       # Pydantic request/response schemas
│   │   └── router.py        # Register, login, get_current_user
│   ├── documents/
│   │   ├── clients.py       # Groq, Tavily, ChromaDB, Nomic clients
│   │   ├── service.py       # Ingestion pipeline, chunking, embedding, retrieval
│   │   └── router.py        # File upload endpoint
│   └── agent/
│       ├── state.py         # RAGState TypedDict + prompts
│       ├── graph.py         # LangGraph wiring (nodes + edges)
│       ├── nodes.py         # Node logic (planner, search, generate, verify)
│       ├── search_tool.py   # Wikipedia + Tavily search
│       ├── models.py        # Chats + Agent_interact SQLAlchemy models
│       └── schemas.py       # Chat/interact Pydantic schemas
├── docker/
├── requirements.txt
├── .env.example
└── .gitignore
```

## License

MIT
