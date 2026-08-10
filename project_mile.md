# Agentic RAG Platform — Functional Architecture & Production MVP Blueprint

> Engineering reference for building the ingestion pipeline, self-correcting agentic RAG loop, and observability layer as a production-ready MVP.

---

## 1. System Topology Overview

The system operates on **three** logical lifecycles:

1. **Document Ingestion Pipeline** — converts static assets (PDF, Markdown) into searchable vector indices.
2. **Self-Correcting Query Loop (Agentic RAG)** — accepts user queries, retrieves context, grades its validity, falls back to live web/Wikipedia search when needed, self-checks the generated answer, and streams the verified response.
3. **Observability & Logging** — asynchronously records every query's trajectory, cost, and latency to PostgreSQL for auditing and debugging.

```
┌────────────────────┐      ┌───────────────────────────┐      ┌────────────────────┐
│  Ingestion Pipeline │      │   Agentic RAG Query Loop   │      │   Observability      │
│  (async, PDF → Vec) │ ───▶ │ (retrieve → grade → gen)   │ ───▶ │ (Postgres logging)   │
└────────────────────┘      └───────────────────────────┘      └────────────────────┘
```

---

## 2. Lifecycle 1 — Document Ingestion Flow

Handles file processing **asynchronously** so the server stays responsive during large multi-page PDF parsing.

### 2.1 Flow

```
[Client] --POST /api/v1/documents/upload--> [Pydantic Validation]
    --> [Save Temp PDF to Disk] --> [Trigger BackgroundTask]
    --> [FastAPI returns 202 Accepted immediately]

Background worker (async):
    [Recursive Character Splitting]
        --> [Text Chunks]
        --> [Convert to Vector Embedding] (Sentence-Transformers)
        --> [Upsert Metadata + Embeddings] --> [ChromaDB (local storage)]
```

### 2.2 Details

- **Route (`POST /api/v1/documents/upload`)**: validates JWT auth, saves the raw file stream, and delegates chunking to a background execution thread. Returns `202 Accepted` without blocking on parse time.
- **Text Chunking Strategy**: context-preserving chunking. Given character length `L`, chunk size `c`, and overlap `o`, the overlap interval for chunk `i` is:

  ```
  Overlap Interval = [ i·(c−o), i·(c−o) + c ]
  ```

  This guarantees keywords are never split across chunk boundaries.

- **Vector Creation**: each chunk is embedded into a **384-dimensional** dense vector via a local Sentence-Transformers model, then upserted into **ChromaDB**.

---

## 3. Lifecycle 2 — Agentic RAG & Self-Correction Loop

The central execution path. A user query triggers a graph of LangGraph nodes that dynamically route between retrieval, grading, web fallback, generation, and answer verification.

### 3.1 Flow

```
[User Sends Query over WS]
        │
        ▼
[Retrieve chunks (ChromaDB)]
        │
        ▼
[Node: Grader (LLM Decision)] ── Is context valid & sufficient? ──┐
        │ YES                                              NO     │
        ▼                                                         ▼
[Node: Generator (LLM)]                          [Node: Scrape & Search (Wikipedia + Tavily)]
   synthesizes response                                            │
        │                                                          ▼
        │                                         [Node: Re-index Temp Context]
        │                                            (adds web data to state)
        │◀───────────────────────────────────────────────────────┘
        ▼
[Node: Answer Grading] — checks hallucination & answer match
        │
        ▼
[Final Stream: token-by-token output over WebSocket]
```

### 3.2 Step-by-Step State Execution

**Step A — Connection & Handshake**
Client connects via persistent WebSocket: `WS /api/v1/agent/query`. Handshake validates the JWT; server initializes a per-connection state tracker.

**Step B — Vector Similarity Search**
Query is embedded into vector `V_q`. Cosine similarity against stored chunk vectors `V_c`:

```
Similarity(V_q, V_c) = (V_q · V_c) / (‖V_q‖ ‖V_c‖)
```

Top-3 scoring chunks are loaded into the LangGraph `state.context`.

**Step C — Relevance Assessment Node ("The Grader")**
An LLM receives the context chunks + query and must return strict JSON:

```json
{
  "relevance_score": "no",
  "reason": "The vector database contains outdated metrics about Q3 profits."
}
```

- `relevance_score: yes` → route to **Generator Node**
- `relevance_score: no` → route to **Web Scraper Fallback Node**

**Step D — Scraping & Web Search Fallback Node**

1. Extract search-friendly keywords from the query.
2. Query the Wikipedia API first; if a match exists, use BeautifulSoup to pull summary paragraphs.
3. Simultaneously trigger a Tavily AI search for live web content.
4. Merge parsed content into `state.context`.

**Step E — Generator Node**
Consolidated context (vector DB chunks **or** fresh web/Wikipedia data) is packaged into an optimized prompt; the LLM streams the response.

**Step F — Hallucination & Answer Grounding Assessment**
Before delivery, cross-evaluate the generated answer:

- **Verification 1**: Is the response grounded only in facts present in the context (no hallucination)?
- **Verification 2**: Does the response actually answer the query?

If either check fails → route back to search/scrape with adjusted queries (bounded retry — see §6.4). If both pass → dispatch to the client over the WebSocket.

---

## 4. Lifecycle 3 — Observability & Logging (PostgreSQL)

Runs as an async "flight data recorder" alongside the agent loop.

**On completion of every query, write a transactional record containing:**

| Field                          | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `query_id`                     | UUID, primary key                                                           |
| `user_id`                      | Foreign key → authenticated user                                            |
| `raw_query` / `final_response` | Full text of input and output                                               |
| `steps_taken`                  | Ordered node trajectory, e.g. `['retrieve', 'grade', 'scrape', 'generate']` |
| `token_cost`                   | Calculated USD cost of LLM usage                                            |
| `latency_ms`                   | Time elapsed from connection to completion                                  |

This table is the backbone for debugging agent behavior, cost monitoring, and usage analytics — treat it as append-only.

---

## 5. File-to-Function Mapping Checklist

| File                       | Responsibility                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `app/core/database.py`     | Async PostgreSQL connection pool; writes session-completion metrics                     |
| `app/documents/service.py` | Background PDF extraction, character chunk splitting, ChromaDB ingestion                |
| `app/agent/state.py`       | Strict Pydantic State Schema — query, active context, loop counters, trajectory history |
| `app/agent/graph.py`       | LangGraph execution layout; conditional routing based on node outputs                   |
| `app/agent/nodes.py`       | Individual node operations (DB calls, grader prompt, generator invocation)              |
| `app/agent/tools/`         | Web + Wikipedia parsers; normalizes HTML into clean raw text                            |

**Additional MVP files not in the original blueprint (recommended for production-readiness):**

| File                      | Responsibility                                                                |
| ------------------------- | ----------------------------------------------------------------------------- |
| `app/core/config.py`      | Centralized settings via `pydantic-settings` (env vars, secrets, model names) |
| `app/core/security.py`    | JWT creation/validation, password hashing (`passlib`/`bcrypt`)                |
| `app/auth/router.py`      | `/register`, `/login`, `/refresh` endpoints                                   |
| `app/documents/router.py` | Upload endpoint, list/delete document endpoints, ingestion status polling     |
| `app/agent/router.py`     | WebSocket route + REST fallback for non-streaming clients                     |
| `app/core/exceptions.py`  | Centralized exception handlers → consistent error JSON shape                  |
| `app/core/rate_limit.py`  | Per-user rate limiting (e.g. `slowapi`) to control LLM spend                  |
| `tests/`                  | Unit tests for chunking math, grader JSON parsing, graph routing logic        |
| `alembic/`                | DB migrations for Postgres schema                                             |
| `docker-compose.yml`      | Orchestrates API + Postgres + ChromaDB containers                             |
| `.env.example`            | Documents required environment variables without leaking secrets              |

---

## 6. Production MVP Hardening — Gaps to Close Before Launch

The original blueprint is architecturally sound but describes a **prototype-grade** flow. To ship as a real MVP, address the following:

### 6.1 Security

- Enforce **file-type and size validation** on upload (magic-byte check, not just extension) to prevent malicious payloads.
- Scan uploaded PDFs for embedded scripts/macros if accepting untrusted user files.
- Rate-limit both the upload endpoint and the WebSocket query endpoint per user/IP.
- Store JWT secrets and Tavily/embedding API keys in a secrets manager, not `.env` in production.

### 6.2 Reliability

- **Background task durability**: FastAPI `BackgroundTasks` are in-memory and lost on server restart. For production, move ingestion to a real task queue (Celery + Redis, or RQ) so uploads survive a redeploy mid-processing.
- **Idempotency**: guard against duplicate uploads/re-ingestion of the same document (hash-based dedupe).
- **Dead-letter handling**: if chunk embedding fails partway through a large PDF, don't leave the document in a silently-partial state — track ingestion status (`pending`, `processing`, `complete`, `failed`) and expose it via an endpoint.

### 6.3 Cost & Performance

- Cache Grader/Generator LLM calls where feasible (e.g. identical query + context hash) to cut redundant spend.
- Set explicit `max_tokens` and timeout budgets per node to prevent runaway generation cost.
- Batch embedding calls during ingestion rather than one chunk at a time.

### 6.4 Loop Safety

- The Hallucination/Answer-Grading retry loop (Step F) needs a **hard iteration cap** (e.g. max 2–3 retries) with a graceful fallback message — otherwise a persistently "insufficient" grader can loop indefinitely and burn cost/latency.
- Log every retry attempt with the reason for failure (hallucination vs. off-topic) for later prompt tuning.

### 6.5 Observability Beyond Logging

- Add structured logging (JSON logs) and a correlation ID that ties WebSocket connection → query_id → all node executions, for tracing in tools like Grafana/Loki.
- Expose a lightweight `/health` and `/metrics` endpoint for uptime monitoring and Prometheus scraping.

### 6.6 Data Lifecycle

- Define a retention/deletion policy for uploaded PDFs and their ChromaDB vectors (GDPR-style "delete my data" support).
- Namespace ChromaDB collections per user/tenant to prevent cross-user context leakage in multi-tenant deployments.

---

## 7. Milestone Breakdown & Execution Strategy

Track progress by functional feature completion rather than fixed calendar dates. Build and test sequentially.

### Feature Subsystem 1 — Core System & Authentication

- **Concepts**: Pydantic v2 config, SQLAlchemy async sessions, async env vars, JWT signing.
- **Build**: DB connection module + register/login routes; FastAPI `Depends` for auth-gating requests.
- **Effort**: ~20–25 hrs (async DB patterns, init logic, Alembic migrations).

### Feature Subsystem 2 — Data Ingestion Engine

- **Concepts**: file streaming, character-based chunking, vector embeddings, metadata filtering.
- **Build**: multi-page PDF upload endpoint → parse → chunk with overlap → embed → store in ChromaDB.
- **Effort**: ~15–20 hrs (semantic extraction config, parsing-exception handling for malformed docs).

### Feature Subsystem 3 — Wikipedia Scraper & Search Fallback

- **Concepts**: BeautifulSoup extraction, API tooling, rate-limit handling.
- **Build**: (1) Wikipedia lookup function for background context, (2) Tavily AI fallback search for topics absent from Wikipedia.
- **Effort**: ~10–12 hrs (clean HTML normalization so the LLM isn't fed noisy markup).

### Feature Subsystem 4 — LangGraph State Machine

- **Concepts**: `StateGraph` structure, conditional node routing, structured LLM output extraction.
- **Build**: orchestrator loading conversation state, pulling doc context, grading relevance, falling back to web search, closing the loop with a grounded answer.
- **Effort**: ~30–35 hrs — **highest-leverage subsystem**; budget extra time for conditional-logic testing, model error handling, and state-history integrity across turns.

### Feature Subsystem 5 — Streaming Observability (WebSockets)

- **Concepts**: async payload delivery, FastAPI WebSocket lifecycle, connection pooling.
- **Build**: upgrade response routes to stream tokens continuously; enable real-time frontend updates.
- **Effort**: ~10 hrs (disconnect handling, state management inside active streams).

### Feature Subsystem 6 — Infrastructure & Deployment

- **Concepts**: `docker-compose` multi-container setup, volume persistence, cloud hosting.
- **Build**: single-command orchestration of API + Postgres + ChromaDB; deploy to a host (e.g. Render).
- **Effort**: ~12–15 hrs (port conflicts, permissions, local-vs-cloud networking).

### Feature Subsystem 7 — Production Hardening _(added for MVP readiness)_

- **Concepts**: task queues (Celery/RQ + Redis), structured logging, retry-cap logic, secrets management.
- **Build**: durable background ingestion, loop-safety caps on the self-correction cycle, `/health` + `/metrics` endpoints, per-tenant ChromaDB namespacing.
- **Effort**: ~15–20 hrs — treat as required for anything beyond a local demo.

---

## 8. Suggested Build Order

1. Core System & Auth (Subsystem 1)
2. Data Ingestion Engine (Subsystem 2)
3. LangGraph State Machine skeleton — retrieve + generate only, no fallback yet (Subsystem 4, partial)
4. Wikipedia/Tavily Fallback (Subsystem 3) → wire into the graph
5. Hallucination/Answer Grading + retry cap (Subsystem 4, complete)
6. Streaming Observability (Subsystem 5)
7. Logging & Metrics (Lifecycle 3)
8. Infrastructure & Deployment (Subsystem 6)
9. Production Hardening pass (Subsystem 7) before public launch

**Total estimated effort**: ~112–137 hrs (original scope) + ~15–20 hrs hardening ≈ **127–157 hrs** for a genuinely production-ready MVP.
