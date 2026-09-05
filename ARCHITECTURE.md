# Architecture — Lean Self-Correcting RAG Agent

> One LLM call per job. No keyword tables, no magic thresholds, no domain assumptions.
> Every factual claim in an answer is cited and verified — or explicitly flagged as unverified.
> Fully stateless backend: all state (relational + vectors + files) lives in Postgres, so it
> deploys to Cloud Run / Render / HF Spaces without local-disk dependencies.

## Pipeline

```
classify_and_plan ─┬─ conversational_response ── END   (small talk / meta)
                   ├─ ask_clarification ──────── END   (genuinely ambiguous)
                   └─ gather_evidence ─► generate_answer ─► verify_answer
                                               ▲                │
                                               └── repair ◄─────┘  (≤ MAX_REPAIR_PASSES)
                                                                │
                                                               END
```

| Node | File | What it does | LLM calls |
|---|---|---|---|
| `classify_and_plan` | `app/agent/nodes.py` | Structured `QueryUnderstanding`: mode (research / conversational / clarification), context-bound query rewrite (pronouns resolved against chat history), source needs, 1–3 search queries, temporal focus, geography. Output is sanitized (invalid modes coerced, empty queries filled); total provider failure degrades to research mode with the raw query | 1 |
| `gather_evidence` | `app/agent/nodes.py` | Parallel document retrieval (pgvector + BM25 hybrid, owner-scoped in SQL) + web search (SearXNG → direct Wikipedia article lookup → Tavily with key auto-rotation), full-page enrichment of top results (lxml, 2MB input cap), dedupe, FlashRank cross-encoder rerank, token-budgeted context fill, cite keys `[E1..En]`, citation-token stripping (prompt-injection defense), merges prior-turn evidence | 0 |
| `generate_answer` | `app/agent/nodes.py` | Cited generation from assembled evidence only; chat history included; tokens stream live via `astream_events`; mechanical citation validation afterwards | 1 |
| `verify_answer` | `app/agent/nodes.py` | Deterministic citation check + MiniLM claim-support gate, then one structured judge call (`Verdict`); routes repair or appends honest caveats | 1 |
| `conversational_response` | `app/agent/nodes.py` | Direct LLM reply for greetings/meta — no retrieval | 1 |

Graph wiring: `app/agent/graph.py`. State schema: `app/agent/state.py` (`TypedDict` with
`_keep_latest` / `_add_to_list` reducers + guard counters: `MAX_GRAPH_STEPS=20`,
`MAX_SEARCHES=4`, `MAX_RETRIEVALS=3`, `MAX_REGENERATIONS=2`, `MAX_REPAIR_PASSES=1`,
`MAX_REPAIR_SEARCHES=3`).

## Design rules

1. **No hardcoded knowledge.** Intent detection, query rewriting, tool selection, and claim verdicts are all structured LLM output. There are no keyword lists, stop-word tables, or scoring thresholds anywhere in the pipeline.
2. **Provenance first.** Every piece of evidence is a typed `Evidence` record with source type, name, URL, and date. Answers must cite `[E#]` keys that resolve to real evidence; the citation validator (`app/agent/citation_validator.py`) enforces this deterministically.
3. **Honest uncertainty.** Claims the judge cannot verify are listed under *Caveats* in the answer with `final_status = "answered_with_caveats"`. The agent never silently guesses.
4. **Bounded self-correction.** When verification finds gaps a targeted search could fix, the judge emits `repair_queries` and the graph loops `gather → generate → verify` once (`settings.MAX_REPAIR_PASSES`). Contradictions never loop — they go straight to caveats (or a clarification question when the judge finds an ambiguous term).
5. **Cheap checks first.** Deterministic citation validation runs before the LLM judge; only unresolved questions reach the model. A local-embedding **claim-support gate** (`app/agent/support.py`, fastembed ONNX MiniLM with keyword-cosine fallback) additionally demotes cited sentences whose evidence does not semantically support them (`CITATION_SUPPORT_MIN_SIM=0.55`) — id resolution alone does not imply entailment.
6. **Stateless storage.** ChromaDB was replaced by **pgvector** (`app/documents/vector_store.py`) because local-disk vector persistence vanished on container restart — breaking Cloud Run / Render / HF Spaces deploys. Embeddings stay Nomic `nomic-embed-text-v1.5` (768 dims); only the storage engine moved.

## Chat dynamics & temporal awareness

- **Conversation history** (`messages`) is included in classify, generate, and conversational prompts. Followups ("what about the growth?") are resolved by the planner into standalone rewritten queries.
- **Cross-turn memory:** verified facts persist between turns as `EvidenceState` (`app/agent/evidence_state.py`) — only evidence that backed judge-*verified* claims (capped at 8) carries forward; unresolved items reset each turn. Serialized into each interaction's `routing_path` and reloaded on the next query; merging dedupes by normalized text.
- **Temporal grounding:** current date/time (+ user timezone/location from `request_context`) is injected into every prompt. The planner extracts `temporal_focus`; the generator prefers period-matching evidence and states each figure's period.

## Retrieval & ranking

- **Documents:** `app/documents/service.py` + `app/documents/vector_store.py` — pgvector cosine search (`<=>`) + BM25 hybrid over owner-scoped rows. **Parent-child chunking**: child chunks (2048 chars / 256 overlap) for precise retrieval, linked to 4× parent chunks re-injected at search time for context. File-hash dedupe prevents double ingestion.
- **Ingestion:** PDF (pypdf), TXT, Markdown, HTML (lxml parser, 2MB cap), CSV, Excel (pandas), JSON, source code, images (Groq Vision OCR). Rate-limited, retrying Nomic embedding client.
- **Web:** `app/agent/search_tool.py` — layered search with Tavily key auto-rotation (`TAVILY_API_KEY_BACKUP`), SearXNG primary + year-pinned variant without the recency filter, direct Wikipedia article lookup for encyclopedic-topic queries, and full-page enrichment of the top results (`EVIDENCE_FETCH_TOP_N`, default 2). Per-user daily Tavily budget. Per-backend failures are logged and skipped — one bad backend never fails a turn.
- **Ranking:** BM25 pre-filter → FlashRank ONNX cross-encoder (`app/agent/reranker.py`, `app/agent/context_assembly.py`), then a token-budgeted fill for the generator context. Recency breaks score ties; authoritative domains get a small tie-break on temporal queries.
- **Document links:** originals are served through **HMAC-signed, expiring URLs** (`app/documents/signing.py`, 30-day TTL, constant-time verify) because `<a href>` links can't carry Authorization headers.

## LLM providers

`app/documents/clients.py` resolves planner/generator/verifier clients per request:

- Providers: OpenRouter, Google AI Studio, Groq — plus **any OpenAI-compatible endpoint via custom base URL** (Together, Fireworks, DeepSeek, xAI, Mistral, Anthropic, Ollama local models). Server env keys or per-user keys (Settings UI).
- Fallback chains walk primary → configured fallbacks → other providers. Placeholder keys (`"fill this later"`) are rejected locally by `valid_key()` so they can never produce confusing remote 401s.
- **Reasoning-effort control** (`reasoning: {effort: "low"}`) on OpenRouter clients — hidden reasoning tokens otherwise blow past timeouts.
- **`_structured_invoke` discipline:** native JSON-schema structured output (30s timeout) → raw-JSON retry with Pydantic coercion on parse failure → on timeout, skip the retry and walk the fallback chain.
- Models are config, not code: `OPENROUTER_*_MODEL` / `*_FALLBACKS` in `.env`. Free-tier defaults are set for easy testing (`nvidia/nemotron-3-super-120b-a12b:free` in `.env.example`).

## Streaming & API surface

All under `/api/v1/agent` (see `app/agent/router.py`, `app/agent/streaming.py`):

- `POST /chats/{id}/query` — JSON response with `answer`, `claims`, `citations`, `verification_errors`, `final_status`.
- `POST /chats/{id}/query_stream` — SSE with **true token streaming**: `astream_events(v2)` intercepts chat-model token deltas so `generate_answer` tokens stream as they are produced (planner/judge streams stay internal). The producer runs in its own task pushing into a queue (LangGraph would otherwise drop mid-stream frames). Events: `status` (live pipeline tracker), `token`, `answer_reset` (repair pass replaces the streamed text), `provenance` (citations pushed before generation ends), `ping` (keepalive), `done`. The WebSocket variant was deliberately removed — it bypassed rate limits and usage budgets.
- `GET /llm-traces` — per-LLM-call tracing (model, node, latency, tokens, outcome) recorded fire-and-forget to `llm_call_traces`, optional Langfuse export.
- Chat CRUD, history, messages, usage endpoints. Post-stream persistence is serialized with a `pg_advisory_xact_lock` per chat; DB sessions are short-lived around graph execution so slow LLM calls never hold pooled connections.

## Auth & token flow

- Access JWT (short-lived, `Authorization: Bearer`) + opaque refresh token stored hashed, single-use, with **reuse detection**: presenting a revoked token revokes the user's entire token family.
- The refresh token is issued as an **httpOnly cookie** (`Path=/api/v1/auth`, `SameSite=None; Secure` in production) so browser clients never store it in JS-readable storage; JSON-body tokens remain supported for API clients.
- Password change/reset revokes all live refresh tokens. Email OTP flows (register/reset) use hashed, attempt-capped, expiry-checked codes; in non-production undeliverable OTPs echo as `debug_otp`, in production undelivered mail → 503.
- Per-user LLM keys are Fernet-encrypted under a dedicated `ENCRYPTION_KEY` (independent of `SECRET_KEY` rotation); legacy rows fall back transparently (`scripts/reencrypt_user_keys.py`).
- **Usage budgets** (`usage_events`) are computed **in SQL with DB-clock windows** — the original Python-side UTC comparison silently never enforced caps on any non-UTC Postgres. Rate limits via `app/core/limiter.py` (proxy-aware IP keys, optional Redis storage).
- **Fail-fast boot validation** in production: CORS origins set, SMTP present, strong `SECRET_KEY` — the service refuses to start unsafe.

## Configuration

Key knobs in `.env` (reference: `.env.example`):

| Variable | Purpose |
|---|---|
| `OPENROUTER_*_MODEL`, `*_FALLBACKS` | Planner/generator/verifier models + fallback chains |
| `MAX_SEARCHES`, `MAX_RETRIEVALS`, `MAX_REPAIR_PASSES`, `MAX_REPAIR_SEARCHES` | Loop budgets |
| `QUERY_TIMEOUT_SECONDS` | Whole-query deadline (0 = disabled) |
| `CITATION_SUPPORT_GATE`, `CITATION_SUPPORT_MIN_SIM` | MiniLM entailment gate (default on, 0.55) |
| `SEARXNG_URL`, `TAVILY_API_KEY(+_BACKUP)` | Web search backends |
| `NOMIC_API_KEY` | Embeddings (system-only) |
| `GROQ_KEY`, `GOOGLE_AI_API_KEY`, `OPENROUTER_API_KEY` | Server provider keys |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional trace export |

## Local development
```bash
make up         # Postgres (:5433) + SearXNG (:8888) via Docker, waits for readiness
make migrate    # alembic upgrade head
make api        # FastAPI on :8000
make web        # Next.js on :3000
```

With no working SMTP in non-production, registration OTPs are echoed back as
`debug_otp` in the register/resend API responses. Full guide: `docs/LOCAL_DEV.md`.


## Testing, evals & deployment

```bash
pytest tests/ -q                      # 144 offline tests (LLMs faked; no network)
INTEGRATION_TEST=1 pytest tests/      # live-key integration tests
python -m evals.run_eval              # 20-case golden-set citation checks
python -m evals.ragas_eval            # RAGAS quality metrics (paid judge, ~$0.04/run)
python -m evals.harness --models <model> --ab-repair --repeat 2   # repair A/B + lift report
```

Headline numbers (all reproducible, saved in `evals/results/`): **RAGAS faithfulness 0.809 /
relevancy 0.919 / context precision 0.809** (12 live cases, 0 NaN), **TruthfulQA-free
faithfulness 0.632** (n=30, adversarial, zero LLM cost), **repair A/B: +0.42 avg judge score
and +12pp verified-claim share** (and under degraded retrieval, repair rescues collapsed
no-repair runs from 1.25 → 3.83 /5). Zero fabricated citations across ~40 recorded runs;
≈ $0.0008/query.

**Deploy:** Google Cloud Run (`cloudbuild.yaml` → Artifact Registry; `RUN_MIGRATIONS=true`
applies Alembic on boot), **Render blueprint** (`render.yaml`, free tier, Supabase Postgres),
split Oracle API + Vercel frontend (`docs/DEPLOY_ORACLE_VERCEL.md`), or a Hugging Face Space
(`deploy/hf-space`). Docker + Caddy TLS for self-hosting (`docker/PRODUCTION.md`).
