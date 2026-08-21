# Architecture — Lean Self-Correcting RAG Agent

> One LLM call per job. No keyword tables, no magic thresholds, no domain assumptions.
> Every factual claim in an answer is cited and verified — or explicitly flagged as unverified.

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
| `classify_and_plan` | `app/agent/nodes.py` | Structured `QueryUnderstanding`: mode (research / conversational / clarification), context-bound query rewrite, source needs, 1–3 search queries, temporal focus, geography | 1 |
| `gather_evidence` | `app/agent/nodes.py` | Parallel document retrieval (Chroma hybrid) + web search (SearXNG → Wikipedia → Tavily), dedupe, FlashRank cross-encoder rerank, cite keys `[E1..En]`, merges prior-turn evidence | 0 |
| `generate_answer` | `app/agent/nodes.py` | Cited generation from assembled evidence only; chat history included; mechanical citation validation afterwards | 1 |
| `verify_answer` | `app/agent/nodes.py` | Mechanical citation check + one structured judge call (`Verdict`); routes repair or appends honest caveats | 1 |
| `conversational_response` | `app/agent/nodes.py` | Direct LLM reply for greetings/meta — no retrieval | 1 |

Graph wiring: `app/agent/graph.py`. State schema: `app/agent/state.py`.

## Design rules

1. **No hardcoded knowledge.** Intent detection, query rewriting, tool selection, and claim verdicts are all structured LLM output. There are no keyword lists, stop-word tables, or scoring thresholds anywhere in the pipeline.
2. **Provenance first.** Every piece of evidence is a typed `Evidence` record with source type, name, URL, and date. Answers must cite `[E#]` keys that resolve to real evidence; the citation validator (`app/agent/citation_validator.py`) enforces this deterministically.
3. **Honest uncertainty.** Claims the judge cannot verify are listed under *Caveats* in the answer with `final_status = "answered_with_caveats"`. The agent never silently guesses.
4. **Bounded self-correction.** When verification finds gaps a targeted search could fix, the judge emits `repair_queries` and the graph loops `gather → generate → verify` once (`settings.MAX_REPAIR_PASSES`). Contradictions never loop — they go straight to caveats.
5. **Cheap checks first.** Deterministic citation validation runs before the LLM judge; only unresolved questions reach the model.

## Chat dynamics & temporal awareness

- **Conversation history** (`messages`) is included in classify, generate, and conversational prompts. Followups ("what about the growth?") are resolved by the planner into standalone rewritten queries.
- **Cross-turn memory:** verified facts persist between turns as `EvidenceState` (`app/agent/evidence_state.py`), serialized into each interaction's `routing_path` and reloaded on the next query. Only claim-backed evidence carries forward; unresolved items reset each turn.
- **Temporal grounding:** current date/time (+ user timezone/location from `request_context`) is injected into every prompt. The planner extracts `temporal_focus`; the generator prefers period-matching evidence and states each figure's period.

## Retrieval & ranking

- **Documents:** `app/documents/service.py` — ChromaDB vector search + BM25 over per-chat collections (Nomic embeddings).
- **Web:** `app/agent/search_tool.py` — SearXNG (self-hosted) → Wikipedia → Tavily, with a per-user daily Tavily budget.
- **Ranking:** BM25 pre-filter → FlashRank cross-encoder (`app/agent/reranker.py`, `app/agent/context_assembly.py`), then a token-budgeted fill for the generator context. Recency breaks score ties. No hand-tuned authority weights.

## LLM providers

`app/documents/clients.py` resolves planner/generator/verifier clients per request:

- Providers: OpenRouter, Google AI Studio, Groq — server env keys or per-user keys (Settings UI).
- Fallback chains walk primary → configured fallbacks → other providers. Placeholder keys (`"fill this later"`) are rejected locally by `valid_key()` so they can never produce confusing remote 401s.
- Models are config, not code: `OPENROUTER_*_MODEL` / `*_FALLBACKS` in `.env`. Free-tier defaults are set for easy testing.

## API surface

All under `/api/v1/agent` (see `app/agent/router.py`):

- `POST /chats/{id}/query` — JSON response with `answer`, `claims`, `citations`, `verification_errors`, `final_status`.
- `POST /chats/{id}/query_stream` — SSE: node status events, streamed tokens, provenance push.
- `WS /ws/{chat_id}` — WebSocket streaming variant.
- Chat CRUD, history, messages, usage endpoints.

## Configuration

Key knobs in `.env` (reference: `.env.example`):

| Variable | Purpose |
|---|---|
| `OPENROUTER_*_MODEL`, `*_FALLBACKS` | Planner/generator/verifier models + fallback chains |
| `MAX_SEARCHES`, `MAX_RETRIEVALS`, `MAX_REPAIR_PASSES` | Loop budgets |
| `QUERY_TIMEOUT_SECONDS` | Whole-query deadline (0 = disabled) |
| `SEARXNG_URL`, `TAVILY_API_KEY` | Web search backends |
| `GROQ_KEY`, `GOOGLE_AI_API_KEY`, `OPENROUTER_API_KEY` | Server provider keys |

## Local development
```bash
make up         # Postgres (:5433) + SearXNG (:8888) via Docker, waits for readiness
make migrate    # alembic upgrade head
make api        # FastAPI on :8000
make web        # Next.js on :3000
```

With no working SMTP in non-production, registration OTPs are echoed back as
`debug_otp` in the register/resend API responses. Full guide: `docs/LOCAL_DEV.md`.


## Testing & evals

```bash
pytest tests/ -q                      # full suite (LLM calls faked; no network)
INTEGRATION_TEST=1 pytest tests/m     # live-key integration tests
python -m evals.run_eval              # golden-set citation checks
python evaluation/test_rag.py         # live RAG quality harness
```

Pipeline tests live in `tests/test_agent_pipeline.py`; citation validator tests in `tests/test_citation_and_golden.py`.
