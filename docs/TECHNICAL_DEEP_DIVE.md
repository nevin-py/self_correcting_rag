# Technical Deep-Dive — Self-Correcting RAG Agent

> Interview-ready explanation of every component: what it does, how it works, and the
> reasoning behind the design. Companion to the shorter `ARCHITECTURE.md` and the
> narrative anchor in `docs/project_brief.md`. Everything here reflects the current
> code: stateless pgvector backend, true token streaming, any-provider keys, LLM call
> tracing, signed document links, and the three-instrument eval harness.

---

## 1. The one-paragraph version

A FastAPI service exposes a **LangGraph state machine** that answers questions in four
LLM-mediated steps: **understand the query** (intent + plan), **gather evidence**
(parallel pgvector+BM25 search over the user's documents + layered web search), **generate
a cited answer** (tokens streamed live), and **verify it** (deterministic citation check +
a local MiniLM entailment gate + an LLM judge). If the judge finds fixable gaps, the graph
loops back and re-generates once. Everything the model asserts must resolve to a retrieved
evidence record; anything it can't verify is explicitly listed as a caveat instead of being
silently asserted. The backend is **fully stateless** — relational data, vectors, files,
and traces all live in Postgres — so it deploys to Cloud Run, Render, or HF Spaces with no
local-disk dependencies.

---

## 2. Stack and why each piece exists

| Layer | Tech | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async-first; SSE streaming for live node status and real token deltas |
| Agent orchestration | **LangGraph** `StateGraph` | Explicit node graph with typed shared state, conditional edges, and cycle support (needed for the repair loop) |
| Relational data | Postgres (Supabase/Render/Cloud SQL) + SQLAlchemy async + Alembic | Users, chats, messages, interaction logs, usage counters, LLM traces — all in one DB |
| Vector store | **pgvector** (`<=>` cosine) + Nomic `nomic-embed-text-v1.5` (768 dims) | Stateless: Chroma persisted to local disk, so docs vanished on container restart — breaking Cloud Run / Render / HF Spaces. Only the storage engine moved; embedding semantics are unchanged. Owner-scoping enforced in SQL |
| Hybrid retrieval | BM25 (rank-bm25) prefilter + vector search + **parent-child chunking** | Child chunks (2048/256) retrieve precisely; linked 4× parent chunks are re-injected at search time for context |
| Reranking | FlashRank (ONNX cross-encoder, CPU) | Real relevance ranking — cross-encoders read query and document *together* — without a hosted reranker |
| Web search | Tavily (key auto-rotation) → SearXNG (self-hosted) → direct Wikipedia lookup | Quality fallback chain; SearXNG keeps working when the Tavily budget runs out |
| Claim-support gate | fastembed ONNX MiniLM (local, lazy-loaded) | Entailment check without torch/any API; keyword-cosine fallback keeps the gate safe-directional |
| LLMs | OpenRouter primary (`xiaomi/mimo-v2.5` in config; free `nvidia/nemotron-3-super-120b-a12b:free` in `.env.example`), Google AI, Groq, **any OpenAI-compatible base URL** | Multi-provider fallback chains; per-user Fernet-encrypted keys override server keys |
| Tracing | `llm_call_traces` table + optional Langfuse | Every LLM invocation attempt recorded fire-and-forget (model, node, latency, tokens, outcome) |
| Frontend | Next.js 16 + React 19 + TypeScript + Tailwind 4 + Zustand + TanStack Query | Consumes the SSE stream; renders live pipeline status, streamed tokens, signed citation links, provenance + analysis panels |

---

## 3. Request lifecycle (streaming endpoint)

`POST /api/v1/agent/chats/{chat_id}/query_stream`

```
1. JWT auth (OAuth2 bearer) → user_id
2. Rate limit (queries/min/user, keyed by JWT sub, proxy-aware IP fallback)
3. Short-lived DB session:
   - enforce usage budgets      (usage_events, SQL-clock windows)
   - verify chat ownership
   - _load_history             → last N messages as LangChain messages
   - _load_prior_evidence_state→ parsed from last interaction's routing_path
   - _load_document_inventory  → what docs the chat can cite
   - load per-user LLM keys
4. create_initial_state()     → RAGState (guard counters zeroed)
5. Producer task runs rag_app.astream_events(initial_state, version="v2")
   pushing events into an asyncio queue; the SSE generator drains the queue:
   - status        — node started/finished + elapsed (live pipeline panel)
   - token         — real generator token deltas, streamed as produced
   - answer_reset  — repair pass started after tokens were sent → client clears text
   - provenance    — citations pushed as soon as evidence exists
   - ping          — heartbeat with elapsed_ms (keeps proxies alive)
   - done          — full answer, claims, citations, errors, final status
6. After the stream (new short-lived DB session): persist Agent_interact
   (routing_path + provenance JSON) and the user+assistant ChatMessage pair
   under a pg_advisory_xact_lock serializing concurrent writes to one chat.
```

**Why a producer task + queue:** LangGraph's async stream can drop mid-iteration frames
under backpressure; isolating `astream_events` in its own task and queueing the events
makes the SSE contract reliable. Only `on_chat_model_stream` deltas from the
`generate_answer` node are forwarded — planner/judge streams stay internal — so the user
never sees reasoning-token noise.

**Why short-lived DB sessions around the graph:** the LLM calls take tens of seconds;
holding a pooled connection across them would exhaust the pool under concurrency.
Sessions are opened only for I/O before and after graph execution.

**Why SSE and not WebSocket:** the original WS variant bypassed rate limits and usage
budgets and kept state alive; it was deliberately removed (Sprint 1). SSE is stateless,
proxy-friendly, and pairs with bearer-token auth (no cookie CSRF surface; rotating
`*.vercel.app` subdomains are handled by a wildcard CORS rule).

---

## 4. The graph

```
classify_and_plan ─┬─ conversational_response ─ END
                   ├─ ask_clarification ──────── END
                   └─ gather_evidence ─► generate_answer ─► verify_answer
                                               ▲                │
                                               └── repair ◄─────┘  (≤ MAX_REPAIR_PASSES)
                                                                END
```

**State** is a `TypedDict(total=False)` where every field carries a reducer:
- `_keep_latest` — last write wins (answers, evidence, understanding)
- `_add_to_list` — append (legacy list buffers)

Plus guard counters checked *inside* nodes: `MAX_GRAPH_STEPS=20`, `MAX_SEARCHES=4`,
`MAX_RETRIEVALS=3`, `MAX_REGENERATIONS=2`, `MAX_REPAIR_PASSES=1`,
`MAX_REPAIR_SEARCHES=3` (an independent search-only budget for gap filling).

**Critical LangGraph lesson (bit us in production):** *a node that omits a key leaves
the previous value in state.* The repair branch wrote `repair_queries`; the terminal
verify path originally didn't mention the key — so the stale value persisted and the
conditional edge looped forever. Rule adopted: **any key that gates a conditional edge
must be explicitly cleared on every path that ends the loop.** A regression test replays
the exact incident.

### 4.1 classify_and_plan — one structured call does four jobs

Single LLM call returning a `QueryUnderstanding`:

```python
mode: research | conversational | clarification
rewritten_query: str      # pronouns resolved against chat history → standalone query
needs_documents / needs_web: bool
search_queries: list[str] # 1–3 self-contained queries
temporal_focus: str       # "latest", "2023-24", …
geography: str
clarification_question: str
```

- **Temporal awareness:** current date/time + user timezone are injected into the
  prompt; the model extracts the period the question is about.
- **Chat dynamics:** history is in the prompt; "what about the growth?" becomes
  "growth of \<topic\>" in `rewritten_query`.
- **Output sanitization** (never trust the model blindly):
  - `clarification` without a question ⇒ coerced to `research` + `needs_web=True`
  - `research` with no source flags ⇒ force `needs_web=True`
  - `research` with empty queries ⇒ fill from rewritten/original query
- **Failure fallback:** if every provider fails, degrade to research mode with the raw
  query — the pipeline never starves.

Prompt rule worth quoting in an interview: *"clarification is only for genuine
which-X-do-you-mean ambiguity; unknown acronyms, news, and current events are not
ambiguity — search will resolve them."*

### 4.2 gather_evidence — parallel retrieval, then one ranking pass

```
asyncio.gather(
    _retrieve_documents(queries),   # pgvector cosine + BM25, owner-scoped, top_k=30
    _search_web(queries),           # SearXNG → Wikipedia → Tavily, 6 results/query
)                                   # Tavily gated by a daily per-user budget
```

Then, in-process:
1. **Full-page enrichment** of the top results (`EVIDENCE_FETCH_TOP_N=2`), parsed with
   **lxml under a 2MB input cap** (memory-safety hardening on hostile HTML).
2. **Dedupe** by normalized 300-char text prefix.
3. **Rerank** everything against the (rewritten) query with a FlashRank
   cross-encoder; score = rerank score, tiny bounded recency nudge.
4. **Select** top `MAX_EVIDENCE=12` under a generator token budget.
5. **Assign cite keys** `[E1..En]`, store `cite_map: {E1: evidence_id}`.
6. **Sanitize** evidence text — strip any `[E#]` tokens found *inside* retrieved text
   and escape role prefixes when history is flattened, so injected fake citations or
   role confusion can't game the verifier (retrieved content is data, never instructions).
7. **Merge prior-turn evidence** (established facts from earlier turns) so followups
   can reference them.

On the repair pass, the judge's `repair_queries` replace the plan's queries, within
`MAX_REPAIR_SEARCHES`. Per-backend failures are logged and skipped — one bad backend
never fails the turn.

**Document-side details:**
- Hybrid search: pgvector cosine distance → 0–1 score, fused with BM25; every row is
  filtered by `user_id`/`chat_id` **in SQL**, so cross-tenant leakage is impossible.
- **Parent-child chunking:** children link to 4×-size parents via metadata `parent_id`;
  at search time `_add_parent_context` re-injects parent text — small-rank precision
  without losing surrounding context.
- **File-hash dedupe** (`ingestion_log.file_hash`) prevents double-ingesting the same
  upload; re-uploads reuse existing chunks.

### 4.3 generate_answer — cited generation, streamed live

System prompt contains: current datetime, the numbered evidence blocks
(`[E3] web: Reuters (2026-07-19): …`), and rules — cite every sentence containing a
number/date/name; never fabricate; report conflicts instead of picking; match the
user's language. Chat history is appended as real messages. Tokens are streamed live
(see §3).

Post-generation (deterministic, no LLM):
- Normalize fullwidth brackets `【E1】` → `[E1]` (a real failure mode).
- `validate_answer_citations`: every citation token must resolve to real evidence;
  every factual-looking sentence *should* carry one. Invalid IDs are flagged inline.

### 4.4 verify_answer — mechanical first, gate second, judge last

**Stage 1 (free, deterministic):** citation validation. Errors:
`INVALID_CITATION` (cites evidence that doesn't exist) and `UNCITED_ASSERTION`
(factual sentence, no citation).

**Stage 1.5 (free, local): the claim-support gate** (`app/agent/support.py`). Citation-id
resolution says nothing about whether E3's text *supports* the sentence citing it — a
real incident had the answer cite [E3] for "Argentina won 2–1" while E3 never mentioned
the result. Fix: each cited sentence is embedded against its cited evidence with a local
**fastembed ONNX MiniLM** (lazy singleton, no torch, no API); cosine below
`CITATION_SUPPORT_MIN_SIM=0.55` **strips the citation marker and demotes the claim to a
Caveat**. Failure is deliberately **safe-directional**: a below-threshold claim loses its
markers, never silently stays. If fastembed/the model isn't available, the gate falls
back to keyword-overlap cosine — weaker, but the direction of failure is unchanged
(containers stay small by defaulting to the fallback).

**Stage 2 (one LLM call):** a judge prompt returns a typed `Verdict`:

```python
claims: list[Claim]        # text, status: verified|contradicted|unverified|uncertain,
                           # evidence_ids, reasoning
overall: supported | partial | unsupported
repair_queries: list[str]  # only when a targeted search could fix the gaps
clarification_question: str  # when contradictions stem from an ambiguous term
```

**Routing logic (the interesting part):**

| Condition | Action |
|---|---|
| unverified gaps + repair budget left + nothing contradicted | write `repair_queries`, `repair_count += 1` → loop to gather |
| contradictions + judge found an ambiguous term | `needs_clarification` — ask the user which meaning they meant, listing the conflicting findings |
| contradictions (genuine) | append *Caveats* section, `answered_with_caveats` |
| unverified/partial leftovers | append caveats |
| everything verified | `answered` |

**Why the loop is safe:** budget `MAX_REPAIR_PASSES=1` checked *inside* the node, and
`repair_queries` is **explicitly cleared** on every terminal return (the LangGraph
stale-state lesson). A regression test replays the exact production incident.

**Why caveats instead of silence:** the agent's core promise is honesty. An assertion
the judge couldn't verify is listed under "Caveats:" verbatim — the user sees exactly
what is and isn't backed by evidence.

---

## 5. Cross-turn memory (chat dynamics)

Each turn ends with `build_evidence_state`:
- **established** = evidence records that backed claims the judge marked *verified*
  (capped at 8) — unsupported guesses are never carried forward.
- **unresolved** = claim texts that couldn't be verified (reset each turn; they
  describe the previous answer, not durable facts).

Serialized as a compact JSON block appended to the interaction's `routing_path`;
next turn parses it back. Merging dedupes by normalized text, so re-verifying the
same fact replaces rather than duplicates.

**Deliberate boundary:** raw conversation lives in `ChatMessage` rows; the evidence
state carries only typed, provenance-preserving records. Prompts get a summary block
("Verified facts carried over from earlier turns"), never a raw transcript dump beyond
the normal history window.

---

## 6. LLM orchestration

`resolve_llms(provider, user_credentials)` returns a `ProviderLLMs`:
`planner / generator / verifier` + fallback chains + label.

- **Providers:** OpenRouter, Google AI Studio, Groq, plus **any OpenAI-compatible
  endpoint via custom base URL** (`_make_custom_llm`): Together, Fireworks, DeepSeek,
  xAI, Mistral, Anthropic, and local Ollama models. One credential shape covers them all.
- **Provider preference:** per-user keys (Settings UI, primary → user fallback) →
  server env keys.
- **`valid_key()` guard:** placeholder strings ("fill this later", short, spaced) are
  treated as *no key* at resolution time — a missing credential fails locally and
  explicitly instead of producing a confusing remote 401 deep in a fallback chain.
  A bogus `OPENAI_API_KEY` env var is even removed from `os.environ` because some
  libraries silently pick it up.
- **Reasoning models:** `reasoning: {effort: "low"}` is set on all OpenRouter clients —
  hidden reasoning tokens otherwise blow past timeouts or exhaust `max_tokens`
  (production incident: 45s×2 timeouts on every verify call).
- **`_structured_invoke`** per attempt:
  1. native structured output (`json_schema`) under a 30s timeout
  2. on *empty/parse* failure → raw JSON retry (extract JSON object from prose,
     coerce nulls to schema defaults, validate with Pydantic)
  3. on *timeout* → **skip the retry** (slowness isn't fixed by retrying the same
     model) and raise so `_llm_with_fallback` walks to the next model
- **Model choice is data-driven:** head-to-head latency probes picked the primary
  role's model by measured latency, not marketing. Models are env config, never code.

---

## 7. Tracing & observability

`app/observability/tracing.py` records **every LLM invocation attempt** to
`llm_call_traces`: model, role (planner/generator/verifier), node, chat/user
(via `contextvars` — no parameter threading), latency, prompt/completion token
counts, and outcome (ok/parse-fallback/timeout/error).

- **Fire-and-forget by design:** a 2-worker `ThreadPoolExecutor` writes traces so
  tracing can never break or slow a query.
- Exposed via `GET /api/v1/agent/llm-traces`; optional **Langfuse export** when
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are configured.
- **Fail-fast config validation on boot in production:** CORS origins set, SMTP
  present, strong `SECRET_KEY` — the service refuses to start unsafe.

---

## 8. Auth, quotas & document security

- **JWT access + rotating refresh tokens** (hashed in DB, single-use) with **reuse
  detection**: presenting a revoked token revokes the user's entire token family.
  Refresh tokens are issued as **httpOnly cookies** (`SameSite=None; Secure` in
  production) so browsers never hold them in JS-readable storage; JSON-body tokens
  remain supported for API clients. Password change/reset revokes all live tokens.
- **Email ownership via OTP** (register/reset): codes hashed, attempt-capped,
  expiry-checked, resend-cooldown enforced. In non-production an undeliverable code is
  echoed as `debug_otp` so the flow is testable without mail; in production it never is,
  and undelivered mail → 503. **Logout revokes ALL of the user's refresh tokens** —
  a concurrent tab's in-flight rotation can otherwise commit after logout and re-install
  a live cookie (session resurrection on next open).
- **Per-user LLM keys Fernet-encrypted** under a dedicated `ENCRYPTION_KEY`
  (independent of `SECRET_KEY` rotation); `scripts/reencrypt_user_keys.py` migrates
  legacy rows transparently. Key reveal requires explicit confirmation; masked by default.
- **Usage budgets** (`usage_events` table): queries/hour, chat creates/hour + total,
  Tavily calls/day, ingest tokens/day.
- **DB connections are pooled persistently** even over the Supabase transaction
  pooler (`statement_cache_size=0` handles pgbouncer; `pool_pre_ping` + 5-min recycle
  handle idle eviction). NullPool here opened a fresh TLS handshake per request —
  measured 0.8–2.8s on every auth/CRUD endpoint.
- **Timezone lesson:** event timestamps are DB-clock (`func.now()`), so budget windows
  are computed **in SQL** (`created_at >= now() - interval`). The original Python-side
  UTC comparison silently never enforced caps on any non-UTC Postgres — a bug class
  worth mentioning in interviews (naive-vs-aware datetime across process/DB boundary).
- **Signed document links:** `<a href>` links can't carry `Authorization` headers, so
  stored originals are served via **HMAC-SHA256-signed, expiring URLs**
  (`app/documents/signing.py`): signature scoped to one ingestion id, 30-day TTL,
  verified with `hmac.compare_digest` (constant time). Any mismatch/expiry → 404.
- Chat creation reuses an existing empty chat **only when the title matches** — a POST
  with a different title must never be silently redirected to an unrelated session.
- **Rate limits** via slowapi (`app/core/limiter.py`): proxy-aware IP keys
  (`TRUST_PROXY_HEADERS`), optional Redis storage for multi-instance deploys.

---

## 9. Retrieval & ingestion quality details

- **Hybrid document search:** pgvector cosine + BM25 over owner-scoped rows, distance →
  0-1 score, fused, then parent context re-injected.
- **Ingestion** (`app/documents/service.py`): PDF (pypdf), TXT, Markdown, HTML
  (lxml parser, **2MB input cap** — the `perf: memory-safe HTML parsing` hardening),
  CSV/Excel (pandas tabular → text), JSON, source code, and **images via Groq Vision
  OCR** (async-extracted). Chunking: children 2048 chars / 256 overlap; parents 4×.
- **Web chain:** SearXNG JSON API (self-hosted, unlimited) + year-pinned variant
  without the recency filter → direct Wikipedia article lookup for encyclopedic-topic
  queries → Tavily (best quality) with **key auto-rotation** to `TAVILY_API_KEY_BACKUP`
  on quota failure (`RotatingTavily`). Per-user daily Tavily budget.
- **Reranking:** BM25 prefilter → FlashRank cross-encoder → budget fill.
  Cross-encoder beats embedding similarity because it reads query and document
  *together*; BM25 prefilter keeps it cheap.
- **Prompt-injection defense:** evidence text is sanitized (citation tokens stripped)
  and role prefixes escaped when history is flattened — retrieved content is data,
  never instructions.

---

## 10. Design principles (the "why" an interviewer wants)

1. **One LLM call per job.** Heuristic stand-ins (keyword intent tables, regex entity
   extraction, magic scoring thresholds) were removed deliberately: they're brittle,
   uncalibratable, and domain-locking. A well-prompted structured call is more
   accurate *and* less code. The v1 of this project had ~30 heuristics pretending to
   be reasoning; v2 has zero.
2. **Deterministic checks before LLM checks.** Citation validation and the local
   MiniLM support gate are free and catch the most objective failures; the expensive
   judge only sees what survives.
3. **Provenance or silence.** Every claim traces to an evidence ID. No evidence →
   "I don't have enough reliable information", never a confident guess.
4. **Bounded autonomy.** Self-correction is powerful and dangerous; it's capped
   (1 repair pass + independent search budget), contradiction-driven loops are forbidden
   (they need re-answer, not more search), and the whole query has a timeout.
5. **Fail toward honesty.** Every failure path (LLM down, empty evidence, judge
   unreachable) degrades to a *more honest* answer — caveats, "insufficient
   information" — never to a fabricated confident one.
6. **Stateless beats cheap.** ChromaDB was removed not because pgvector is faster but
   because local-disk persistence made every containerized deploy a data-loss trap.
   "Where does state live?" is the first question for any free-tier deployment.

**Tradeoffs to volunteer:**
- LLM judge adds ~2–10s and a call's cost per query — accepted because mechanical
  checks can't judge *semantic* support.
- Free-tier models have high latency variance → fallback chains + timeout discipline
  exist precisely to absorb that.
- Single-instance constraints on free tiers keep cost at zero at the price of
  throughput; Redis-backed rate-limit storage is ready when that changes.

---

## 11. Evaluation — three instruments, one verdict

No single eval is trustworthy for an LLM system. This project uses three whose
failure modes don't overlap, plus a targeted A/B that measures the core feature
directly. All raw runs are committed under `evals/results/`.

| Instrument | Measures | Noise source | Trust |
|---|---|---|---|
| Deterministic citation validator (`python -m evals.run_eval`, 20-case golden set) | citation integrity — every factual sentence resolves to real evidence | none (pure code) | highest; regression gate |
| Live multi-model harness (`evals/harness.py`) | end-to-end correctness (LLM judge 0–5), caveat rate, verified-claim %, latency, tokens | live web result drift + judge variance | medium; coarse signal |
| RAGAS (`evals/ragas_eval.py`, paid non-reasoning judge `xiaomi/mimo-v2.5`) | industry-standard faithfulness / relevancy / context precision / correctness | judge quality + stale references on temporal cases | medium; complements the harness |
| Repair A/B (`evals/harness.py --ab-repair --repeat 2`) | does the self-correction loop actually help? Δ judge score + Δ verified-claim share vs `MAX_REPAIR_PASSES=0` | same-judge paired runs reduce drift noise; still small-n | direct measure of the core feature |

### What the numbers say (committed runs)

**RAGAS** — full 12-case live run, MiMo judge, **all four metrics complete with 0/12 NaN**:

| Metric | Value | What it measures |
|---|---|---|
| faithfulness | **0.809** | is every claim supported by the retrieved evidence? |
| answer_relevancy | **0.919** | does the answer address the question? |
| context_precision | **0.809** | is the evidence ranked usefully? |
| answer_correctness | 0.46 | factual match vs reference (dataset-bound; see below) |

Why faithfulness is the lead number: correctness grades against a fixed golden
reference, so a *correct, evidence-grounded* answer to a temporal question scores low
when the reference is stale. Faithfulness measures exactly the property this system
optimizes — every claim backed by cited evidence.

**TruthfulQA** (adversarial open-ended, n=30): **faithfulness 0.632, 0/30 NaN**, run on
a **free** model (`nvidia/nemotron-3-super-120b-a12b:free`) at zero LLM cost. TruthfulQA
baits common false beliefs; the system's frequent `answered_with_caveats` (refusing to
assert unverifiable claims) is *the intended honest behavior* — grounding held even
adversarially.

**Repair A/B** (paired live-web runs, repair on/off, 12 cases × 2 repeats, MiMo judge):

- Typical run (`20260826_002720`): repair **3.92** avg /5, 89% verified, 33% caveats
  vs no-repair **3.50**, 77% verified, 50% caveats → **Δ score +0.42, Δ verified +12pp,
  caveat rate roughly halved**.
- Degraded-retrieval run (`20260826_115948`): no-repair **collapses to 1.25** /5
  (refusals under retrieval failure) while repair **rescues to 3.83** — the repair loop
  re-searches and rescues refusals instead of dead-ending. (That run's caveats were 92%
  under retrieval stress — the harness flags the case variance explicitly.)
- The harness also reports unstable cases (score σ ≥ 1.0 across repeats) so noisy
  deltas are treated as noise, never as signal.

**Golden set:** 20 cases (metric naming, geography, citation fixtures) — zero
fabricated citations across ~40 recorded live runs. **Stable factual questions score
~100%**; the pipeline correctly reports facts that post-date any model's training data.

**Cost/latency:** ≈ **$0.0008/query**, 13–30s end-to-end on the paid primary
(harness medians ~20–26s on qwen3-30b-a3b).

**Tests:** **144 offline tests** across 15 suites (pipeline, auth, security hardening,
support gate, tracing, rate limiting, document links, ingestion, provider CRUD) —
LLMs faked, no network in CI; separate `INTEGRATION_TEST=1` live-key suite.

### Hard-won evaluation lessons

1. **Judge knowledge cutoff breaks current-event grading.** The agent answered
   "who won the most recent World Cup" correctly (verified externally); a judge
   with an earlier training cutoff scored it 0/5 from a stale prior. Temporal
   cases now use `evidence_faithful` grading (consistency with cited sources)
   instead of reference comparison.
2. **Citation-id resolution is not entailment.** The answer once cited [E3] for
   "Argentina won 2–1" while E3's text never mentioned the result — the
   mechanical validator only checked that E3 exists. Fix: the local-MiniLM
   support gate (`app/agent/support.py`, `CITATION_SUPPORT_MIN_SIM=0.55`)
   demotes cited sentences whose evidence does not semantically support them;
   markers are stripped from prose and the claim moves to Caveats.
3. **LLM-judge frameworks fail silently.** RAGAS (before the hardening) scored an
   objectively correct answer ("EU has 27 members") at faithfulness 0.00 because
   its claim-decomposition call failed and unparseable claims count as
   unsupported — and its per-sample fan-out tripped every free-tier judge's rate
   limits into silent NaN. Rules baked into `ragas_eval.py`: non-reasoning judge
   only, paid endpoint, serialized sub-calls, JSON-fence stripping — and never
   trust a metric that gives perfect zeros to verified-perfect answers (audit
   per-case rows before believing aggregates).
4. **Run-to-run variance is real.** Live-web evals re-sample search results each
   run; the harness reports per-case σ and unstable cases explicitly. Differences
   under ±0.4 score / ±15pp on 12 cases are noise. Grow the case set or average
   multiple runs before concluding anything small.
5. **Intuition fails measurably.** "More evidence should be better" was tested
   and reverted — richer snippets raised caveats 8%→25% and lowered scores.
   The harness paid for itself in one experiment.

Workflow after any change: `python -m evals.run_eval` (fast gate) →
`python -m evals.harness --models <model> --ab-repair --repeat 2` (real-world check) →
compare against `evals/results/*_comparison.md`.

---

## 12. Ops

- **Deploy targets (all stateless — the pgvector migration is what enables them):**
  - **Google Cloud Run:** `cloudbuild.yaml` (Cloud Build → Artifact Registry) →
    `gcloud run deploy scrag-api --env-vars-file .cloudrun.env.yaml`. The env file is
    the single source of Cloud Run config; `RUN_MIGRATIONS=true` applies Alembic on boot.
  - **Render:** `render.yaml` Blueprint — free web service, Supabase-backed Postgres,
    `/health` health check, auto-deploy.
  - **Split deploy:** Oracle Cloud API + Vercel frontend (`docs/DEPLOY_ORACLE_VERCEL.md`).
  - **Hugging Face Space:** `deploy/hf-space` package (Space-tailored Dockerfile).
  - **Self-hosted Docker:** `docker/docker-compose.prod.yml` + Caddy TLS;
    `docker/backup.sh` snapshots Postgres.
- **Rollback:** redeploy a previous commit (image tag is `:latest`; pin version tags
  for instant revision rollbacks).
- **Traces:** every LLM call visible in `llm_call_traces` / `GET /api/v1/agent/llm-traces`
  (optional Langfuse export) — model upgrades are graded against real production calls.

---

## 13. Interview quick-answers

**"How do you prevent hallucinations?"**
Three layers: the generator may only use supplied evidence; a deterministic validator
checks every factual sentence carries a resolvable citation; a local MiniLM gate checks
that the cited evidence *semantically supports* each cited sentence; then an LLM judge
verifies per claim. Anything failing ends up in an explicit Caveats section.

**"How does self-correction work?"**
The verifier emits targeted search queries for fixable gaps; the graph loops
gather→generate→verify once (budget-capped, with an independent search-only budget).
Contradictions never loop — they either trigger a clarification question (ambiguous term)
or surface as caveats. The A/B harness proves the lift: +12pp verified-claim share in a
typical run, and collapsed no-repair runs (1.25/5) rescued to 3.83/5 under degraded
retrieval.

**"How do you handle multi-turn context?"**
Chat history in every prompt + a planner that rewrites followups into standalone
queries + a typed evidence memory (verified facts only, capped at 8) persisted per chat
and merged each turn.

**"Why did you drop ChromaDB?"**
Statelessness. Chroma persisted to local disk, so documents vanished on container
restart — that breaks Cloud Run/Render/HF Spaces deploys. All chunks moved to Postgres
pgvector (same Nomic embeddings, same semantics); owner-scoping is now enforced in SQL,
backups cover everything, and the backend became deploy-anywhere.

**"What was the hardest bug?"**
Two worth telling: (1) LangGraph keeps prior values for keys a node omits — a stale
`repair_queries` caused an infinite repair loop; fixed by explicitly clearing
loop-gating keys on every terminal path, with a replay test. (2) Rate limits never
fired because event timestamps used the DB clock while windows used Python UTC —
fixed by computing windows in SQL.

**"What would you improve next?"**
A learned reranker distilled from judge verdicts; per-user model routing by query
complexity; an eval harness that replays production traces (already captured in
`llm_call_traces`) against model upgrades; multi-instance deploys with Redis-backed
rate-limit storage.
