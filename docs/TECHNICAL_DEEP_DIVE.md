# Technical Deep-Dive — Self-Correcting RAG Agent

> Interview-ready explanation of every component: what it does, how it works, and the
> reasoning behind the design. Companion to the shorter `ARCHITECTURE.md`.

---

## 1. The one-paragraph version

A FastAPI service exposes a **LangGraph state machine** that answers questions in four
LLM-mediated steps: **understand the query** (intent + plan), **gather evidence**
(parallel vector search over the user's documents + web search), **generate a cited
answer**, and **verify it** (deterministic citation check + an LLM judge). If the judge
finds fixable gaps, the graph loops back and re-generates once. Everything the model
asserts must resolve to a retrieved evidence record; anything it can't verify is
explicitly listed as a caveat instead of being silently asserted. Conversation memory
and verified facts persist across turns in Postgres.

---

## 2. Stack and why each piece exists

| Layer | Tech | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async-first; SSE/WebSocket streaming for live node status |
| Agent orchestration | **LangGraph** `StateGraph` | Explicit node graph with typed shared state, conditional edges, and cycle support (needed for the repair loop) |
| Relational data | Postgres (Supabase pooler) + SQLAlchemy async | Users, chats, messages, interaction logs, usage counters |
| Vector store | ChromaDB + Nomic embeddings | Per-chat document collections; hybrid vector + BM25 retrieval |
| Reranking | FlashRank (ONNX cross-encoder, CPU) + BM25 prefilter | Real relevance ranking without a hosted reranker |
| Web search | Tavily → SearXNG (self-hosted) → Wikipedia | Quality fallback chain; SearXNG keeps working when the Tavily budget runs out |
| LLMs | OpenRouter primary (`qwen3-30b-a3b`), fallbacks (`gpt-oss-120b`, free models), Groq, Google AI | Multi-provider fallback chain; per-user keys override server keys |
| Frontend | Next.js + Zustand | Consumes the SSE stream; renders live pipeline status, citations, claims |

---

## 3. Request lifecycle (streaming endpoint)

`POST /api/v1/agent/chats/{chat_id}/query_stream`

```
1. JWT auth (OAuth2 bearer) → user_id
2. Rate limit (10 queries/min/user, keyed by JWT sub, IP fallback)
3. Short-lived DB session:
   - enforce_query_rate        (usage_events table, DB-clock window)
   - verify chat ownership
   - _load_history             → last N messages as LangChain messages
   - _load_prior_evidence_state→ parsed from last interaction's routing_path
   - load per-user LLM keys
4. create_initial_state()     → RAGState dict (27 fields, counters zeroed)
5. rag_app.astream(state, stream_mode="updates")
   - each completed node yields {node_name: output_delta}
   - translated into SSE events: status / token / provenance / answer_reset / done
6. After the stream: persist Agent_interact (routing_path + provenance JSON)
   and the user+assistant ChatMessage pair (pg_advisory_xact_lock serializes
   concurrent writes to one chat).
```

**Why short-lived DB sessions around the graph:** the LLM calls take tens of seconds;
holding a pooled connection across them would exhaust the pool under concurrency.
Sessions are opened only for I/O before and after graph execution.

**SSE event contract:**
- `status` — node name + human label + detail (drives the live pipeline panel)
- `token` — 100-char chunks of the generated answer
- `answer_reset` — emitted when a repair pass produces a *replacement* answer; the
  client clears the previously streamed text (otherwise old+new concatenate)
- `provenance` — citations pushed as soon as evidence exists, before generation ends
- `done` — full answer, claims, citations, verification errors, final status

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

**Critical LangGraph lesson (bit us in production):** *a node that omits a key leaves
the previous value in state.* The repair branch wrote `repair_queries`; the terminal
verify path originally didn't mention the key — so the stale value persisted and the
conditional edge looped forever. Rule adopted: **any key that gates a conditional edge
must be explicitly cleared on every path that ends the loop.**

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
    _retrieve_documents(queries),   # Chroma hybrid search, per-chat scope, top_k=30
    _search_web(queries),           # Tavily → SearXNG → Wikipedia, 6 results/query
)                                   # Tavily gated by a daily per-user budget
```

Then, in-process:
1. **Dedupe** by normalized 300-char text prefix.
2. **Rerank** everything against the (rewritten) query with a FlashRank
   cross-encoder; score = rerank score, tiny bounded recency nudge.
3. **Select** top `MAX_EVIDENCE=12` under a generator token budget.
4. **Assign cite keys** `[E1..En]`, store `cite_map: {E1: evidence_id}`.
5. **Sanitize** evidence text — strip any `[E#]` tokens found *inside* retrieved text
   so injected fake citations can't game the verifier.
6. **Merge prior-turn evidence** (established facts from earlier turns) so followups
   can reference them.

On the repair pass, the judge's `repair_queries` replace the plan's queries.

### 4.3 generate_answer — cited generation

System prompt contains: current datetime, the numbered evidence blocks
(`[E3] web: Reuters (2026-07-19): …`), and rules — cite every sentence containing a
number/date/name; never fabricate; report conflicts instead of picking; match the
user's language. Chat history is appended as real messages.

Post-generation (deterministic, no LLM):
- Normalize fullwidth brackets `【E1】` → `[E1]` (a real failure mode).
- `validate_answer_citations`: every citation token must resolve to real evidence;
  every factual-looking sentence *should* carry one. Invalid IDs are flagged inline.

### 4.4 verify_answer — mechanical first, judge second

**Stage 1 (free, deterministic):** citation validation. Errors:
`INVALID_CITATION` (cites evidence that doesn't exist) and `UNCITED_ASSERTION`
(factual sentence, no citation).

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
("Verified facts carried over from earlier turns"), never a raw transcript dump
beyond the normal history window.

---

## 6. LLM orchestration

`resolve_llms(provider, user_credentials)` returns a `ProviderLLMs`:
`planner / generator / verifier` + fallback chains + label.

- **Provider preference:** per-user keys (Settings UI) → server env keys.
  Providers: OpenRouter, Google AI Studio, Groq.
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
- **Model choice is data-driven:** head-to-head latency probes picked
  `qwen3-30b-a3b` (2.4s) over `gpt-oss-120b` (9.8s) for the primary role despite
  similar pricing. Models are env config, never code.

---

## 7. Auth & quotas

- **JWT access + rotating refresh tokens** (hashed in DB); email ownership via OTP.
  OTP codes are hashed, single-purpose, attempt-capped, expiry-checked. In
  non-production, an undeliverable code is echoed as `debug_otp` so the flow is
  testable without mail; in production it never is, and undelivered mail → 503.
- **Usage budgets** (`usage_events` table): queries/hour, chat creates/hour + total,
  Tavily calls/day, ingest tokens/day.
- **Timezone lesson:** event timestamps are DB-clock (`func.now()`), so budget windows
  are computed **in SQL** (`created_at >= now() - interval`). The original Python-side
  UTC comparison silently never enforced caps on any non-UTC Postgres — a bug class
  worth mentioning in interviews (naive-vs-aware datetime across process/DB boundary).
- Chat creation reuses an existing empty chat **only when the title matches** — a POST
  with a different title must never be silently redirected to an unrelated session.

---

## 8. Retrieval quality details

- **Hybrid document search:** Chroma vector + BM25 over per-chat collections
  (`scope="chat"`), distance → 0-1 score.
- **Web chain:** Tavily (best quality) → SearXNG JSON API (self-hosted, unlimited) →
  Wikipedia REST. Per-query failures are logged and skipped; one bad backend never
  fails the turn.
- **Reranking:** BM25 prefilter (top 25) → FlashRank cross-encoder → budget fill.
  Cross-encoder beats embedding similarity because it reads query and document
  *together*; BM25 prefilter keeps it cheap.
- **Prompt-injection defense:** evidence text is sanitized (citation tokens stripped)
  and role prefixes escaped when history is flattened — retrieved content is data,
  never instructions.

---

## 9. Design principles (the "why" an interviewer wants)

1. **One LLM call per job.** Heuristic stand-ins (keyword intent tables, regex entity
   extraction, magic scoring thresholds) were removed deliberately: they're brittle,
   uncalibratable, and domain-locking. A well-prompted structured call is more
   accurate *and* less code. The v1 of this project had ~30 heuristics pretending to
   be reasoning; v2 has zero.
2. **Deterministic checks before LLM checks.** Citation validation is free and
   catches the most objective failures; the expensive judge only sees what survives.
3. **Provenance or silence.** Every claim traces to an evidence ID. No evidence →
   "I don't have enough reliable information", never a confident guess.
4. **Bounded autonomy.** Self-correction is powerful and dangerous; it's capped
   (1 repair pass), contradiction-driven loops are forbidden (they need re-answer,
   not more search), and the whole query has a timeout.
5. **Fail toward honesty.** Every failure path (LLM down, empty evidence, judge
   unreachable) degrades to a *more honest* answer — caveats, "insufficient
   information" — never to a fabricated confident one.

**Tradeoffs to volunteer:**
- LLM judge adds ~2–10s and a call's cost per query — accepted because mechanical
  checks can't judge *semantic* support.
- Free-tier models have high latency variance → fallback chains + timeout discipline
  exist precisely to absorb that.
- `max-instances=1` on Cloud Run keeps cost at free tier at the price of throughput.

---

## 10. Evaluation — three instruments, one verdict

No single eval is trustworthy for an LLM system. This project uses three whose
failure modes don't overlap:

| Instrument | Measures | Noise source | Trust |
|---|---|---|---|
| Deterministic citation validator (`make eval`, 22-case golden set) | citation integrity — every factual sentence resolves to real evidence | none (pure code) | highest; regression gate |
| Live multi-model harness (`evals/harness.py`) | end-to-end correctness (LLM judge 0–5), caveat rate, verified-claim %, latency, tokens | live web result drift + judge variance | medium; coarse signal |
| RAGAS (`evals/ragas_eval.py`, paid non-reasoning judge) | industry-standard faithfulness / relevancy / context precision / correctness | judge quality + stale references on temporal cases | medium; complements the harness |
| Repair A/B (`evals/harness.py --ab-repair`) | does the self-correction loop actually help? Δ judge score + Δ verified-claim share vs `MAX_REPAIR_PASSES=0` | same-judge paired runs reduce drift noise; still small-n | direct measure of the core feature |

### What the numbers say

- Stable factual questions: ~100% correct across all runs (8/8 cases, 5/5).
- Current-events questions: answers independently verified against live sources —
  the pipeline correctly reported facts that post-date any model's training data.
- Citation integrity: zero fabricated citations across ~40 recorded runs.
- Cost/latency: ≈ $0.0008/query, 13–30s end-to-end on the paid primary.

### Hard-won evaluation lessons

1. **Judge knowledge cutoff breaks current-event grading.** The agent answered
   "who won the most recent World Cup" correctly (verified externally); a judge
   with an earlier training cutoff scored it 0/5 from a stale prior. Temporal
   cases now use `evidence_faithful` grading (consistency with cited sources)
   instead of reference comparison.
2. **Citation-id resolution is not entailment.** The answer once cited [E3] for
   "Argentina won 2–1" while E3's text never mentioned the result — the
   mechanical validator only checked that E3 exists. Fix: a local-MiniLM
   support gate (`app/agent/support.py`, `CITATION_SUPPORT_MIN_SIM=0.55`)
   demotes cited sentences whose evidence does not semantically support them;
   markers are stripped from prose and the claim moves to Caveats.
3. **LLM-judge frameworks fail silently.** RAGAS (since removed) scored an
   objectively correct answer ("EU has 27 members") at faithfulness 0.00 because
   its claim-decomposition call failed and unparseable claims count as
   unsupported — and its per-sample fan-out tripped every free-tier judge's rate
   limits into silent NaN. Rule: never trust a metric that gives perfect zeros
   to verified-perfect answers — audit per-case rows before believing
   aggregates.
4. **Run-to-run variance is real.** Live-web evals re-sample search results each
   run; differences under ±0.4 score / ±15pp on 12 cases are noise. Grow the
   case set or average multiple runs before concluding anything small.
5. **Intuition fails measurably.** "More evidence should be better" was tested
   and reverted — richer snippets raised caveats 8%→25% and lowered scores.
   The harness paid for itself in one experiment.

Workflow after any change: `make eval` (fast gate) → `make eval-live
MODELS=...` (real-world check) → compare against `evals/results/*_comparison.md`.

---

## 11. Ops

- **Deploy:** `gcloud builds submit` (Cloud Build → Artifact Registry) →
  `gcloud run deploy scrag-api --env-vars-file .cloudrun.env.yaml`. Env file is the
  single source of Cloud Run config; `RUN_MIGRATIONS=true` applies Alembic on boot.
- **Fail-fast config validation** on boot in production: CORS origins set, SMTP
  present, strong `SECRET_KEY` — the service refuses to start unsafe.
- **Rollback:** redeploy a previous commit (image tag is `:latest`; pin version tags
  for instant revision rollbacks).
- **Tests:** 100 offline tests (LLMs faked — no network in CI) + integration-marked
  live tests + golden-set citation evals (`python -m evals.run_eval`) +
  live multi-model harness with repair A/B (`evals/harness.py`); every LLM call traced
  to `llm_call_traces` (see `GET /api/v1/agent/llm-traces`, optional Langfuse export).

---

## 12. Interview quick-answers

**"How do you prevent hallucinations?"**
Three layers: the generator may only use supplied evidence; a deterministic validator
checks every factual sentence carries a resolvable citation; an LLM judge verifies
semantic support per claim. Anything failing ends up in an explicit Caveats section.

**"How does self-correction work?"**
The verifier emits targeted search queries for fixable gaps; the graph loops
gather→generate→verify once (budget-capped). Contradictions never loop — they either
trigger a clarification question (ambiguous term) or surface as caveats.

**"How do you handle multi-turn context?"**
Chat history in every prompt + a planner that rewrites followups into standalone
queries + a typed evidence memory (verified facts only) persisted per chat and merged
each turn.

**"What was the hardest bug?"**
Two worth telling: (1) LangGraph keeps prior values for keys a node omits — a stale
`repair_queries` caused an infinite repair loop; fixed by explicitly clearing
loop-gating keys on every terminal path, with a replay test. (2) Rate limits never
fired because event timestamps used the DB clock while windows used Python UTC —
fixed by computing windows in SQL.

**"What would you improve next?"**
Streaming token-level generation (today tokens are chunked after full generation);
a learned reranker distilled from judge verdicts; per-user model routing by query
complexity; eval harness that replays production traces against model upgrades.
