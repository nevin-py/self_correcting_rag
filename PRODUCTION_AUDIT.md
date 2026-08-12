# Production Readiness Audit: Self-Correcting RAG Agent

**Audit Date**: 2024  
**Scope**: Agent architecture, control flow, guard rails, error handling, state management, resource limits

---

## Executive Summary

**Overall Status**: ⚠️ **PRODUCTION-READY with 8 CRITICAL issues requiring immediate attention**

The system demonstrates solid modular architecture with proper separation of concerns, but has several critical gaps in production safeguards that could lead to infinite loops, resource exhaustion, and financial risk.

---

## Critical Issues (Must Fix Before Production)

### 🔴 CRITICAL #1: Missing Global Graph Step Counter
**Severity**: CRITICAL  
**Impact**: Infinite loop risk, runaway costs

**Problem**:
- `graph_steps` counter is initialized in state but **never incremented**
- `max_graph_steps` (default: 12) is defined but **never checked**
- The repair loop can theoretically run indefinitely if per-action limits aren't hit

**Evidence**:
```bash
# Counter defined but never incremented:
grep "graph_steps.*+=" app/agent/nodes.py  # NO RESULTS
grep "max_graph_steps.*>=" app/agent/nodes.py  # NO RESULTS
```

**Fix Required**:
```python
# In EVERY node function, add at entry:
def classify_query(state: dict) -> dict:
    if state.get("graph_steps", 0) >= state.get("max_graph_steps", 12):
        logger.error("Max graph steps exceeded, terminating")
        return {
            "final_status": "max_steps_exceeded",
            "answer": "Unable to complete: maximum reasoning steps exceeded.",
            "graph_steps": state.get("graph_steps", 0)
        }
    
    # ... existing logic ...
    
    return {
        # ... existing returns ...
        "graph_steps": state.get("graph_steps", 0) + 1,
    }
```

**OR** (preferred): Use LangGraph's built-in recursion limit:
```python
# In graph.py:
graph = builder.compile()
rag_app = graph.with_config(recursion_limit=settings.MAX_GRAPH_STEPS)
```

---

### 🔴 CRITICAL #2: Unbounded Repair Loop
**Severity**: CRITICAL  
**Impact**: Cost explosion, stuck queries

**Problem**:
The repair loop path is:
```
verify_answer_claims → repair_claims → retrieve_documents|search_web → assemble_evidence → 
extract_verify_claims → generate_answer → verify_answer_claims → ...
```

**Current guards**:
- `max_regenerations` (default: 2) — BUT this is only checked in `hallucination_router` and `repair_claims`
- If `regeneration_count` reaches limit, it returns `max_attempts` — ✅ Good
- BUT: `regeneration_count` is incremented in `generate_answer`, NOT in `repair_claims`

**Gap**: If generate_answer fails to increment counter (exception, etc.), loop never exits.

**Fix Required**:
```python
# In repair_claims, add defensive check:
def repair_claims(state: dict) -> dict:
    regen_count = state.get("regeneration_count", 0)
    max_regen = state.get("max_regenerations", settings.MAX_REGENERATIONS)
    
    # Defensive check (should also be in router, but guard here too)
    if regen_count >= max_regen:
        logger.warning("Max regenerations reached in repair_claims (defensive)")
        return {
            "repair_state": RepairDecision.MAX_ATTEMPTS.value,
            "final_status": "max_attempts",
            "answer": _add_caveats(state.get("answer", ""), failed),
            "regeneration_count": regen_count,  # Don't increment
        }
    
    # ... existing logic ...
    
    return {
        # ... existing returns ...
        "regeneration_count": regen_count + 1,  # Increment here too
    }
```

---

### 🔴 CRITICAL #3: Counter State Not Propagated on Early Exit
**Severity**: HIGH  
**Impact**: Guards bypassed, cost overruns

**Problem**:
When a node hits a limit and returns early (e.g., `retrieve_documents` at line 437), it returns the CURRENT counter value without incrementing:

```python
# Line 437-443 in nodes.py
if state.get("retrieval_count", 0) >= state.get("max_retrievals", settings.MAX_RETRIEVALS):
    logger.warning("Max retrievals reached; skipping document retrieval")
    return {
        "evidence": list(state.get("evidence", [])),
        "chunks": [...],
        "retrieval_count": state.get("retrieval_count", 0),  # ❌ Not incremented!
    }
```

**Impact**: If the node is called again in the same graph execution (e.g., via repair loop), it will hit the exact same limit check again, but the counter never moves forward. This is actually CORRECT behavior (don't charge a "retrieval" if we skipped it), but it creates a risk: the repair loop might keep returning to this node expecting new data.

**Fix**: This is actually correct (don't increment on skip), but we need to ensure the **router** doesn't keep sending us back to a node that's exhausted. The `_repair_next` routing function doesn't check counters:

```python
# In graph.py, line 147:
def _repair_next(state: RAGState) -> str:
    """Route repair to the first action in the new repair plan."""
    plan = state.get("plan")
    if plan and plan.steps:
        for step in plan.steps:
            if step.action == "retrieve_documents":
                return "retrieve_documents"  # ❌ No guard check!
    return "search_web"  # ❌ No guard check!
```

**Fix Required**:
```python
def _repair_next(state: RAGState) -> str:
    """Route repair intelligently based on counters."""
    plan = state.get("plan")
    retrieval_count = state.get("retrieval_count", 0)
    search_count = state.get("search_count", 0)
    max_retrievals = state.get("max_retrievals", settings.MAX_RETRIEVALS)
    max_searches = state.get("max_searches", settings.MAX_SEARCHES)
    
    if plan and plan.steps:
        for step in plan.steps:
            if step.action == "retrieve_documents" and retrieval_count < max_retrievals:
                return "retrieve_documents"
            if step.action == "search_web" and search_count < max_searches:
                return "search_web"
    
    # Both exhausted — go to assembly with what we have
    logger.warning("Repair requested but all sources exhausted, proceeding to assembly")
    return "assemble_evidence"  # Bypass retrieval/search, go direct to assembly
```

**ALSO**: The `_repair_next` function is defined locally in graph.py but not used as a proper conditional edge. The current code uses:
```python
builder.add_conditional_edges(
    "repair_claims",
    _repair_next,
    {"retrieve_documents": "retrieve_documents", "search_web": "search_web"},
)
```
This needs to also support `"assemble_evidence"` as a possible target.

---

### 🔴 CRITICAL #4: No Request Timeout
**Severity**: HIGH  
**Impact**: Stuck requests, resource exhaustion

**Problem**:
- `rag_app.ainvoke(initial_state)` has **no timeout**
- A stuck LLM call (network hang, provider outage) will hold the request forever
- FastAPI workers will be exhausted

**Fix Required**:
```python
# In router.py, query_agent function:
import asyncio

try:
    final_state = await asyncio.wait_for(
        rag_app.ainvoke(initial_state),
        timeout=settings.QUERY_TIMEOUT_SECONDS  # Add to config, e.g., 120s
    )
except asyncio.TimeoutError:
    logger.error("Query timeout for chat %s after %ds", chat_id, settings.QUERY_TIMEOUT_SECONDS)
    raise HTTPException(
        status_code=504,
        detail=f"Query timed out after {settings.QUERY_TIMEOUT_SECONDS} seconds",
    )
except Exception as e:
    logger.exception("RAG graph failed for chat %s", chat_id)
    raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")
```

**Also add to config.py**:
```python
class Settings(BaseSettings):
    # ... existing ...
    QUERY_TIMEOUT_SECONDS: int = 120  # 2 minutes max per query
    STREAM_NODE_TIMEOUT_SECONDS: int = 30  # 30s max per node in streaming mode
```

---

### 🔴 CRITICAL #5: Token Counter is Naive
**Severity**: MEDIUM  
**Impact**: Context overflow, LLM failures

**Problem**:
```python
# Line 556 in nodes.py
def _count_tokens(text: str) -> int:
    return len(text) // 4  # ❌ Wildly inaccurate for non-English, code, JSON
```

**Issues**:
- Dividing character count by 4 is a rough heuristic for English prose
- Fails for: code blocks, JSON, Hindi/Chinese characters, special formatting
- Can underestimate by 2-3x for technical content
- `CONTEXT_TOKEN_BUDGET = 3500` might actually be 7000+ tokens → LLM context overflow

**Fix Required**:
Use `tiktoken` (OpenAI's tokenizer):
```python
import tiktoken

# At module level
_tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude compatible

def _count_tokens(text: str) -> int:
    """Accurate token count using tiktoken."""
    try:
        return len(_tokenizer.encode(text))
    except Exception as exc:
        # Fallback to conservative estimate
        logger.warning("Token count failed, using conservative fallback: %s", exc)
        return len(text) // 3  # More conservative than //4
```

**Add to requirements.txt**:
```
tiktoken>=0.5.1
```

---

### 🔴 CRITICAL #6: Missing Cost Tracking
**Severity**: HIGH  
**Impact**: Uncontrolled costs, abuse

**Problem**:
- Multiple LLM calls per query (classify, plan, generate, verify, potentially repair)
- No cost estimation or tracking
- No per-user budget limits
- No alerting on high-cost queries

**Fix Required**:
```python
# Add to state.py:
class RAGState(TypedDict):
    # ... existing ...
    estimated_cost_usd: float  # Track cumulative cost
    llm_call_count: int        # Track number of LLM invocations

# In nodes.py, wrap all LLM calls:
def _track_llm_call(state: dict, model: str, input_tokens: int, output_tokens: int) -> dict:
    """Update cost tracking in state."""
    # Rough pricing (update with actual provider rates)
    cost_per_1k = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        # ... etc
    }
    pricing = cost_per_1k.get(model, {"input": 0.01, "output": 0.03})
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    
    return {
        "estimated_cost_usd": state.get("estimated_cost_usd", 0.0) + cost,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }

# Add alerting:
if state.get("estimated_cost_usd", 0) > settings.MAX_COST_PER_QUERY:
    logger.error("Query cost exceeded limit: $%.4f", state["estimated_cost_usd"])
    raise HTTPException(status_code=429, detail="Query cost limit exceeded")
```

---

### 🟡 CRITICAL #7: No Circuit Breaker for External Services
**Severity**: MEDIUM  
**Impact**: Cascading failures

**Problem**:
- `retrieve_chunks` (ChromaDB), `search_structured` (Tavily/SearXNG), LLM calls have no circuit breaker
- If ChromaDB is down, every query will wait for full timeout
- Retry logic is ad-hoc (`_llm_with_fallback` only tries once)

**Fix Required**:
Use a circuit breaker pattern (e.g., `aiobreaker`):
```python
from aiobreaker import CircuitBreaker

chroma_breaker = CircuitBreaker(fail_max=5, timeout_duration=60)
tavily_breaker = CircuitBreaker(fail_max=3, timeout_duration=30)

@chroma_breaker
async def retrieve_chunks_with_breaker(*args, **kwargs):
    return await retrieve_chunks(*args, **kwargs)

# In retrieve_documents:
try:
    results = await asyncio.gather(*tasks)
except Exception as exc:
    if chroma_breaker.current_state == "open":
        logger.warning("ChromaDB circuit breaker open, skipping retrieval")
        return {"evidence": [], "chunks": [], "retrieval_count": state.get("retrieval_count", 0)}
    raise
```

---

### 🟡 CRITICAL #8: Inadequate Rate Limiting
**Severity**: MEDIUM  
**Impact**: Abuse, cost overruns

**Problem**:
- Current limit: 10 queries/min per user (line 58 in router.py)
- No daily/monthly caps
- No cost-based limits
- No distinction between cheap (cached) vs expensive (regeneration) queries

**Fix Required**:
```python
# Add tiered rate limits:
@router.post("/query")
@_query_limiter.limit("10/minute")  # Existing
@_query_limiter.limit("100/hour")   # Add hourly cap
@_query_limiter.limit("500/day")    # Add daily cap
async def query_agent(...):
    # ...

# Add cost-based limit (check against daily user spend in DB)
async def _check_user_budget(db: AsyncSession, user_id: uuid.UUID, estimated_cost: float):
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.sum(Agent_interact.estimated_cost))
        .where(Agent_interact.user_id == user_id, Agent_interact.created_at >= today_start)
    )
    daily_spend = result.scalar() or 0.0
    if daily_spend + estimated_cost > settings.MAX_DAILY_SPEND_PER_USER:
        raise HTTPException(status_code=429, detail="Daily budget exceeded")
```

---

## High-Priority Issues (Fix Before Scale)

### 🟡 HIGH #9: Missing Graceful Degradation
**Impact**: Poor UX on partial failures

**Problem**:
- If document retrieval fails (`retrieve_chunks` exception), the entire node fails
- No fallback to web-only mode
- No "answer with caveats" mode

**Fix**: Wrap retrieval/search in try-except and continue with partial evidence:
```python
# In retrieve_documents:
try:
    # ... retrieval logic ...
except Exception as exc:
    logger.exception("Document retrieval failed, continuing with existing evidence")
    return {
        "evidence": list(state.get("evidence", [])),
        "chunks": [],
        "retrieval_count": state.get("retrieval_count", 0) + 1,
        "retrieval_error": str(exc),  # Track for debugging
    }
```

---

### 🟡 HIGH #10: State Size Unbounded
**Impact**: Memory exhaustion, slow graph execution

**Problem**:
- `evidence`, `claims`, `conflicts` lists grow without bound
- A query that hits max_retrievals=3 + max_searches=2 could accumulate 50+ evidence items
- Cross-turn evidence state compounds this
- LangGraph passes the entire state dict through every node

**Current situation**:
```python
# No limits on:
state["evidence"]  # Could be 100+ items
state["claims"]     # Could be 50+ claims
state["conflicts"]  # Could be 20+ conflicts
```

**Fix Required**:
```python
# After assemble_evidence, cap the evidence list:
def assemble_evidence(state: dict) -> dict:
    # ... existing logic ...
    
    # Rank and keep only top N
    MAX_EVIDENCE_ITEMS = 50
    unique = rank_evidence(unique, classification)[:MAX_EVIDENCE_ITEMS]
    
    # ... context building ...
```

---

### 🟡 HIGH #11: No Observability/Tracing
**Impact**: Difficult debugging, no performance insights

**Problem**:
- Basic logging only
- No distributed tracing (OpenTelemetry)
- No metrics (Prometheus)
- No structured logging
- Can't trace a query through the graph visually

**Fix Required**:
Add OpenTelemetry tracing:
```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

# In each node:
@tracer.start_as_current_span("classify_query")
def classify_query(state: dict) -> dict:
    span = trace.get_current_span()
    span.set_attribute("query.text", state["query"][:100])
    # ... logic ...
    span.set_attribute("classification.needs_documents", classification.needs_documents)
    return result
```

---

### 🟡 HIGH #12: Deterministic Checks Too Late
**Impact**: Wasted LLM calls

**Problem**:
`audit_claims` (verification.py) runs **after** generate_answer, which means:
1. Generate answer (expensive LLM call)
2. Verify claims (expensive LLM call)
3. Run deterministic checks (cheap, catches obvious errors)
4. If errors found → repair → regenerate (another expensive call)

**Better flow**:
Run deterministic checks on **evidence** before generation:
```python
# In assemble_evidence, before returning:
evidence_quality_issues = audit_evidence_quality(unique, classification)
if len(evidence_quality_issues) > 3:
    logger.warning("Evidence quality issues: %s", evidence_quality_issues)
    # Optionally trigger another search/retrieval here

return {
    "evidence": unique,
    "evidence_quality_warnings": evidence_quality_issues,  # Pass to generator
    # ...
}
```

---

## Medium-Priority Issues (Fix for Robustness)

### 🟠 MEDIUM #13: Race Condition in Message Storage
**Partially Fixed**: Advisory lock added (line 338 in router.py) ✓  
**Remaining issue**: Lock is on chat_id hash, which could collide (MD5 → int64 modulo)

**Better approach**: Use row-level locking or a sequence table per chat.

---

### 🟠 MEDIUM #14: No Idempotency Keys
**Impact**: Duplicate charges on retry

**Problem**: If a client retries a failed request, the entire graph re-runs → double cost.

**Fix**: Add idempotency key support:
```python
@router.post("/query")
async def query_agent(
    body: QueryRequest,
    idempotency_key: str = Header(None, alias="Idempotency-Key"),
    # ...
):
    if idempotency_key:
        # Check if already processed
        cached = await _check_idempotency_cache(idempotency_key)
        if cached:
            return cached
    # ... process ...
    await _store_idempotency_cache(idempotency_key, response, ttl=3600)
```

---

### 🟠 MEDIUM #15: Evidence Deduplication is Text-Only
**Impact**: Near-duplicates with slight rewording aren't caught

**Current**:
```python
# Line 566 in nodes.py
seen_texts = set()
for ev in combined:
    if ev.text in seen_texts:
        continue
```

**Better**: Use embeddings or fuzzy hashing (simhash):
```python
from datasketch import MinHash

def _compute_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=128)
    for word in text.lower().split():
        m.update(word.encode())
    return m

# Deduplicate by similarity threshold (e.g., 0.9)
```

---

### 🟠 MEDIUM #16: No Schema Validation on LLM Outputs
**Impact**: Crashes on malformed LLM responses

**Current**: `.with_structured_output()` is used, but LLMs can still return invalid values (e.g., negative scores, invalid enum strings).

**Fix**: Add Pydantic validators:
```python
class QueryClassification(BaseModel):
    needs_documents: bool
    needs_web: bool
    # ...
    
    @validator("needs_documents", "needs_web")
    def validate_bools(cls, v):
        if not isinstance(v, bool):
            raise ValueError("Must be boolean")
        return v
```

---

## Low-Priority Issues (Polish)

### 🔵 LOW #17: Hardcoded Magic Numbers
- `CONTEXT_TOKEN_BUDGET = 3500` (line 608)
- `MAX_HISTORY_MESSAGES = 20` (line 229 router.py)
- `MAX_PRIOR_EVIDENCE = 5` (line 230 router.py)
- `top_k=10` (line 460 nodes.py)

**Fix**: Move to `settings` (config.py) for tunability.

---

### 🔵 LOW #18: Inconsistent Logging
- Some nodes use `logger.info`, others use `logger.warning` for similar events
- No structured logging (JSON format for production)

**Fix**: Use `structlog` for consistent kvp logging.

---

### 🔵 LOW #19: No Monitoring Dashboard
**Fix**: Add Grafana dashboard with:
- Query latency (p50, p95, p99)
- Cost per query
- Repair loop frequency
- LLM call distribution
- Error rates by node

---

## Architecture Strengths (Keep These)

✅ **Modular design**: Separate concerns (normalization, ranking, conflicts, etc.)  
✅ **Deterministic verification layer**: `audit_claims` catches LLM hallucinations  
✅ **Cross-turn state management**: `EvidenceState` preserves context without bloat  
✅ **Typed prompts**: Pydantic models for LLM I/O  
✅ **Conflict detection**: Distinguishes genuine contradictions from status updates  
✅ **Source authority scoring**: DNS-based + org allowlist (not hardcoded per-domain)  
✅ **Geographic generalization**: LLM-driven, no hardcoded state lists  
✅ **Explicit search queries**: Includes metric acronyms + geography  
✅ **Comprehensive test coverage**: 208 tests (deterministic)

---

## Recommended Fix Priority

### Phase 1 (Pre-Production Blockers): 1-2 days
1. ✅ Add global step counter check (use recursion_limit)
2. ✅ Add request timeout (120s)
3. ✅ Fix token counting (use tiktoken)
4. ✅ Add cost tracking + alerting
5. ✅ Fix _repair_next routing (check counters)

### Phase 2 (Scale Readiness): 3-5 days
6. ✅ Add circuit breakers (aiobreaker)
7. ✅ Cap state size (MAX_EVIDENCE_ITEMS)
8. ✅ Add tiered rate limits (hourly, daily)
9. ✅ Add graceful degradation (continue on partial failure)
10. ✅ Add OpenTelemetry tracing

### Phase 3 (Production Hardening): 1 week
11. ✅ Run deterministic checks earlier (pre-generation)
12. ✅ Add idempotency keys
13. ✅ Improve deduplication (embeddings)
14. ✅ Add cost-based user limits
15. ✅ Set up monitoring dashboard

---

## Conclusion

The self-correcting RAG system has a **solid foundation** with excellent modular architecture and comprehensive test coverage. However, it has **8 critical production gaps** that must be addressed before deploying at scale:

1. Missing global step counter → infinite loop risk
2. Unbounded repair loop → cost explosion
3. Counter propagation gaps → guard bypass
4. No request timeout → stuck requests
5. Naive token counting → context overflow
6. No cost tracking → uncontrolled spend
7. No circuit breakers → cascading failures
8. Weak rate limiting → abuse vulnerability

**Recommendation**: **DO NOT DEPLOY TO PRODUCTION** until Phase 1 fixes (critical blockers) are implemented and tested. Phase 2 should be completed before handling significant traffic.

**Estimated effort**: 2 days (Phase 1) + 5 days (Phase 2) + 1 week (Phase 3) = **~2.5 weeks** to production-ready.
