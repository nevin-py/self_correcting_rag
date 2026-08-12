# Critical Production Fixes - Implementation Guide

This document provides copy-paste ready fixes for the 8 critical production issues.

---

## Fix #1: Global Graph Step Counter

### Option A (Recommended): Use LangGraph Built-in Recursion Limit

**File**: `app/agent/graph.py`

Replace:
```python
graph = builder.compile()
rag_app = graph
```

With:
```python
from app.core.config import settings

graph = builder.compile()
rag_app = graph

# Apply recursion limit as a wrapper
def _create_bounded_app():
    """Create graph with hard recursion limit to prevent infinite loops."""
    return graph.with_config({
        "recursion_limit": settings.MAX_GRAPH_STEPS,
    })

rag_app = _create_bounded_app()
```

### Option B: Manual Counter in Each Node

Add this helper at the top of `app/agent/nodes.py`:

```python
def _check_step_limit(state: dict) -> dict | None:
    """Check if graph steps exceeded. Returns error state if exceeded, None otherwise."""
    current_steps = state.get("graph_steps", 0)
    max_steps = state.get("max_graph_steps", settings.MAX_GRAPH_STEPS)
    
    if current_steps >= max_steps:
        logger.error("Max graph steps (%d) exceeded, terminating", max_steps)
        return {
            "final_status": "max_steps_exceeded",
            "answer": "Unable to complete your request: Maximum reasoning steps exceeded. Please simplify your question or try again.",
            "graph_steps": current_steps,
        }
    return None
```

Then at the start of EACH node function, add:

```python
def classify_query(state: dict) -> dict:
    # Guard check
    if error_state := _check_step_limit(state):
        return error_state
    
    # ... existing logic ...
    
    # Increment counter in return
    return {
        # ... existing returns ...
        "graph_steps": state.get("graph_steps", 0) + 1,
    }
```

**Apply to**: `classify_query`, `build_plan`, `retrieve_documents`, `search_web`, `assemble_evidence`, `extract_verify_claims`, `generate_answer`, `verify_answer_claims`, `repair_claims`

---

## Fix #2: Request Timeout

**File**: `app/core/config.py`

Add:
```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Agent execution limits
    QUERY_TIMEOUT_SECONDS: int = 120  # 2 minutes max per query (all graph steps combined)
    STREAM_NODE_TIMEOUT_SECONDS: int = 30  # Max time per node in streaming mode
    LLM_CALL_TIMEOUT_SECONDS: int = 30  # Max time for a single LLM call
```

**File**: `app/agent/router.py`

Replace the try-except block in `query_agent` (around line 399):

```python
import asyncio
from app.core.config import settings

# In query_agent function, replace:
#   try:
#       final_state = await rag_app.ainvoke(initial_state)
#   except Exception as e:
#       ...

# With:
try:
    final_state = await asyncio.wait_for(
        rag_app.ainvoke(initial_state),
        timeout=settings.QUERY_TIMEOUT_SECONDS
    )
except asyncio.TimeoutError:
    logger.error(
        "Query timeout for chat %s after %ds (user: %s)",
        chat_id, settings.QUERY_TIMEOUT_SECONDS, current_user.user_id
    )
    raise HTTPException(
        status_code=504,
        detail=f"Your request timed out after {settings.QUERY_TIMEOUT_SECONDS} seconds. "
               f"Please try a simpler question or contact support if this persists."
    )
except Exception as e:
    logger.exception("RAG graph failed for chat %s", chat_id)
    raise HTTPException(
        status_code=500,
        detail=f"Agent pipeline failed: {str(e)}"
    )
```

**File**: `app/agent/nodes.py`

Wrap all LLM calls with timeout:

```python
import asyncio
from app.core.config import settings

def _llm_with_fallback(primary: Any, fallback: Any | None, messages: list, output_schema: Any):
    """Call primary LLM with timeout; on failure invoke fallback."""
    async def _invoke_with_timeout(llm, messages, schema):
        bound = llm.with_structured_output(schema)
        # Note: invoke is sync in langchain, wrap if needed
        return await asyncio.wait_for(
            asyncio.to_thread(bound.invoke, messages),
            timeout=settings.LLM_CALL_TIMEOUT_SECONDS
        )
    
    try:
        return asyncio.run(_invoke_with_timeout(primary, messages, output_schema))
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("Primary LLM failed (%s: %s), trying fallback", type(exc).__name__, exc)
        if fallback is None:
            raise
        return asyncio.run(_invoke_with_timeout(fallback, messages, output_schema))
```

**Note**: If langchain's `.invoke()` is synchronous, you'll need to wrap it with `asyncio.to_thread()` or use `.ainvoke()` if available.

---

## Fix #3: Accurate Token Counting

**File**: `requirements.txt`

Add:
```
tiktoken>=0.5.2
```

**File**: `app/agent/nodes.py`

Replace the `_count_tokens` function:

```python
import tiktoken
from functools import lru_cache

# At module level (after imports)
@lru_cache(maxsize=1)
def _get_tokenizer():
    """Lazy-load tiktoken encoder (expensive operation)."""
    try:
        return tiktoken.get_encoding("cl100k_base")  # GPT-4 / Claude / GPT-3.5-turbo compatible
    except Exception as exc:
        logger.warning("Failed to load tiktoken: %s. Using fallback.", exc)
        return None

def _count_tokens(text: str) -> int:
    """Accurate token count using tiktoken (OpenAI tokenizer).
    
    Falls back to conservative estimate if tiktoken unavailable.
    """
    encoder = _get_tokenizer()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception as exc:
            logger.warning("Token encoding failed: %s. Using fallback.", exc)
    
    # Fallback: more conservative than old //4 (handles non-English better)
    return len(text) // 3
```

---

## Fix #4: Cost Tracking

**File**: `app/agent/state.py`

Add to `RAGState`:
```python
class RAGState(TypedDict, total=False):
    # ... existing fields ...
    
    # Cost & usage tracking
    estimated_cost_usd: Annotated[float, _keep_latest]
    llm_call_count: Annotated[int, _keep_latest]
    input_tokens_total: Annotated[int, _keep_latest]
    output_tokens_total: Annotated[int, _keep_latest]
```

**File**: `app/core/config.py`

Add:
```python
class Settings(BaseSettings):
    # ... existing ...
    
    # Cost limits
    MAX_COST_PER_QUERY_USD: float = 0.50  # $0.50 per query max
    MAX_DAILY_SPEND_PER_USER_USD: float = 10.00  # $10/day per user
    
    # LLM pricing (cost per 1K tokens, update based on your providers)
    LLM_PRICING: dict = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "groq-mixtral": {"input": 0.0002, "output": 0.0002},  # Very cheap
        "default": {"input": 0.01, "output": 0.03},  # Conservative fallback
    }
```

**File**: `app/agent/nodes.py`

Add cost tracking helper:

```python
from app.core.config import settings

def _track_llm_call(
    state: dict,
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> dict:
    """Track LLM usage and estimate cost.
    
    Returns dict with updated cost counters to merge into state.
    """
    pricing = settings.LLM_PRICING.get(model_name, settings.LLM_PRICING["default"])
    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    
    current_cost = state.get("estimated_cost_usd", 0.0)
    new_cost = current_cost + cost
    
    # Guard: check per-query limit
    if new_cost > settings.MAX_COST_PER_QUERY_USD:
        logger.error(
            "Query cost (%.4f USD) exceeds limit (%.2f USD). Terminating.",
            new_cost, settings.MAX_COST_PER_QUERY_USD
        )
        raise ValueError(f"Query cost limit exceeded: ${new_cost:.4f} > ${settings.MAX_COST_PER_QUERY_USD}")
    
    return {
        "estimated_cost_usd": new_cost,
        "llm_call_count": state.get("llm_call_count", 0) + 1,
        "input_tokens_total": state.get("input_tokens_total", 0) + input_tokens,
        "output_tokens_total": state.get("output_tokens_total", 0) + output_tokens,
    }
```

Update all node functions that call LLMs to track cost. Example for `classify_query`:

```python
def classify_query(state: dict) -> dict:
    # ... existing logic ...
    
    # After LLM call:
    result = _llm_with_fallback(...)
    
    # Estimate tokens (rough - LLMs don't always return exact counts)
    input_text = "\n".join([m.content for m in messages])
    input_tokens = _count_tokens(input_text)
    output_tokens = _count_tokens(json.dumps(result.model_dump()))
    
    cost_update = _track_llm_call(state, "gpt-4", input_tokens, output_tokens)
    
    return {
        "classification": result,
        **cost_update,  # Merge cost tracking
    }
```

**File**: `app/agent/router.py`

Add daily spend check in `query_agent`:

```python
from sqlalchemy import func
from app.agent.models import Agent_interact

async def _check_daily_budget(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Raise HTTPException if user has exceeded daily spend limit."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    result = await db.execute(
        select(func.sum(Agent_interact.estimated_cost))
        .where(
            Agent_interact.user_id == user_id,
            Agent_interact.created_at >= today_start
        )
    )
    daily_spend = result.scalar() or 0.0
    
    if daily_spend >= settings.MAX_DAILY_SPEND_PER_USER_USD:
        logger.warning("User %s exceeded daily budget: $%.2f", user_id, daily_spend)
        raise HTTPException(
            status_code=429,
            detail=f"Daily budget exceeded (${daily_spend:.2f}). Limit resets at midnight UTC."
        )

# In query_agent, after ownership check:
@router.post("/query", response_model=QueryResponse)
@_query_limiter.limit("10/minute")
async def query_agent(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ... existing ownership check ...
    
    # Check daily budget
    await _check_daily_budget(db, current_user.user_id)
    
    # ... rest of function ...
```

**Database Migration**: Add `estimated_cost` column to `agent_interact` table:

```sql
-- migrations/add_cost_tracking.sql
ALTER TABLE agent_interact ADD COLUMN estimated_cost REAL DEFAULT 0.0;
CREATE INDEX idx_agent_interact_user_created ON agent_interact(user_id, created_at);
```

---

## Fix #5: Intelligent Repair Routing

**File**: `app/agent/graph.py`

Replace the `_repair_next` function:

```python
def _repair_next(state: RAGState) -> str:
    """Route repair intelligently based on available actions and counter limits.
    
    If all retrieval/search limits are exhausted, go directly to assembly
    (don't loop back to exhausted nodes).
    """
    plan = state.get("plan")
    retrieval_count = state.get("retrieval_count", 0)
    search_count = state.get("search_count", 0)
    max_retrievals = state.get("max_retrievals", settings.MAX_RETRIEVALS)
    max_searches = state.get("max_searches", settings.MAX_SEARCHES)
    
    retrieval_available = retrieval_count < max_retrievals
    search_available = search_count < max_searches
    
    # Try plan steps in order, respecting limits
    if plan and plan.steps:
        for step in plan.steps:
            if step.action == "retrieve_documents" and retrieval_available:
                logger.info("Repair: routing to retrieve_documents (%d/%d used)",
                           retrieval_count, max_retrievals)
                return "retrieve_documents"
            if step.action == "search_web" and search_available:
                logger.info("Repair: routing to search_web (%d/%d used)",
                           search_count, max_searches)
                return "search_web"
    
    # Fallback: try any available action
    if retrieval_available:
        return "retrieve_documents"
    if search_available:
        return "search_web"
    
    # All sources exhausted — go directly to assembly with what we have
    logger.warning(
        "Repair requested but all sources exhausted "
        "(retrievals: %d/%d, searches: %d/%d). Proceeding to assembly.",
        retrieval_count, max_retrievals, search_count, max_searches
    )
    return "assemble_evidence"
```

Update the conditional edge to support `assemble_evidence` as a target:

```python
builder.add_conditional_edges(
    "repair_claims",
    _repair_next,
    {
        "retrieve_documents": "retrieve_documents",
        "search_web": "search_web",
        "assemble_evidence": "assemble_evidence",  # Add this target
    },
)
```

---

## Fix #6: Increment Regeneration Counter in Repair

**File**: `app/agent/nodes.py`

In `repair_claims` function (around line 850), add defensive counter check and increment:

```python
def repair_claims(state: dict) -> dict:
    """Decide whether to repair, give up, or accept the current answer."""
    claims: list[Claim] = state.get("claims", [])
    failed = [c for c in claims if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)]

    regen_count = state.get("regeneration_count", 0)
    max_regen = state.get("max_regenerations", settings.MAX_REGENERATIONS)

    if not failed:
        return {"repair_state": RepairDecision.SATISFACTORY.value, "final_status": "answered"}

    # Defensive check (should be caught by hallucination_router, but belt-and-suspenders)
    if regen_count >= max_regen:
        logger.warning("Max regenerations reached in repair_claims (defensive guard)")
        return {
            "repair_state": RepairDecision.MAX_ATTEMPTS.value,
            "final_status": "max_attempts",
            "answer": _add_caveats(state.get("answer", ""), failed),
            "regeneration_count": regen_count,  # Don't increment
        }

    # ... existing repair logic (building new_steps) ...

    return {
        "repair_state": RepairDecision.REPAIR.value,
        "plan": PlannerOutput(
            classification=state.get("classification") or QueryClassification(),
            steps=deduped,
        ),
        "final_status": "repairing",
        "regeneration_count": regen_count + 1,  # ✅ INCREMENT HERE
    }
```

---

## Fix #7: Circuit Breakers for External Services

**File**: `requirements.txt`

Add:
```
aiobreaker>=1.3.0
```

**File**: `app/documents/service.py` (or create `app/agent/resilience.py`)

```python
from aiobreaker import CircuitBreaker
import logging

logger = logging.getLogger(__name__)

# Circuit breakers for external services (fail_max = failures before opening, timeout = cooldown seconds)
chroma_breaker = CircuitBreaker(fail_max=5, timeout_duration=60, name="chromadb")
tavily_breaker = CircuitBreaker(fail_max=3, timeout_duration=30, name="tavily")
searxng_breaker = CircuitBreaker(fail_max=3, timeout_duration=30, name="searxng")
llm_breaker = CircuitBreaker(fail_max=10, timeout_duration=120, name="llm_provider")

def log_breaker_state(breaker: CircuitBreaker):
    """Log circuit breaker state change."""
    logger.warning(
        "Circuit breaker '%s' state: %s (failures: %d/%d)",
        breaker.name, breaker.current_state, breaker.fail_counter, breaker._fail_max
    )

# Attach listeners
chroma_breaker.add_listener(lambda *_: log_breaker_state(chroma_breaker))
tavily_breaker.add_listener(lambda *_: log_breaker_state(tavily_breaker))
searxng_breaker.add_listener(lambda *_: log_breaker_state(searxng_breaker))
llm_breaker.add_listener(lambda *_: log_breaker_state(llm_breaker))
```

**File**: `app/agent/nodes.py`

Wrap retrieval and search with circuit breakers:

```python
from app.agent.resilience import chroma_breaker, tavily_breaker, searxng_breaker

async def retrieve_documents(state: dict) -> dict:
    # ... existing guard checks ...
    
    try:
        # Wrap with circuit breaker
        @chroma_breaker
        async def _retrieve_with_breaker():
            tasks = [retrieve_chunks(...) for q in retrieval_queries[:3]]
            return await asyncio.gather(*tasks)
        
        results = await _retrieve_with_breaker()
        
        # ... existing processing ...
        
    except Exception as exc:
        # Check if circuit breaker is open (service is down)
        if chroma_breaker.current_state == "open":
            logger.warning("ChromaDB circuit breaker OPEN, skipping retrieval")
            return {
                "evidence": list(state.get("evidence", [])),
                "chunks": [],
                "retrieval_count": state.get("retrieval_count", 0) + 1,
                "retrieval_error": "ChromaDB temporarily unavailable",
            }
        
        logger.exception("Document retrieval failed: %s", exc)
        # Continue with existing evidence (graceful degradation)
        return {
            "evidence": list(state.get("evidence", [])),
            "chunks": [],
            "retrieval_count": state.get("retrieval_count", 0) + 1,
            "retrieval_error": str(exc),
        }
```

Apply similar pattern to `search_web` with `tavily_breaker`/`searxng_breaker`.

---

## Fix #8: State Size Limits

**File**: `app/core/config.py`

Add:
```python
class Settings(BaseSettings):
    # ... existing ...
    
    # State size limits (prevent memory exhaustion)
    MAX_EVIDENCE_ITEMS: int = 50  # Top N evidence items to keep after ranking
    MAX_CLAIMS: int = 30           # Max claims to track
    MAX_CONFLICTS: int = 20        # Max conflicts to track
    MAX_CONTEXT_TOKENS: int = 4000 # Context budget for LLM (was hardcoded at 3500)
```

**File**: `app/agent/nodes.py`

In `assemble_evidence`, cap the lists:

```python
from app.core.config import settings

def assemble_evidence(state: dict) -> dict:
    # ... existing deduplication & ranking ...
    
    unique = rank_evidence(unique, classification)
    
    # ✅ CAP THE EVIDENCE LIST
    if len(unique) > settings.MAX_EVIDENCE_ITEMS:
        logger.info(
            "Capping evidence list from %d to %d items (ranked by relevance)",
            len(unique), settings.MAX_EVIDENCE_ITEMS
        )
        unique = unique[:settings.MAX_EVIDENCE_ITEMS]
    
    # ... conflict detection ...
    
    conflicts = detect_conflicts(unique)
    if len(conflicts) > settings.MAX_CONFLICTS:
        logger.info("Capping conflicts from %d to %d", len(conflicts), settings.MAX_CONFLICTS)
        conflicts = conflicts[:settings.MAX_CONFLICTS]
    
    # ... context building (already token-budgeted) ...
    CONTEXT_TOKEN_BUDGET = settings.MAX_CONTEXT_TOKENS
    
    # ... rest of function ...
```

In `verify_answer_claims`, cap claims:

```python
def verify_answer_claims(state: dict) -> dict:
    # ... existing logic ...
    
    claims = result.claims
    if len(claims) > settings.MAX_CLAIMS:
        logger.info("Capping claims from %d to %d", len(claims), settings.MAX_CLAIMS)
        claims = claims[:settings.MAX_CLAIMS]
    
    # ... rest of function ...
```

---

## Testing the Fixes

After applying all fixes, run:

```bash
# Install new dependencies
pip install tiktoken>=0.5.2 aiobreaker>=1.3.0

# Run DB migration
# (create migration file from the SQL above)

# Run tests
pytest -xvs tests/test_agent.py tests/test_graph.py tests/test_failure_modes.py

# Test with high-load simulation
python scripts/load_test_agent.py --users 10 --queries 50 --check-limits
```

Create a monitoring script:

```python
# scripts/monitor_agent_health.py
import asyncio
from app.agent.resilience import chroma_breaker, tavily_breaker, llm_breaker

async def check_health():
    breakers = [chroma_breaker, tavily_breaker, llm_breaker]
    for b in breakers:
        print(f"{b.name}: {b.current_state} (failures: {b.fail_counter}/{b._fail_max})")

if __name__ == "__main__":
    asyncio.run(check_health())
```

---

## Validation Checklist

After applying fixes, verify:

- [ ] Graph terminates within `MAX_GRAPH_STEPS` (test with `MAX_GRAPH_STEPS=5`)
- [ ] Queries timeout after 120s (test with slow mock LLM)
- [ ] Token counts are accurate (compare tiktoken vs old //4 method)
- [ ] Cost tracking appears in logs and DB
- [ ] Daily budget prevents queries when exceeded
- [ ] Circuit breaker opens after 5 ChromaDB failures
- [ ] Evidence list capped at 50 items in assembled context
- [ ] Repair routing skips exhausted sources
- [ ] Regeneration counter increments in repair_claims

**Monitor in production**:
- Query latency p95 < 30s
- Cost per query < $0.10 average
- Repair loop frequency < 20%
- Circuit breaker open events (should be rare)
