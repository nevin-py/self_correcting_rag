# Production Readiness Summary

## TL;DR

**Status**: ⚠️ **NOT PRODUCTION-READY** (8 critical issues)  
**Estimated Time to Production**: 2-3 weeks  
**Risk Level**: HIGH (infinite loops, cost overruns, resource exhaustion)

---

## What Works Well ✅

1. **Solid Architecture**: Modular, testable, well-separated concerns
2. **Comprehensive Testing**: 208 passing tests with deterministic coverage
3. **Cross-Turn State**: Proper evidence persistence without context bloat
4. **Conflict Detection**: Distinguishes genuine contradictions from estimate updates
5. **Source Authority**: DNS-based ranking (not hardcoded per-domain)
6. **Deterministic Verification**: Catches LLM hallucinations with structured checks
7. **Typed Interfaces**: Pydantic models for LLM I/O reduce runtime errors

---

## Critical Gaps 🔴

### 1. **No Global Loop Prevention**
- Graph can execute unlimited steps
- `graph_steps` counter defined but never checked
- **Risk**: Infinite loops, runaway costs
- **Fix**: Add `recursion_limit` to graph compilation (1 line)

### 2. **No Request Timeout**
- Queries can hang forever on stuck LLM calls
- **Risk**: Worker pool exhaustion, poor UX
- **Fix**: Wrap `rag_app.ainvoke()` with `asyncio.wait_for(timeout=120s)`

### 3. **Inaccurate Token Counting**
- Uses `len(text) // 4` which fails for non-English/code/JSON
- Can underestimate by 2-3x → context overflow → LLM failures
- **Fix**: Replace with `tiktoken` library (accurate tokenization)

### 4. **No Cost Tracking**
- Multiple LLM calls per query (6-10 on average)
- No per-query or per-user budget limits
- **Risk**: Uncontrolled spend, abuse
- **Fix**: Track tokens/cost per call, enforce limits

### 5. **Repair Loop Can Bypass Limits**
- `_repair_next()` routing doesn't check counters
- Can route to exhausted retrieval/search nodes repeatedly
- **Fix**: Check `retrieval_count`/`search_count` before routing

### 6. **Missing Circuit Breakers**
- External services (ChromaDB, Tavily) have no failure protection
- **Risk**: Cascading failures, long timeouts
- **Fix**: Add `aiobreaker` circuit breakers (fail_max=5, timeout=60s)

### 7. **Unbounded State Size**
- Evidence/claims/conflicts lists grow without limit
- **Risk**: Memory exhaustion, slow graph execution
- **Fix**: Cap at MAX_EVIDENCE_ITEMS=50 after ranking

### 8. **Weak Rate Limiting**
- Only 10 queries/min per user (no hourly/daily caps)
- No cost-based limits
- **Fix**: Add tiered limits (100/hour, 500/day) + budget checks

---

## Recommended Action Plan

### Phase 1: Critical Blockers (2 days) 🔥
**Must complete before ANY production deployment**

1. Add recursion limit to graph (`graph.with_config(recursion_limit=12)`)
2. Add request timeout (`asyncio.wait_for(..., timeout=120)`)
3. Replace token counter with `tiktoken`
4. Add basic cost tracking (log estimated cost per query)
5. Fix `_repair_next()` to check counters

**Deliverable**: Agent cannot loop infinitely or hang

---

### Phase 2: Scale Readiness (1 week) 📈
**Complete before handling >100 queries/day**

6. Add circuit breakers for ChromaDB, Tavily, LLM providers
7. Cap state sizes (evidence/claims lists)
8. Add tiered rate limits (hourly, daily)
9. Add per-user daily budget limits ($10/day default)
10. Add graceful degradation (continue on partial failures)

**Deliverable**: System can handle traffic spikes and service outages

---

### Phase 3: Production Hardening (1 week) 🛡️
**Complete before production scale (>1000 queries/day)**

11. Add OpenTelemetry tracing (end-to-end query visibility)
12. Set up Prometheus metrics + Grafana dashboards
13. Add idempotency keys (prevent double-charging on retries)
14. Implement cost-based query rejection (preflight check)
15. Add alerting (PagerDuty/Slack) for circuit breakers, high costs, errors

**Deliverable**: Full observability and cost control

---

## Cost Estimate (Current State)

**Without fixes**, a typical query:
- LLM calls: 4-6 (classify, plan, generate, verify, possibly repair)
- Tokens per call: ~2000 input, ~500 output
- Model: GPT-4 Turbo ($0.01 input, $0.03 output per 1K tokens)
- **Cost per query**: $0.10 - $0.30
- **With repair loop**: $0.30 - $0.60

**Risk scenarios**:
- Stuck loop (no recursion limit): **Unlimited cost**
- Expensive model + complex query: **$1.00+**
- Malicious user (no budget limit): **$100+/day**

**After fixes**:
- Hard limits prevent runaway costs
- Budget limits cap per-user spend
- Circuit breakers prevent retry storms
- **Estimated**: $0.05 - $0.15 per query (with cheaper models + limits)

---

## Files to Modify

### Must Edit (Phase 1):
1. `app/agent/graph.py` — add recursion_limit
2. `app/agent/router.py` — add timeout, budget check
3. `app/agent/nodes.py` — fix token counter
4. `app/core/config.py` — add timeout/cost settings
5. `requirements.txt` — add tiktoken

### Should Edit (Phase 2):
6. `app/agent/resilience.py` — NEW: circuit breakers
7. `app/agent/state.py` — add cost tracking fields
8. `app/agent/models.py` — add estimated_cost column

### Can Edit (Phase 3):
9. `app/agent/tracing.py` — NEW: OpenTelemetry
10. `app/agent/metrics.py` — NEW: Prometheus metrics

---

## Test Coverage

Current: **208 passing tests** (deterministic unit/integration)

**Missing tests**:
- [ ] Recursion limit enforcement
- [ ] Timeout behavior
- [ ] Cost limit rejection
- [ ] Circuit breaker state transitions
- [ ] Load testing (concurrent queries)

**Add these tests**:
```bash
# tests/test_agent_limits.py
def test_recursion_limit_enforced()
def test_query_timeout()
def test_cost_limit_rejection()
def test_repair_routing_respects_counters()

# tests/test_resilience.py  
def test_circuit_breaker_opens_after_failures()
def test_graceful_degradation_on_chroma_down()
```

---

## Deployment Checklist

Before deploying to production:

- [ ] All Phase 1 fixes applied and tested
- [ ] Recursion limit verified (test with `MAX_GRAPH_STEPS=5`)
- [ ] Timeout verified (test with slow mock LLM)
- [ ] Token counting accurate (compare tiktoken vs old method)
- [ ] Cost tracking logs visible
- [ ] Daily budget enforced (test with multiple queries)
- [ ] DB migration run (add `estimated_cost` column)
- [ ] Circuit breakers tested (simulate ChromaDB failure)
- [ ] Load test passed (10 concurrent users, 50 queries each)
- [ ] Monitoring dashboard deployed (Grafana)
- [ ] Alerting configured (PagerDuty/Slack)
- [ ] Runbook written (how to respond to high-cost alerts)

---

## Key Metrics to Monitor

### Pre-Deployment (Staging):
- Query latency p95: < 30s
- Cost per query: < $0.20
- Repair loop rate: < 20%
- Test coverage: > 90%

### Post-Deployment (Production):
- Query success rate: > 98%
- Query latency p50: < 10s, p95: < 30s
- Cost per query: < $0.15
- Daily cost per user: < $5.00
- Circuit breaker open events: < 1/day
- Timeout rate: < 1%
- Repair exhaustion rate: < 5%

---

## Risk Assessment

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| Infinite loop | Critical | High | $1000+ bill | ✅ Recursion limit |
| Stuck query | High | Medium | Workers exhausted | ✅ Timeout |
| Cost overrun | High | High | $100+/user/day | ✅ Budget limits |
| Context overflow | Medium | Medium | LLM failures | ✅ tiktoken |
| Repair loop bypass | Medium | Medium | Extra LLM calls | ✅ Counter checks |
| Service cascade | Medium | Low | Total downtime | ✅ Circuit breakers |
| Memory exhaustion | Low | Low | OOM crash | ✅ State size caps |

---

## Conclusion

The self-correcting RAG agent has **excellent architecture and test coverage**, but **8 critical production gaps** must be addressed before deployment.

**Recommendation**: 
1. **DO NOT DEPLOY** until Phase 1 fixes are complete
2. Complete Phase 2 before handling any real users
3. Complete Phase 3 before production scale

**Timeline**: 2-3 weeks from current state to production-ready.

**Effort**: 
- Phase 1: 1 engineer, 2 days
- Phase 2: 1 engineer, 1 week  
- Phase 3: 1 engineer, 1 week

**Total**: ~2.5 weeks (1 engineer) or 1.5 weeks (2 engineers pairing on critical sections)

---

## Quick Start: Apply Phase 1 Fixes Now

```bash
# 1. Install dependencies
pip install tiktoken>=0.5.2

# 2. Apply critical fixes (see CRITICAL_FIXES.md for details)
# - Edit app/agent/graph.py: add recursion_limit
# - Edit app/agent/router.py: add timeout
# - Edit app/agent/nodes.py: replace _count_tokens with tiktoken

# 3. Test
pytest -xvs tests/test_agent.py tests/test_graph.py

# 4. Verify limits work
# Create a test that forces MAX_GRAPH_STEPS=5 and verifies graph stops

# Time investment: ~4 hours
```

See `CRITICAL_FIXES.md` for detailed copy-paste code.
