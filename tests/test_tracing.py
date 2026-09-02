"""Tests for LLM call tracing (fire-and-forget, never breaks a query)."""

import time
from types import SimpleNamespace

import pytest

from app.observability import tracing


class _FakeLLM:
    __name__ = "ChatOpenAI"
    model_name = "test/model-x"
    openai_api_base = "https://openrouter.ai/api/v1"


def test_record_persists_row(monkeypatch):
    rows = []
    monkeypatch.setattr(tracing, "_persist", lambda row: rows.append(row))
    monkeypatch.setattr(tracing, "_export_langfuse", lambda row: None)

    t0 = time.perf_counter()
    tracing.record_llm_call(
        role="verifier", llm=_FakeLLM(), attempt=2, status="ok",
        started_at=t0, ended_at=t0 + 1.5,
        prompt_chars=4000, completion_chars=800,
    )
    # Fire-and-forget executor — wait briefly for the row.
    deadline = time.time() + 2
    while not rows and time.time() < deadline:
        time.sleep(0.05)

    assert len(rows) == 1
    r = rows[0]
    assert r["role"] == "verifier"
    assert r["model"] == "test/model-x"
    assert r["provider"] == "openrouter"
    assert r["attempt"] == 2
    assert r["status"] == "ok"
    assert 1400 <= r["latency_ms"] <= 1600
    assert r["prompt_tokens_est"] == 1000   # chars // 4
    assert r["completion_tokens_est"] == 200


def test_trace_context_carries_chat_and_user():
    import uuid as _uuid

    chat_uuid = str(_uuid.uuid4())
    tracing.set_trace_context(node="verify_answer", chat_id=chat_uuid, user_id=None)
    ctx = tracing._trace_context.get()
    assert ctx["node"] == "verify_answer"
    assert ctx["chat_id"] == _uuid.UUID(chat_uuid)
    assert ctx["user_id"] is None
    tracing.clear_trace_context()
    assert tracing._trace_context.get() == {}


def test_model_name_fallbacks():
    assert tracing.current_model_name(SimpleNamespace(model_name="m/one")) == "m/one"
    ns = SimpleNamespace()
    assert tracing.current_model_name(ns) == type(ns).__name__




@pytest.mark.asyncio
async def test_llm_traces_scoped_to_requesting_user(
    client,
    registered_user,
    auth_headers,
    second_user,
    second_auth_headers,
    db_session,
):
    """Regression (Sprint 1): /agent/llm-traces previously returned traces for
    ALL users (and crashed on an undefined `sa`). It must be scoped to the
    authenticated caller."""
    import uuid as _uuid

    from app.observability.models import LLMCallTrace

    session, _engine = db_session
    session.add(LLMCallTrace(
        user_id=_uuid.UUID(registered_user["user_id"]),
        role="planner", model="model-A", status="ok",
    ))
    session.add(LLMCallTrace(
        user_id=_uuid.UUID(second_user["user_id"]),
        role="planner", model="model-B", status="error",
    ))
    await session.commit()

    r1 = await client.get("/api/v1/agent/llm-traces", headers=auth_headers)
    assert r1.status_code == 200
    agg1 = r1.json()["aggregates"]
    assert [a["model"] for a in agg1] == ["model-A"]
    assert agg1[0]["error_rate"] == 0

    r2 = await client.get("/api/v1/agent/llm-traces", headers=second_auth_headers)
    assert r2.status_code == 200
    agg2 = r2.json()["aggregates"]
    assert [a["model"] for a in agg2] == ["model-B"]
    assert agg2[0]["error_rate"] == 1.0
