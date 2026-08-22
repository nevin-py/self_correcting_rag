"""LLM call tracing.

Every LLM invocation attempt is recorded (model, latency, size, outcome) to the
`llm_call_traces` table, fire-and-forget — tracing must never break or slow a
query. Optionally forwarded to Langfuse when LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY are configured.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Set by nodes around each LLM call so traces know which chat/user/node they
# belong to without threading parameters through every helper.
_trace_context: contextvars.ContextVar[dict] = contextvars.ContextVar("llm_trace_context", default={})

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-trace")


def set_trace_context(*, node: str = "", chat_id: Any = None, user_id: Any = None) -> None:
    _trace_context.set({
        "node": node,
        "chat_id": _uuid_or_none(chat_id),
        "user_id": _uuid_or_none(user_id),
    })


def clear_trace_context() -> None:
    _trace_context.set({})


def current_model_name(llm: Any) -> str:
    """Best-effort model id off a LangChain chat model object."""
    for attr in ("model_name", "model", "deployment_name"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return v
    return getattr(llm.__class__, "__name__", "unknown")


def _uuid_or_none(v: Any) -> uuid.UUID | None:
    if isinstance(v, uuid.UUID):
        return v
    if isinstance(v, str) and len(v) == 36:
        try:
            return uuid.UUID(v)
        except ValueError:
            return None
    return None


def record_llm_call(
    *,
    role: str,
    llm: Any,
    attempt: int,
    status: str,
    started_at: float,
    ended_at: float,
    prompt_chars: int,
    completion_chars: int,
    error: str | None = None,
) -> None:
    """Persist one trace row + optional Langfuse event. Never raises."""
    ctx = _trace_context.get() or {}
    row = {
        "chat_id": ctx.get("chat_id"),
        "user_id": ctx.get("user_id"),
        "role": role,
        "node": ctx.get("node", ""),
        "provider": _provider_of(llm),
        "model": current_model_name(llm),
        "attempt": attempt,
        "status": status,
        "error": (error or "")[:500] or None,
        "latency_ms": round((ended_at - started_at) * 1000, 1),
        "prompt_chars": prompt_chars,
        "completion_chars": completion_chars,
        "prompt_tokens_est": prompt_chars // 4,
        "completion_tokens_est": completion_chars // 4,
    }
    try:
        _executor.submit(_persist, row)
    except Exception:
        logger.debug("trace persist submit failed", exc_info=True)
    try:
        _export_langfuse(row)
    except Exception:
        logger.debug("langfuse export failed", exc_info=True)


def _provider_of(llm: Any) -> str:
    cls = llm.__class__.__name__
    base = str(
        getattr(llm, "openai_api_base", "")
        or getattr(llm, "base_url", "")
        or ""
    )
    if "openrouter" in base:
        return "openrouter"
    if base and "ChatOpenAI" in cls:
        return "openai"
    if "Groq" in cls:
        return "groq"
    if "Google" in cls or "Gemini" in cls:
        return "google"
    return cls or "unknown"


def _persist(row: dict) -> None:
    from app.auth.models import UsageEvent  # noqa: F401  (ensures models imported)
    from app.core.database import AsyncLocalSession
    from app.observability.models import LLMCallTrace
    import asyncio

    async def _write():
        async with AsyncLocalSession() as session:
            session.add(LLMCallTrace(**row))
            await session.commit()

    asyncio.run(_write())


# ── Optional Langfuse export ─────────────────────────────────────────────────


def langfuse_enabled() -> bool:
    return bool(
        getattr(settings, "LANGFUSE_PUBLIC_KEY", "")
        and getattr(settings, "LANGFUSE_SECRET_KEY", "")
        and settings.ENVIRONMENT != ""
    )


def _export_langfuse(row: dict) -> None:
    """Forward the trace to Langfuse via its ingestion API (no SDK dependency)."""
    import base64

    pub = getattr(settings, "LANGFUSE_PUBLIC_KEY", "") or ""
    sec = getattr(settings, "LANGFUSE_SECRET_KEY", "") or ""
    if not pub or not sec:
        return

    host = (getattr(settings, "LANGFUSE_HOST", "") or "https://cloud.langfuse.com").rstrip("/")
    now = datetime.now(timezone.utc).isoformat()
    trace_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())

    payload = {
        "batch": [
            {
                "id": trace_id,
                "type": "trace",
                "timestamp": now,
                "name": f"llm.{row['role']}",
                "metadata": {"node": row["node"], "attempt": row["attempt"]},
            },
            {
                "id": obs_id,
                "type": "generation",
                "traceId": trace_id,
                "name": f"{row['role']}.{row['model']}",
                "startTime": now,
                "endTime": now,
                "model": row["model"],
                "usage": {
                    "promptTokens": row["prompt_tokens_est"],
                    "completionTokens": row["completion_tokens_est"],
                    "totalTokens": row["prompt_tokens_est"] + row["completion_tokens_est"],
                },
                "metadata": {
                    "latency_ms": row["latency_ms"],
                    "status": row["status"],
                    "attempt": row["attempt"],
                    "error": row["error"] or "",
                },
                "level": "ERROR" if row["status"] != "ok" else "DEFAULT",
            },
        ]
    }

    auth = base64.b64encode(f"{pub}:{sec}".encode()).decode()

    def _post():
        import httpx

        httpx.post(
            f"{host}/api/public/ingestion",
            json=payload,
            headers={"Authorization": f"Basic {auth}"},
            timeout=5,
        )

    _executor.submit(_safe_post, _post)


def _safe_post(fn) -> None:
    try:
        fn()
    except Exception:
        pass
