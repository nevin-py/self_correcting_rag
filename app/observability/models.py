"""Observability models — LLM call tracing."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class LLMCallTrace(Base):
    """One row per LLM invocation attempt (including failed/fallback attempts).

    Written fire-and-forget: tracing must never break or slow a user query.
    """

    __tablename__ = "llm_call_traces"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    # Context (nullable — evals/offline calls have no chat)
    chat_id: Mapped[uuid.UUID | None] = mapped_column(index=True, nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(index=True, nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="")   # planner | generator | verifier | judge
    node: Mapped[str] = mapped_column(String(64), default="")   # graph node that made the call

    # What was called and how it went
    provider: Mapped[str] = mapped_column(String(32), default="")   # openrouter | google | groq | fake
    model: Mapped[str] = mapped_column(String(128), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=1)        # 1=primary, 2+=fallback index
    status: Mapped[str] = mapped_column(String(16), default="ok")   # ok | error | timeout
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Cost/latency signals
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0)
    completion_chars: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens_est: Mapped[int] = mapped_column(Integer, default=0)      # chars // 4
    completion_tokens_est: Mapped[int] = mapped_column(Integer, default=0)
