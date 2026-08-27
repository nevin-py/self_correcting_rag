import enum
import json
import time
import uuid as _uuid
import logging
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import case as sql_case, delete as sql_delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db, get_session_factory
from app.core.config import settings
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats, Agent_interact
from app.agent.message_models import ChatMessage
from app.agent.graph import rag_app, create_initial_state
from app.agent.state import Evidence, Claim, EvidenceState
from app.agent.evidence_state import (
    build_evidence_state,
    merge_evidence_state,
    load_evidence_state_from_text,
    serialize_for_storage,
)
from app.agent.schemas import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    QueryRequest,
    QueryResponse,
    CitationResponse,
    ClaimResponse,
    InteractionResponse,
    InteractionListResponse,
    MessageResponse,
    MessageListResponse,
    UsageSummary,
)
from app.documents.service import estimate_tokens
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

# Split out of this module (routes stay here, logic lives there):
from app.agent.chat_service import (  # noqa: F401
    _delete_chroma_for_chat,
    _delete_chat_children,
    _verify_chat_ownership,
    MAX_HISTORY_MESSAGES,
    MAX_PRIOR_EVIDENCE,
    _load_history,
    _load_prior_evidence_state,
    _finalize_evidence_state,
    _load_prior_evidence_summary,
    _store_messages,
    _log_interaction,
)
from app.agent.streaming import (  # noqa: F401
    NODE_LABELS,
    _build_initial_state,
    _normalize_verification_errors,
    _stream_query,
    _citations_payload,
    _claims_payload,
    _build_trajectory,
)

router = APIRouter(prefix="/agent", tags=["Agent"])

# Heavy endpoint — 10 queries/min per user (each triggers multiple LLM calls)
def _user_key_func(request: Request) -> str:
    """Rate limit by JWT user ID, falling back to IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            import jwt as pyjwt
            from app.core.config import settings as cfg
            token = auth[7:]
            decoded = pyjwt.decode(token, cfg.SECRET_KEY, algorithms=[cfg.ALGORITHM])
            return f"user:{decoded['sub']}"
        except Exception:
            pass
    return f"ip:{get_remote_address(request)}"


_query_limiter = Limiter(key_func=_user_key_func)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────




# ──────────────────────────────────────────────────────────────────────────────
# Chat CRUD
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/chats", response_model=ChatResponse, status_code=201)
async def create_chat(
    body: ChatCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat session (rate-limited).

    Reuses an existing empty chat only when the requested title matches —
    a POST with a different title must never be silently redirected to
    an unrelated session.
    """
    from datetime import timedelta
    from sqlalchemy import exists, func
    from app.core.usage import record_usage, count_events_in_last

    empty = await db.execute(
        select(Chats)
        .where(Chats.user_id == current_user.user_id)
        .where(Chats.title == body.title)
        .where(~exists(select(ChatMessage.id).where(ChatMessage.chat_id == Chats.chat_id)))
        .order_by(Chats.created_at.desc())
        .limit(1)
    )
    existing_empty = empty.scalar_one_or_none()
    if existing_empty:
        return existing_empty

    total = await db.execute(
        select(func.count()).select_from(Chats).where(Chats.user_id == current_user.user_id)
    )
    total_n = int(total.scalar_one() or 0)
    creates = await count_events_in_last(db, current_user.user_id, "chat_create", timedelta(hours=1))
    recent_chats = await db.execute(
        select(func.count()).select_from(Chats).where(
            Chats.user_id == current_user.user_id,
            Chats.created_at >= func.now() - timedelta(hours=1),
        )
    )
    recent_n = int(recent_chats.scalar_one() or 0)
    if total_n >= settings.MAX_CHATS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Chat limit reached ({settings.MAX_CHATS_PER_USER} total).",
        )
    if creates >= settings.MAX_CHAT_CREATES_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Chat create limit reached ({settings.MAX_CHAT_CREATES_PER_HOUR}/hour).",
        )
    if recent_n >= settings.MAX_CHAT_CREATES_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Chat create limit reached ({settings.MAX_CHAT_CREATES_PER_HOUR}/hour).",
        )

    chat = Chats(user_id=current_user.user_id, title=body.title)
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
    await record_usage(db, current_user.user_id, "chat_create")
    logger.info("Chat created: chat_id=%s user_id=%s title=%s", chat.chat_id, current_user.user_id, body.title)
    return chat


@router.get("/chats", response_model=ChatListResponse)
async def list_chats(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List chats for the current user, newest first, with pagination."""
    result = await db.execute(
        select(Chats)
        .where(Chats.user_id == current_user.user_id)
        .order_by(Chats.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    chats = result.scalars().all()
    return ChatListResponse(chats=chats)


@router.get("/chats/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: _uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single chat by ID (must belong to current user)."""
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat






@router.post("/chats/purge")
async def purge_all_chats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete every chat (messages, interactions, ingest logs) and the user's Chroma collection."""
    result = await db.execute(select(Chats).where(Chats.user_id == current_user.user_id))
    chats = list(result.scalars().all())
    for chat in chats:
        await _delete_chat_children(db, chat.chat_id)
        await db.delete(chat)
    from app.auth.models import UsageEvent
    await db.execute(sql_delete(UsageEvent).where(UsageEvent.user_id == current_user.user_id))
    await db.commit()
    _delete_chroma_for_chat(current_user.user_id, None)
    logger.info("Purged %d chats for user_id=%s", len(chats), current_user.user_id)
    return {"deleted": len(chats)}


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: _uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat and all its interactions (cascade)."""
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    await _delete_chat_children(db, chat_id)
    await db.delete(chat)
    await db.commit()
    _delete_chroma_for_chat(current_user.user_id, chat_id)
    logger.info("Chat deleted: chat_id=%s user_id=%s", chat_id, current_user.user_id)




# ──────────────────────────────────────────────────────────────────────────────
# Conversation memory helpers
# ──────────────────────────────────────────────────────────────────────────────













# ──────────────────────────────────────────────────────────────────────────────
# Query (the core RAG endpoint)
# ──────────────────────────────────────────────────────────────────────────────


@_query_limiter.limit("10/minute")
@router.post("/chats/{chat_id}/query", response_model=QueryResponse)
async def query_agent(request: Request,
    chat_id: _uuid.UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    session_factory=Depends(get_session_factory),
):
    """
    Send a message to the self-correcting RAG agent.

    Session lifecycle:
      1. Short-lived session for ownership check — closed immediately.
      2. Graph runs — zero DB connections held.
      3. Short-lived session for interaction logging — closed immediately.
    """
    # ── Step 1: Ownership check + load history + prior evidence (short-lived session) ────
    history = []
    prior_evidence_state: EvidenceState | None = None
    user_credentials: dict = {}
    async with session_factory() as verify_session:
        from app.core.usage import enforce_query_rate, record_usage
        from app.settings.router import load_user_provider_credentials

        await enforce_query_rate(verify_session, current_user.user_id)
        await _verify_chat_ownership(verify_session, chat_id, current_user.user_id)
        history = await _load_history(verify_session, chat_id)
        prior_evidence_state = await _load_prior_evidence_state(verify_session, chat_id)
        user_credentials = await load_user_provider_credentials(
            verify_session, current_user.user_id
        )
        await record_usage(verify_session, current_user.user_id, "query")
    turn = (prior_evidence_state.turn + 1) if prior_evidence_state else 1
    logger.info("Query started: chat_id=%s user_id=%s history=%d msgs prior_state=%s turn=%d message=%s...",
                chat_id, current_user.user_id, len(history),
                "yes" if prior_evidence_state else "no", turn, body.message[:80])

    # ── Step 2: RAG graph — zero DB connections held ─────────────────────
    initial_state = _build_initial_state(
        query=body.message,
        user_id=current_user.user_id,
        chat_id=chat_id,
        history=history,
        provider=body.provider,
        prior_evidence_state=prior_evidence_state,
        user_credentials=user_credentials,
    )

    start_time = time.perf_counter()
    try:
        if settings.QUERY_TIMEOUT_SECONDS and settings.QUERY_TIMEOUT_SECONDS > 0:
            final_state = await asyncio.wait_for(
                rag_app.ainvoke(initial_state),
                timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
        else:
            final_state = await rag_app.ainvoke(initial_state)
    except asyncio.TimeoutError:
        logger.error(
            "Query timeout for chat %s after %ds (user: %s)",
            chat_id,
            settings.QUERY_TIMEOUT_SECONDS,
            current_user.user_id,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Your request timed out after {settings.QUERY_TIMEOUT_SECONDS} seconds. "
                "Please try a simpler question."
            ),
        )
    except Exception as e:
        logger.exception("RAG graph failed for chat %s", chat_id)
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed: {str(e)}",
        )
    elapsed = time.perf_counter() - start_time
    latency_ms = round(elapsed * 1000, 2)
    logger.info("Query completed: chat_id=%s latency=%.0fms steps=%d", chat_id, latency_ms, final_state.get("graph_steps", 0))

    answer = final_state.get("answer", "")
    if not answer:
        answer = "I was unable to generate an answer. Please try rephrasing your question."

    # Build & merge the persistent cross-turn evidence state, then serialize it
    # into the trajectory so the next turn can load it.
    merged_state = _finalize_evidence_state(final_state, prior_evidence_state, turn)
    trajectory = _build_trajectory(final_state) + "\n" + serialize_for_storage(merged_state)

    evidence = final_state.get("evidence", [])
    claims = final_state.get("claims", [])
    verification_errors = _normalize_verification_errors(final_state.get("verification_errors", []))
    citations = [
        CitationResponse(
            evidence_id=ev.evidence_id,
            text=ev.text[:500],
            source_type=ev.source_type.value,
            source_name=ev.source_name,
            source_url=ev.source_url,
            source_date=ev.source_date,
            metric_type="unknown",
            metric_value="",
            geographic_scope="unknown",
            geography="",
            year_period="",
            temporal_qualifier="unknown",
            source_quality="unknown",
        )
        for ev in evidence[:10]
    ]
    claim_responses = [
        ClaimResponse(
            claim_id=c.claim_id,
            text=c.text,
            status=c.status.value,
            claim_type="fact",
            evidence_ids=c.evidence_ids,
            contradicting_evidence_ids=[],
            reasoning=c.reasoning,
        )
        for c in claims
    ]
    provider_used = final_state.get("provider_used", "unknown")
    provenance = {
        "citations": [c.model_dump(mode="json") for c in citations],
        "claims": [c.model_dump(mode="json") for c in claim_responses],
        "conflicts": [],
        "final_status": final_state.get("final_status", "answered"),
        "latency_ms": latency_ms,
        "provider_used": provider_used,
        "verification_errors": verification_errors,
    }

    # ── Step 3: Log interaction + store messages (fresh session) ─────────
    async with session_factory() as log_session:
        meta = await _log_interaction(
            db=log_session,
            chat_id=chat_id,
            user_input=body.message,
            agent_output=answer,
            routing_path=trajectory,
            latency=elapsed,
            provenance=provenance,
            provider_used=provider_used,
        )
        await _store_messages(
            log_session,
            chat_id,
            body.message,
            answer,
            provenance=provenance,
            token_estimate=meta.get("token_metric"),
            estimated_cost_usd=meta.get("estimated_cost_usd"),
        )

    return QueryResponse(
        answer=answer,
        chat_id=chat_id,
        latency_ms=latency_ms,
        provider_used=provider_used,
        final_status=final_state.get("final_status", "answered"),
        claims=claim_responses,
        citations=citations,
        conflicts=[],
        verification_errors=verification_errors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming query — SSE for real-time node status
# ──────────────────────────────────────────────────────────────────────────────







@_query_limiter.limit("10/minute")
@router.post("/chats/{chat_id}/query_stream")
async def query_agent_stream(
    request: Request,
    chat_id: _uuid.UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    session_factory=Depends(get_session_factory),
):
    """Streaming query endpoint — returns SSE events for real-time node status."""
    user_credentials: dict = {}
    async with session_factory() as verify_session:
        from app.core.usage import enforce_query_rate, record_usage
        from app.settings.router import load_user_provider_credentials

        await enforce_query_rate(verify_session, current_user.user_id)
        await _verify_chat_ownership(verify_session, chat_id, current_user.user_id)
        user_credentials = await load_user_provider_credentials(
            verify_session, current_user.user_id
        )
        await record_usage(verify_session, current_user.user_id, "query")

    return StreamingResponse(
        _stream_query(
            chat_id,
            current_user.user_id,
            body.message,
            session_factory,
            provider=body.provider,
            user_credentials=user_credentials,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )










# ──────────────────────────────────────────────────────────────────────────────
# Interaction history
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/chats/{chat_id}/history", response_model=InteractionListResponse)
async def get_chat_history(
    chat_id: _uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the interaction history for a chat, oldest first, with pagination."""
    # Verify chat ownership first
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    result = await db.execute(
        select(Agent_interact)
        .where(Agent_interact.chat_id == chat_id)
        .order_by(Agent_interact.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    interactions = result.scalars().all()
    return InteractionListResponse(interactions=interactions)


@router.get("/chats/{chat_id}/messages", response_model=MessageListResponse)
async def get_chat_messages(
    chat_id: _uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get conversation messages for a chat, oldest first."""
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence.asc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return MessageListResponse(messages=messages)


@router.get("/chats/{chat_id}/usage", response_model=UsageSummary)
async def get_chat_usage(
    chat_id: _uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Estimated token + USD cost for one chat (from logged interactions)."""
    from sqlalchemy import func

    await _verify_chat_ownership(db, chat_id, current_user.user_id)
    result = await db.execute(
        select(
            func.coalesce(func.sum(Agent_interact.token_metric), 0),
            func.coalesce(func.sum(Agent_interact.estimated_cost_usd), 0.0),
            func.count(),
        ).where(Agent_interact.chat_id == chat_id)
    )
    tokens, cost, count = result.one()
    return UsageSummary(
        token_total=int(tokens or 0),
        estimated_cost_usd=float(cost or 0.0),
        interaction_count=int(count or 0),
        chat_id=chat_id,
    )


@router.get("/llm-traces")
async def get_llm_traces(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recent LLM call traces + per-model aggregates (latency, tokens, error rate)."""
    from app.observability.models import LLMCallTrace

    recent = (await db.execute(
        select(LLMCallTrace)
        .order_by(LLMCallTrace.created_at.desc())
        .limit(limit)
    )).scalars().all()

    agg_rows = (await db.execute(
        select(
            LLMCallTrace.role,
            LLMCallTrace.model,
            func.count().label("calls"),
            func.sum(func.cast(LLMCallTrace.status == "ok", sa.Integer)).label("ok"),
            func.avg(LLMCallTrace.latency_ms).label("avg_latency"),
            func.sum(LLMCallTrace.prompt_tokens_est).label("prompt_tokens"),
            func.sum(LLMCallTrace.completion_tokens_est).label("completion_tokens"),
            func.sum(sql_case((LLMCallTrace.status == "ok", 1), else_=0)).label("ok"),
        )
        .group_by(LLMCallTrace.role, LLMCallTrace.model)
        .order_by(func.count().desc())
    )).all()

    return {
        "aggregates": [
            {
                "role": r.role,
                "model": r.model,
                "calls": int(r.calls),
                "ok": int(r.ok or 0),
                "error_rate": round(1 - (int(r.ok or 0) / int(r.calls)), 3) if r.calls else 0,
                "avg_latency_ms": round(float(r.avg_latency or 0), 1),
                "prompt_tokens_est": int(r.prompt_tokens or 0),
                "completion_tokens_est": int(r.completion_tokens or 0),
            }
            for r in agg_rows
        ],
        "recent": [
            {
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "role": t.role,
                "node": t.node,
                "provider": t.provider,
                "model": t.model,
                "attempt": t.attempt,
                "status": t.status,
                "latency_ms": t.latency_ms,
                "prompt_tokens_est": t.prompt_tokens_est,
                "completion_tokens_est": t.completion_tokens_est,
                "error": (t.error or "")[:160],
            }
            for t in recent
        ],
    }


@router.get("/context-window")
async def get_context_window_config(current_user: User = Depends(get_current_user)):
    """Return configured soft context window size for the UI meter."""
    return {"context_window_tokens": settings.CONTEXT_WINDOW_TOKENS}


@router.get("/usage", response_model=UsageSummary)
async def get_user_usage(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Estimated token + USD cost across all of the current user's chats."""
    from sqlalchemy import func

    result = await db.execute(
        select(
            func.coalesce(func.sum(Agent_interact.token_metric), 0),
            func.coalesce(func.sum(Agent_interact.estimated_cost_usd), 0.0),
            func.count(),
        )
        .select_from(Agent_interact)
        .join(Chats, Chats.chat_id == Agent_interact.chat_id)
        .where(Chats.user_id == current_user.user_id)
    )
    tokens, cost, count = result.one()
    return UsageSummary(
        token_total=int(tokens or 0),
        estimated_cost_usd=float(cost or 0.0),
        interaction_count=int(count or 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket streaming
# ──────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/{chat_id}")
async def ws_query(websocket: WebSocket, chat_id: _uuid.UUID):
    """
    WebSocket endpoint for streaming query results.

    Protocol:
      Client sends: {"message": "..."}
      Server sends: {"type": "status", "node": "...", "detail": "..."}
                   {"type": "token", "content": "..."}   (partial answer)
                   {"type": "done", "answer": "...", "latency_ms": ...}
                   {"type": "error", "detail": "..."}
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        message = data.get("message", "")
        if not message:
            await websocket.send_json({"type": "error", "detail": "Empty message"})
            return

        # Authenticate via token in first message or query param
        token = data.get("token", "")
        if not token:
            await websocket.send_json({"type": "error", "detail": "Missing token"})
            return

        # Validate token
        import jwt as pyjwt
        from app.core.config import settings as cfg
        try:
            decoded = pyjwt.decode(token, cfg.SECRET_KEY, algorithms=[cfg.ALGORITHM])
            user_id = _uuid.UUID(decoded["sub"])
        except (pyjwt.PyJWTError, ValueError):
            await websocket.send_json({"type": "error", "detail": "Invalid token"})
            return

        # Verify chat ownership
        async with get_session_factory()() as session:
            await _verify_chat_ownership(session, chat_id, user_id)

        # Stream graph execution
        await websocket.send_json({"type": "status", "node": "start", "detail": "Processing started"})

        provider = data.get("provider", "auto")
        initial_state = _build_initial_state(query=message, user_id=user_id, chat_id=chat_id, provider=provider)
        start_time = time.perf_counter()

        # Use astream for node-level streaming — accumulate answer, never re-run graph
        trajectory_nodes = []
        answer = ""
        async for event in rag_app.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name in ("__start__", "__end__"):
                    continue
                trajectory_nodes.append(node_name)
                await websocket.send_json({
                    "type": "status",
                    "node": node_name,
                    "detail": f"{node_name} completed",
                })

                # Stream tokens from answer generation
                if node_name == "generate_answer" and isinstance(node_output, dict):
                    new_answer = node_output.get("answer", "")
                    if new_answer and new_answer != answer:
                        if answer:
                            await websocket.send_json({"type": "answer_reset"})
                        answer = new_answer
                        chunk_size = 100
                        for i in range(0, len(answer), chunk_size):
                            await websocket.send_json({
                                "type": "token",
                                "content": answer[i:i + chunk_size],
                            })
                elif node_name == "repair_claims" and isinstance(node_output, dict):
                    # Caveated final answer after max attempts
                    repaired = node_output.get("answer")
                    if repaired:
                        answer = repaired

        elapsed = time.perf_counter() - start_time
        latency_ms = round(elapsed * 1000, 2)

        if not answer:
            answer = "I was unable to generate an answer. Please try rephrasing your question."

        # Log interaction using session factory
        async with session_factory() as log_session:
            await _log_interaction(
                db=log_session,
                chat_id=chat_id, user_input=message, agent_output=answer,
                routing_path=" → ".join(trajectory_nodes), latency=elapsed,
            )

        await websocket.send_json({
            "type": "done",
            "answer": answer,
            "latency_ms": latency_ms,
            "trajectory": trajectory_nodes,
        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
