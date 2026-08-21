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
from sqlalchemy import delete as sql_delete
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

def _build_initial_state(
    query: str,
    user_id: _uuid.UUID,
    chat_id: _uuid.UUID,
    provider: str = "auto",
    history: list | None = None,
    prior_evidence_state: EvidenceState | None = None,
    user_credentials: dict | None = None,
) -> dict:
    """Build the initial LangGraph state via the graph's canonical factory."""
    return create_initial_state(
        query=query,
        user_id=user_id,
        chat_id=chat_id,
        provider=provider,
        messages=history or [],
        user_credentials=user_credentials or {},
        prior_evidence_state=prior_evidence_state,
    )



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


def _delete_chroma_for_chat(user_id: _uuid.UUID, chat_id: _uuid.UUID | None = None) -> None:
    """Best-effort vector cleanup: one chat or the whole user collection."""
    try:
        from app.documents.clients import get_chroma_client

        client = get_chroma_client()
        if client is None:
            return
        name = f"user_{user_id.hex[:16]}"
        collection = client.get_or_create_collection(name=name)
        if chat_id is None:
            client.delete_collection(name)
        else:
            collection.delete(where={"chat_id": str(chat_id)})
    except Exception:
        logger.warning("Chroma cleanup failed for user=%s chat=%s", user_id, chat_id, exc_info=True)


async def _delete_chat_children(db: AsyncSession, chat_id: _uuid.UUID) -> None:
    """Remove FK children before deleting chats (DB FKs are ON DELETE RESTRICT)."""
    from app.documents.models import IngestionLog

    await db.execute(sql_delete(ChatMessage).where(ChatMessage.chat_id == chat_id))
    await db.execute(sql_delete(Agent_interact).where(Agent_interact.chat_id == chat_id))
    await db.execute(sql_delete(IngestionLog).where(IngestionLog.chat_id == chat_id))


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


async def _verify_chat_ownership(
    db: AsyncSession,
    chat_id: _uuid.UUID,
    user_id: _uuid.UUID,
) -> None:
    """Verify chat exists and belongs to user. Raises 404 if not."""
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")


# ──────────────────────────────────────────────────────────────────────────────
# Conversation memory helpers
# ──────────────────────────────────────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 20  # last N message pairs to include in context
MAX_PRIOR_EVIDENCE = 5     # max evidence items to carry forward from prior turns


async def _load_history(session: AsyncSession, chat_id: _uuid.UUID) -> list:
    """Load recent conversation history as LangChain messages."""
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence.desc())
        .limit(MAX_HISTORY_MESSAGES * 2)
    )
    rows = list(reversed(result.scalars().all()))  # oldest first
    messages = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            messages.append(AIMessage(content=row.content))
        elif row.role == "system":
            messages.append(SystemMessage(content=row.content))
    return messages


async def _load_prior_evidence_state(session: AsyncSession, chat_id: _uuid.UUID) -> EvidenceState | None:
    """Load the structured cross-turn evidence state from the last interaction.

    The state is stored as a compact JSON block appended to the interaction's
    routing_path. Returns None when there is no prior state.
    """
    result = await session.execute(
        select(Agent_interact)
        .where(Agent_interact.chat_id == chat_id)
        .order_by(Agent_interact.created_at.desc())
        .limit(1)
    )
    last_interaction = result.scalar_one_or_none()
    if not last_interaction:
        return None
    routing_path = last_interaction.routing_path or ""
    return load_evidence_state_from_text(routing_path)


def _finalize_evidence_state(
    final_state: dict,
    prior_state: EvidenceState | None,
    turn: int,
) -> EvidenceState:
    """Build the turn's evidence state and merge it with the prior state."""
    evidence: list[Evidence] = final_state.get("evidence", [])
    claims: list[Claim] = final_state.get("claims", [])
    current = build_evidence_state(evidence, claims, turn=turn)
    return merge_evidence_state(prior_state, current)


async def _load_prior_evidence_summary(session: AsyncSession, chat_id: _uuid.UUID) -> str:
    """Build a summary of key evidence from prior turns to carry forward.

    This prevents cross-turn evidence state loss by including the most important
    evidence from the last interaction as context for the current query.
    """
    # Get the last interaction to extract its evidence metadata
    result = await session.execute(
        select(Agent_interact)
        .where(Agent_interact.chat_id == chat_id)
        .order_by(Agent_interact.created_at.desc())
        .limit(1)
    )
    last_interaction = result.scalar_one_or_none()
    if not last_interaction:
        return ""

    # Parse routing_path to extract prior evidence if stored
    # We store evidence metadata in the routing_path as JSON
    routing_path = last_interaction.routing_path or ""
    try:
        import json as _json
        # Check if routing_path contains evidence metadata (new format)
        if routing_path.startswith("{"):
            data = _json.loads(routing_path)
            prior_evidence = data.get("evidence", [])
            if prior_evidence:
                lines = ["PRIOR TURN EVIDENCE (from previous conversation):"]
                for ev in prior_evidence[:MAX_PRIOR_EVIDENCE]:
                    line = f"- {ev.get('metric', 'N/A')} | {ev.get('geography', 'N/A')} | {ev.get('period', 'N/A')} | {ev.get('value', 'N/A')}"
                    if ev.get('temporal'):
                        line += f" ({ev['temporal']})"
                    line += f" [{ev.get('source', 'unknown')}]"
                    lines.append(line)
                return "\n".join(lines)
    except (ValueError, KeyError):
        pass

    return ""


async def _store_messages(
    session: AsyncSession,
    chat_id: _uuid.UUID,
    user_msg: str,
    ai_msg: str,
    *,
    provenance: dict | None = None,
    token_estimate: int | None = None,
    estimated_cost_usd: float | None = None,
    is_ingest_notice: bool = False,
) -> None:
    """Store a user+assistant message pair (assistant row keeps analysis provenance).

    A5: When is_ingest_notice=True, stores a typed metadata flag on the system
    message so downstream detection uses the flag, not text sniffing.
    """
    from sqlalchemy import text
    import hashlib
    import json as _json

    chat_id_hash = hashlib.md5(chat_id.bytes).digest()[:8]
    chat_id_int = int.from_bytes(chat_id_hash, byteorder="big") % (2**63)
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:chat_id)"),
        {"chat_id": chat_id_int},
    )

    result = await session.execute(
        select(ChatMessage.sequence)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence.desc())
        .limit(1)
    )
    last_seq = result.scalar()
    next_seq = (last_seq or 0) + 1

    prov_text = _json.dumps(provenance, default=str) if provenance else None
    session.add(ChatMessage(chat_id=chat_id, role="user", content=user_msg, sequence=next_seq))
    # A5: Store typed ingest signal in provenance_json for system messages
    assistant_prov = dict(provenance) if provenance else {}
    if is_ingest_notice:
        assistant_prov["_ingest_chat_id"] = str(chat_id)
    prov_text_final = _json.dumps(assistant_prov, default=str) if assistant_prov else prov_text
    session.add(
        ChatMessage(
            chat_id=chat_id,
            role="assistant",
            content=ai_msg,
            sequence=next_seq + 1,
            provenance_json=prov_text_final,
            token_estimate=token_estimate,
            estimated_cost_usd=estimated_cost_usd,
        )
    )
    await session.commit()


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

NODE_LABELS = {
    "classify_and_plan": "Understanding & planning",
    "gather_evidence": "Gathering evidence",
    "generate_answer": "Generating answer",
    "verify_answer": "Verifying facts",
    "conversational_response": "Responding",
    "ask_clarification": "Asking for clarification",
}


def _normalize_verification_errors(errors) -> list[dict]:
    """Accept VerificationError objects or already-serialized dicts."""
    out: list[dict] = []
    for e in errors or []:
        if hasattr(e, "to_dict"):
            out.append(e.to_dict())
        elif isinstance(e, dict):
            out.append(e)
    return out


async def _stream_query(
    chat_id: _uuid.UUID,
    user_id: _uuid.UUID,
    message: str,
    session_factory,
    provider: str = "auto",
    user_credentials: dict | None = None,
):
    """SSE generator that yields node-level status events during graph execution."""
    start_time = time.perf_counter()
    # Load prior cross-turn evidence state (short-lived session).
    prior_state: EvidenceState | None = None
    async with session_factory() as _vs:
        await _verify_chat_ownership(_vs, chat_id, user_id)
        prior_state = await _load_prior_evidence_state(_vs, chat_id)
    turn = (prior_state.turn + 1) if prior_state else 1
    initial_state = _build_initial_state(
        query=message,
        user_id=user_id,
        chat_id=chat_id,
        provider=provider,
        prior_evidence_state=prior_state,
        user_credentials=user_credentials or {},
    )
    answer = ""
    accumulated_state: dict = {**initial_state}
    trajectory_nodes = []

    try:
        # Use stream_mode="updates" which gives {node_name: output} per node
        async def _run_stream():
            nonlocal answer, accumulated_state, trajectory_nodes
            async for event in rag_app.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name in ("__start__", "__end__"):
                        continue

                    # Track trajectory
                    if node_name not in trajectory_nodes:
                        trajectory_nodes.append(node_name)

                    # Build rich status details based on node
                    detail = ""
                    if node_name == "classify_and_plan" and isinstance(node_output, dict):
                        u = node_output.get("understanding")
                        if u is not None:
                            mode = getattr(u, "mode", None)
                            detail = f"Mode: {getattr(mode, 'value', mode)}"
                    elif node_name == "gather_evidence" and isinstance(node_output, dict):
                        ev_count = len(node_output.get("evidence", []) or [])
                        detail = f"Found {ev_count} evidence items"
                    elif node_name == "verify_answer" and isinstance(node_output, dict):
                        claims = node_output.get("claims", []) or []
                        failed = [
                            c for c in claims
                            if (c.status.value if hasattr(c.status, "value") else str(c.status))
                            in ("unverified", "contradicted", "uncertain")
                        ]
                        detail = f"Claims: {len(claims)} total, {len(failed)} need attention"

                    # Send status update
                    label = NODE_LABELS.get(node_name, node_name)
                    yield f"event: status\ndata: {json.dumps({'node': node_name, 'label': label, 'detail': detail, 'status': 'running'})}\n\n"

                    # If answer generation completed, stream the answer tokens
                    if node_name == "generate_answer" and isinstance(node_output, dict):
                        new_answer = node_output.get("answer", "")
                        if new_answer and new_answer != answer:
                            if answer:
                                # Repair pass produced a replacement — tell the
                                # client to discard the previously streamed text.
                                yield "event: answer_reset\ndata: {}\n\n"
                            answer = new_answer
                            chunk_size = 100
                            for i in range(0, len(answer), chunk_size):
                                yield f"event: token\ndata: {json.dumps({'content': answer[i:i+chunk_size]})}\n\n"

                    # Accumulate node outputs so we can build the provenance payload without re-running the graph.
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

                    # Push evidence to the UI as soon as it exists (don't wait for verify/done).
                    if node_name in ("gather_evidence", "generate_answer") and isinstance(node_output, dict):
                        ev = accumulated_state.get("evidence") or []
                        if ev:
                            yield (
                                "event: provenance\ndata: "
                                + json.dumps({"citations": _citations_payload(ev), "conflicts": []})
                                + "\n\n"
                            )

        # Overall query timeout (0 / unset = disabled for slow OpenRouter / networks)
        deadline = None
        if settings.QUERY_TIMEOUT_SECONDS and settings.QUERY_TIMEOUT_SECONDS > 0:
            deadline = start_time + settings.QUERY_TIMEOUT_SECONDS
        async for chunk in _run_stream():
            if deadline is not None and time.perf_counter() > deadline:
                raise asyncio.TimeoutError()
            yield chunk

        final_state = accumulated_state
        answer = answer or final_state.get("answer", "") or ""
        if not answer.strip():
            answer = "I was unable to generate an answer. Please try rephrasing your question."

        elapsed = time.perf_counter() - start_time
        trajectory = " → ".join(trajectory_nodes) if trajectory_nodes else "unknown"

        # Persist the structured cross-turn evidence state for the next turn.
        merged_state = _finalize_evidence_state(final_state, prior_state, turn)
        trajectory = trajectory + "\n" + serialize_for_storage(merged_state)
        verification_errors = _normalize_verification_errors(final_state.get("verification_errors", []))

        # Build structured provenance payload from accumulated node outputs.
        evidence = final_state.get("evidence", []) if isinstance(final_state, dict) else []
        claims = final_state.get("claims", []) if isinstance(final_state, dict) else []
        conflicts: list = []
        citations = _citations_payload(evidence)
        claim_payload = _claims_payload(claims)

        provider_used = final_state.get("provider_used")
        latency_ms = round(elapsed * 1000, 2)
        provenance = {
            "citations": citations,
            "claims": claim_payload,
            "conflicts": conflicts,
            "final_status": final_state.get("final_status"),
            "latency_ms": latency_ms,
            "provider_used": provider_used,
            "verification_errors": verification_errors,
            "trajectory": trajectory,
        }

        meta: dict = {}
        # Persist interaction + assistant provenance for Analysis panel restore
        async with session_factory() as log_session:
            meta = await _log_interaction(
                db=log_session,
                chat_id=chat_id,
                user_input=message,
                agent_output=answer,
                routing_path=trajectory,
                latency=elapsed,
                provenance=provenance,
                provider_used=provider_used,
            )
            await _store_messages(
                log_session,
                chat_id,
                message,
                answer,
                provenance=provenance,
                token_estimate=meta.get("token_metric"),
                estimated_cost_usd=meta.get("estimated_cost_usd"),
            )

        yield (
            "event: done\ndata: "
            + json.dumps(
                {
                    "answer": answer,
                    "chat_id": str(chat_id),
                    "latency_ms": latency_ms,
                    "provider_used": provider_used,
                    "final_status": final_state.get("final_status"),
                    "claims": claim_payload,
                    "citations": citations,
                    "conflicts": conflicts,
                    "verification_errors": verification_errors,
                    "trajectory": trajectory,
                    "token_estimate": meta.get("token_metric"),
                    "estimated_cost_usd": meta.get("estimated_cost_usd"),
                }
            )
            + "\n\n"
        )
    except Exception as e:
        # Timeout, GraphRecursionError, or any mid-pipeline failure: keep answer + evidence.
        is_timeout = isinstance(e, asyncio.TimeoutError)
        status = "timeout" if is_timeout else "partial"
        if is_timeout:
            logger.error("Streaming query timeout for chat %s after %ds", chat_id, settings.QUERY_TIMEOUT_SECONDS)
        else:
            logger.exception("Streaming query failed for chat %s", chat_id)

        if answer:
            elapsed = time.perf_counter() - start_time
            evidence = accumulated_state.get("evidence", []) if isinstance(accumulated_state, dict) else []
            claims = accumulated_state.get("claims", []) if isinstance(accumulated_state, dict) else []
            if not claims:
                from app.agent.citation_validator import validate_answer_citations
                try:
                    claims = validate_answer_citations(
                        answer, evidence, cite_map=accumulated_state.get("cite_map") or {}
                    ).claims
                except Exception:
                    claims = []
            citations = _citations_payload(evidence)
            claim_payload = _claims_payload(claims)
            suffix = f" ({status})"
            try:
                provenance = {
                    "citations": citations,
                    "claims": claim_payload,
                    "conflicts": [],
                    "final_status": status,
                    "latency_ms": round(elapsed * 1000, 2),
                    "provider_used": accumulated_state.get("provider_used") if isinstance(accumulated_state, dict) else None,
                }
                async with session_factory() as log_session:
                    meta = await _log_interaction(
                        db=log_session,
                        chat_id=chat_id,
                        user_input=message,
                        agent_output=answer,
                        routing_path=" → ".join(trajectory_nodes) + suffix,
                        latency=elapsed,
                        provenance=provenance,
                        provider_used=provenance.get("provider_used"),
                    )
                    await _store_messages(
                        log_session,
                        chat_id,
                        message,
                        answer,
                        provenance=provenance,
                        token_estimate=meta.get("token_metric"),
                        estimated_cost_usd=meta.get("estimated_cost_usd"),
                    )
            except Exception:
                logger.exception("Failed to persist partial answer after %s for chat %s", status, chat_id)
            yield (
                "event: done\ndata: "
                + json.dumps(
                    {
                        "answer": answer,
                        "chat_id": str(chat_id),
                        "latency_ms": round(elapsed * 1000, 2),
                        "provider_used": accumulated_state.get("provider_used"),
                        "final_status": status,
                        "claims": claim_payload,
                        "citations": citations,
                        "conflicts": [],
                        "verification_errors": [],
                        "trajectory": " → ".join(trajectory_nodes) + suffix,
                    }
                )
                + "\n\n"
            )
        else:
            detail = (
                f"Request timed out after {settings.QUERY_TIMEOUT_SECONDS}s"
                if is_timeout
                else str(e)
            )
            yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"


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


async def _log_interaction(
    db: AsyncSession,
    chat_id: _uuid.UUID,
    user_input: str,
    agent_output: str,
    routing_path: str,
    latency: float,
    *,
    provenance: dict | None = None,
    provider_used: str | None = None,
) -> dict:
    """Write one interaction row. Returns token/cost metrics for message storage."""
    import json as _json
    from app.core.costing import estimate_cost_usd

    tokens = estimate_tokens(user_input) + estimate_tokens(agent_output)
    cost = estimate_cost_usd(tokens, provider_used)
    meta = {
        "token_metric": tokens,
        "estimated_cost_usd": cost,
        "provider_used": provider_used,
    }
    try:
        interaction = Agent_interact(
            chat_id=chat_id,
            user_input=user_input,
            agent_output=agent_output,
            routing_path=routing_path,
            token_metric=tokens,
            latency=round(latency, 4),
            provenance_json=_json.dumps(provenance, default=str) if provenance else None,
            estimated_cost_usd=cost,
            provider_used=provider_used,
        )
        db.add(interaction)
        await db.commit()
    except Exception:
        logger.exception("Failed to log interaction for chat %s", chat_id)
    return meta


def _citations_payload(evidence: list, limit: int = 15) -> list[dict]:
    """Serialize evidence for SSE/API provenance panels."""
    out: list[dict] = []
    for ev in (evidence or [])[:limit]:
        out.append(
            {
                "evidence_id": ev.evidence_id,
                "text": (ev.text or "")[:500],
                "source_type": ev.source_type.value if hasattr(ev.source_type, "value") else str(ev.source_type),
                "source_name": ev.source_name,
                "source_url": ev.source_url,
                "source_date": ev.source_date.isoformat() if ev.source_date else None,
                "metric_type": "unknown",
                "metric_value": "",
                "geographic_scope": "unknown",
                "geography": "",
                "year_period": "",
                "temporal_qualifier": "unknown",
                "source_quality": "unknown",
            }
        )
    return out


def _claims_payload(claims: list) -> list[dict]:
    out: list[dict] = []
    for c in claims or []:
        out.append(
            {
                "claim_id": c.claim_id,
                "text": c.text,
                "status": c.status.value if hasattr(c.status, "value") else str(c.status),
                "claim_type": "fact",
                "evidence_ids": c.evidence_ids,
                "contradicting_evidence_ids": [],
                "reasoning": c.reasoning,
            }
        )
    return out


def _build_trajectory(state: dict) -> str:
    """
    Reconstruct which nodes were visited based on the final state.
    Also stores key evidence metadata for cross-turn carry-forward.
    """
    steps = ["classify_and_plan"]

    if state.get("retrieval_count", 0) > 0:
        steps.append("retrieve_documents")
    if state.get("search_count", 0) > 0:
        steps.append("search_web")

    steps.extend(["generate_answer", "verify_answer"])

    if state.get("repair_count", 0) > 0:
        steps.append(f"repair×{state['repair_count']}")

    trajectory = " → ".join(steps)
    return trajectory


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
