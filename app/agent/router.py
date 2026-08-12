import enum
import json
import time
import uuid as _uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db, get_session_factory
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats, Agent_interact
from app.agent.message_models import ChatMessage
from app.agent.graph import rag_app
from app.agent.state import PlannerDecision, RepairDecision, Evidence, Claim, EvidenceState
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
)
from app.documents.service import estimate_tokens
from langchain_core.messages import HumanMessage, AIMessage

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

def _build_initial_state(query: str, user_id: _uuid.UUID, chat_id: _uuid.UUID, provider: str = "auto", history: list = None, prior_evidence_summary: str = "", prior_evidence_state: EvidenceState | None = None) -> dict:
    """Build the initial LangGraph state dict for a new query."""
    from app.core.config import settings

    effective_query = query
    if prior_evidence_summary:
        effective_query = f"{query}\n\n[Context from prior conversation turn:]\n{prior_evidence_summary}"

    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "query": effective_query,
        "provider": provider,
        "messages": history or [],
        "chunks": [],
        "search": [],
        "planner_state": PlannerDecision.NOT_ENOUGH,
        "retrieval_queries": [],
        "wiki_queries": [],
        "tavily_queries": [],
        "searxng_queries": [],
        "cross_chat_enabled": False,
        "answer": "",
        "provider_used": "",
        "need_repair": RepairDecision.REPAIR,
        "hallucination_reason": [],
        "max_tries_planner": 0,
        "max_tries_hallucinator": 0,
        "steps_taken": 0,
        "searches_done": 0,
        "retrievals_done": 0,
        "regenerations_done": 0,
        # Business-ready structured state
        "classification": None,
        "plan": None,
        "evidence": [],
        "claims": [],
        "conflicts": [],
        "citation_usage": [],
        "assembled_context": "",
        "evidence_state": None,
        "prior_evidence_state": prior_evidence_state,
        "final_status": "",
        "graph_steps": 0,
        "search_count": 0,
        "retrieval_count": 0,
        "regeneration_count": 0,
        "max_graph_steps": settings.MAX_GRAPH_STEPS,
        "max_searches": settings.MAX_SEARCHES,
        "max_retrievals": settings.MAX_RETRIEVALS,
        "max_regenerations": settings.MAX_REGENERATIONS,
    }



# ──────────────────────────────────────────────────────────────────────────────
# Chat CRUD
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/chats", response_model=ChatResponse, status_code=201)
async def create_chat(
    body: ChatCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat session."""
    chat = Chats(user_id=current_user.user_id, title=body.title)
    db.add(chat)
    await db.commit()
    await db.refresh(chat)
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
    await db.delete(chat)
    await db.commit()
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
    conflicts: list[dict] = final_state.get("conflicts", [])
    current = build_evidence_state(evidence, claims, conflicts, turn=turn)
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
    session: AsyncSession, chat_id: _uuid.UUID, user_msg: str, ai_msg: str
) -> None:
    """Store a user+assistant message pair.
    
    Uses advisory lock to prevent race conditions on concurrent inserts.
    """
    from sqlalchemy import text
    import hashlib
    
    # Use advisory lock on chat_id to serialize message inserts per chat
    # Convert UUID to a hash-based integer that fits in int64 range (max 9223372036854775807)
    # This prevents race conditions when multiple requests try to insert simultaneously
    # The lock is automatically released when the transaction ends
    # Use MD5 hash and take modulo to ensure it fits in int64
    chat_id_hash = hashlib.md5(chat_id.bytes).digest()[:8]
    chat_id_int = int.from_bytes(chat_id_hash, byteorder='big') % (2**63)
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:chat_id)"),
        {"chat_id": chat_id_int}
    )
    
    # Get next sequence number (now safe due to advisory lock)
    result = await session.execute(
        select(ChatMessage.sequence)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.sequence.desc())
        .limit(1)
    )
    last_seq = result.scalar()
    next_seq = (last_seq or 0) + 1

    session.add(ChatMessage(chat_id=chat_id, role="user", content=user_msg, sequence=next_seq))
    session.add(ChatMessage(chat_id=chat_id, role="assistant", content=ai_msg, sequence=next_seq + 1))
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
    async with session_factory() as verify_session:
        await _verify_chat_ownership(verify_session, chat_id, current_user.user_id)
        history = await _load_history(verify_session, chat_id)
        prior_evidence_state = await _load_prior_evidence_state(verify_session, chat_id)
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
    )

    start_time = time.perf_counter()
    try:
        final_state = await rag_app.ainvoke(initial_state)
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

    # ── Step 3: Log interaction + store messages (fresh session) ─────────
    async with session_factory() as log_session:
        await _log_interaction(
            db=log_session,
            chat_id=chat_id,
            user_input=body.message,
            agent_output=answer,
            routing_path=trajectory,
            latency=elapsed,
        )
        await _store_messages(log_session, chat_id, body.message, answer)

    evidence = final_state.get("evidence", [])
    claims = final_state.get("claims", [])
    verification_errors = [e.to_dict() for e in final_state.get("verification_errors", [])]
    citations = [
        CitationResponse(
            evidence_id=ev.evidence_id,
            text=ev.text[:500],
            source_type=ev.source_type.value,
            source_name=ev.source_name,
            source_url=ev.source_url,
            source_date=ev.source_date,
            authority_score=ev.authority_score,
            recency_score=ev.recency_score,
            metric_type=ev.metric_type.value if hasattr(ev, "metric_type") else "unknown",
            metric_value=ev.metric_value if hasattr(ev, "metric_value") else "",
            geographic_scope=ev.geographic_scope.value if hasattr(ev, "geographic_scope") else "unknown",
            geography=ev.geography if hasattr(ev, "geography") else "",
            year_period=ev.year_period if hasattr(ev, "year_period") else "",
            temporal_qualifier=ev.temporal_qualifier.value if hasattr(ev, "temporal_qualifier") else "unknown",
            source_quality=ev.source_quality.value if hasattr(ev, "source_quality") else "unknown",
        )
        for ev in evidence[:10]
    ]
    claim_responses = [
        ClaimResponse(
            claim_id=c.claim_id,
            text=c.text,
            status=c.status.value,
            claim_type=c.claim_type.value if hasattr(c, "claim_type") else "fact",
            evidence_ids=c.evidence_ids,
            contradicting_evidence_ids=c.contradicting_evidence_ids,
            reasoning=c.reasoning,
        )
        for c in claims
    ]

    return QueryResponse(
        answer=answer,
        chat_id=chat_id,
        latency_ms=latency_ms,
        provider_used=final_state.get("provider_used", "unknown"),
        final_status=final_state.get("final_status", "answered"),
        claims=claim_responses,
        citations=citations,
        conflicts=final_state.get("conflicts", []),
        verification_errors=verification_errors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming query — SSE for real-time node status
# ──────────────────────────────────────────────────────────────────────────────

NODE_LABELS = {
    "classify_query": "Understanding your question",
    "build_plan": "Planning search strategy",
    "retrieve_documents": "Retrieving documents",
    "search_web": "Searching the web",
    "assemble_evidence": "Assembling evidence",
    "extract_verify_claims": "Extracting claims",
    "generate_answer": "Generating answer",
    "verify_answer_claims": "Verifying facts",
    "repair_claims": "Repairing answer",
}


async def _stream_query(
    chat_id: _uuid.UUID, user_id: _uuid.UUID, message: str, session_factory, provider: str = "auto"
):
    """SSE generator that yields node-level status events during graph execution."""
    NODE_LABELS = {
        "classify_query": "Understanding your question",
        "build_plan": "Planning search strategy",
        "retrieve_documents": "Retrieving documents",
        "search_web": "Searching the web",
        "assemble_evidence": "Assembling evidence",
        "extract_verify_claims": "Extracting claims",
        "generate_answer": "Generating answer",
        "verify_answer_claims": "Verifying facts",
        "repair_claims": "Repairing answer",
    }

    start_time = time.perf_counter()
    # Load prior cross-turn evidence state (short-lived session).
    prior_state: EvidenceState | None = None
    async with session_factory() as _vs:
        await _verify_chat_ownership(_vs, chat_id, user_id)
        prior_state = await _load_prior_evidence_state(_vs, chat_id)
    turn = (prior_state.turn + 1) if prior_state else 1
    initial_state = _build_initial_state(query=message, user_id=user_id, chat_id=chat_id, provider=provider, prior_evidence_state=prior_state)
    answer = ""
    accumulated_state: dict = {**initial_state}
    trajectory_nodes = []

    try:
        # Use stream_mode="updates" which gives {node_name: output} per node
        async for event in rag_app.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name in ("__start__", "__end__"):
                    continue

                # Track trajectory
                if node_name not in trajectory_nodes:
                    trajectory_nodes.append(node_name)

                # Build rich status details based on node
                detail = ""
                if node_name == "build_plan" and isinstance(node_output, dict):
                    plan = node_output.get("plan")
                    if plan:
                        actions = ", ".join({s.action for s in plan.steps})
                        detail = f"Planned actions: {actions}"
                elif node_name == "search_web" and isinstance(node_output, dict):
                    search_count = len(node_output.get("search", []))
                    detail = f"Found {search_count} web results"
                elif node_name == "retrieve_documents" and isinstance(node_output, dict):
                    chunk_count = len(node_output.get("chunks", []))
                    detail = f"Found {chunk_count} document matches"
                elif node_name == "assemble_evidence" and isinstance(node_output, dict):
                    conflict_count = len(node_output.get("conflicts", []))
                    detail = f"Detected {conflict_count} evidence conflicts"
                elif node_name == "verify_answer_claims" and isinstance(node_output, dict):
                    claims = node_output.get("claims", [])
                    failed = [c for c in claims if c.status.value in ("unverified", "contradicted", "uncertain")]
                    detail = f"Claims: {len(claims)} total, {len(failed)} need repair"

                # Send status update
                label = NODE_LABELS.get(node_name, node_name)
                yield f"event: status\ndata: {json.dumps({'node': node_name, 'label': label, 'detail': detail, 'status': 'running'})}\n\n"

                # If answer generation completed, stream the answer tokens
                if node_name == "generate_answer" and isinstance(node_output, dict):
                    new_answer = node_output.get("answer", "")
                    if new_answer and new_answer != answer:
                        answer = new_answer
                        chunk_size = 100
                        for i in range(0, len(answer), chunk_size):
                            yield f"event: token\ndata: {json.dumps({'content': answer[i:i+chunk_size]})}\n\n"

                # Accumulate node outputs so we can build the provenance payload without re-running the graph.
                if isinstance(node_output, dict):
                    accumulated_state.update(node_output)

        final_state = accumulated_state
        answer = answer or final_state.get("answer", "") or ""

        elapsed = time.perf_counter() - start_time
        trajectory = " → ".join(trajectory_nodes) if trajectory_nodes else "unknown"

        # Persist the structured cross-turn evidence state for the next turn.
        merged_state = _finalize_evidence_state(final_state, prior_state, turn)
        trajectory = trajectory + "\n" + serialize_for_storage(merged_state)
        verification_errors = [e.to_dict() for e in final_state.get("verification_errors", [])]

        # Build structured provenance payload from accumulated node outputs.
        evidence = final_state.get("evidence", []) if isinstance(final_state, dict) else []
        claims = final_state.get("claims", []) if isinstance(final_state, dict) else []
        conflicts = final_state.get("conflicts", []) if isinstance(final_state, dict) else []
        citations = [
            {
                "evidence_id": ev.evidence_id,
                "text": ev.text[:500],
                "source_type": ev.source_type.value,
                "source_name": ev.source_name,
                "source_url": ev.source_url,
                "source_date": ev.source_date.isoformat() if ev.source_date else None,
                "authority_score": ev.authority_score,
                "recency_score": ev.recency_score,
            }
            for ev in evidence[:10]
        ]
        claim_responses = [
            {
                "claim_id": c.claim_id,
                "text": c.text,
                "status": c.status.value,
                "evidence_ids": c.evidence_ids,
                "contradicting_evidence_ids": c.contradicting_evidence_ids,
                "reasoning": c.reasoning,
            }
            for c in claims
        ]

        # Log interaction
        async with session_factory() as log_session:
            await _log_interaction(
                db=log_session, chat_id=chat_id, user_input=message,
                agent_output=answer, routing_path=trajectory, latency=elapsed,
            )
            await _store_messages(log_session, chat_id, message, answer)

        yield f"event: done\ndata: {json.dumps({
            'answer': answer,
            'final_status': final_state.get('final_status', 'answered') if isinstance(final_state, dict) else 'answered',
            'latency_ms': round(elapsed * 1000, 2),
            'trajectory': trajectory,
            'provider': provider,
            'citations': citations,
            'claims': claim_responses,
            'conflicts': conflicts,
            'verification_errors': verification_errors,
        }, default=str)}\n\n"

    except Exception as e:
        logger.exception("Streaming query failed for chat %s", chat_id)
        yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"


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
    # Verify chat ownership
    async with session_factory() as verify_session:
        await _verify_chat_ownership(verify_session, chat_id, current_user.user_id)

    return StreamingResponse(
        _stream_query(chat_id, current_user.user_id, body.message, session_factory, provider=body.provider),
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
) -> None:
    """Write one interaction row using the caller's session. Failure is logged, not raised."""
    try:
        interaction = Agent_interact(
            chat_id=chat_id,
            user_input=user_input,
            agent_output=agent_output,
            routing_path=routing_path,
            token_metric=estimate_tokens(user_input) + estimate_tokens(agent_output),
            latency=round(latency, 4),
        )
        db.add(interaction)
        await db.commit()
    except Exception:
        logger.exception("Failed to log interaction for chat %s", chat_id)


def _build_trajectory(state: dict) -> str:
    """
    Reconstruct which nodes were visited based on the final state.
    Also stores key evidence metadata for cross-turn carry-forward.
    """
    steps = ["classify_query", "build_plan"]

    if state.get("retrieval_count", 0) > 0:
        steps.append("retrieve_documents")
    if state.get("search_count", 0) > 0:
        steps.append("search_web")

    steps.extend(["assemble_evidence", "extract_verify_claims", "generate_answer", "verify_answer_claims"])

    repair_state = state.get("repair_state")
    if repair_state == "repair":
        steps.append("repair_claims→repair")
    elif repair_state == "max_attempts":
        steps.append("repair_claims→max_attempts")
    else:
        steps.append("repair_claims→satisfactory")

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

        # Use astream_events for node-level streaming
        trajectory_nodes = []
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
                    answer = node_output.get("answer", "")
                    if answer:
                        # Send answer in chunks for progressive display
                        chunk_size = 100
                        for i in range(0, len(answer), chunk_size):
                            await websocket.send_json({
                                "type": "token",
                                "content": answer[i:i + chunk_size],
                            })

        elapsed = time.perf_counter() - start_time
        latency_ms = round(elapsed * 1000, 2)

        # Extract answer from streamed output - don't re-run the graph
        # The graph was already fully executed in the astream loop above
        if not answer:
            # Fallback: get final state using ainvoke with stream_mode='values' to get final accumulated state
            async for state in rag_app.astream(initial_state, stream_mode="values"):
                answer = state.get("answer", "")

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
