"""Chat persistence + cross-turn memory helpers for the agent API.

Split out of router.py: DB access, history loading, evidence-state carry-over,
message storage, and Chroma cleanup. Routes live in router.py.
"""
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
