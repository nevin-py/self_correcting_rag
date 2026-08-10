import re
import time
import uuid
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db, AsyncLocalSession
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats, Agent_interact
from app.agent.graph import rag_app
from app.agent.state import RAGState
from app.agent.schemas import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    QueryRequest,
    QueryResponse,
    InteractionResponse,
    InteractionListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# Heavy endpoint — 10 queries/min per user (each triggers multiple LLM calls)
_query_limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_initial_state(query: str, user_id: uuid.UUID) -> dict:
    """Build the initial LangGraph state dict for a new query."""
    return {
        "user_id": user_id,
        "query": query,
        "messages": [],
        "chunks": [],
        "search": [],
        "planner_state": "not_enough",
        "retrieval_queries": [],
        "wiki_queries": [],
        "tavily_queries": [],
        "executed_retrieval_queries": [],
        "executed_wiki_queries": [],
        "executed_tavily_queries": [],
        "answer": "",
        "need_repair": "repair",
        "hallucination_reason": [],
        "max_tries_planner": 0,
        "max_tries_hallucinator": 0,
    }


def _estimate_tokens(text: str) -> int:
    """Estimate token count using word-boundary splitting.

    This is a rough heuristic — real tokenizers (tiktoken, BPE) vary by
    language and vocabulary.  For English prose this lands within ~20 % of
    the true count, which is good enough for cost monitoring.
    """
    # Split on word boundaries + count punctuation as separate tokens
    words = re.findall(r"\w+|[^\s\w]", text)
    return max(1, len(words))


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
    chat_id: uuid.UUID,
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
    chat_id: uuid.UUID,
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


# ──────────────────────────────────────────────────────────────────────────────
# Query (the core RAG endpoint)
# ──────────────────────────────────────────────────────────────────────────────

@_query_limiter.limit("10/minute")
@router.post("/chats/{chat_id}/query", response_model=QueryResponse)
async def query_agent(request: Request,
    chat_id: uuid.UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the self-correcting RAG agent.

    1. Verifies the chat belongs to the user (quick DB hit, then release).
    2. Runs the full LangGraph pipeline — no DB session held open.
    3. Logs the interaction with a fresh session (best-effort).
    4. Returns the final answer.
    """
    # ── Step 1: Verify chat ownership (short-lived) ──────────────────────
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Ownership confirmed — the db session from Depends(get_db) will close
    # when this endpoint returns. The RAG graph below does not use it.

    # ── Step 2: Run the RAG graph (no DB session held) ───────────────────
    initial_state = _build_initial_state(
        query=body.message,
        user_id=current_user.user_id,
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

    answer = final_state.get("answer", "")
    if not answer:
        answer = "I was unable to generate an answer. Please try rephrasing your question."

    trajectory = _build_trajectory(final_state)

    # ── Step 3: Log interaction (fresh session, best-effort) ─────────────
    await _log_interaction(
        chat_id=chat_id,
        user_input=body.message,
        agent_output=answer,
        routing_path=trajectory,
        latency=elapsed,
    )

    return QueryResponse(
        answer=answer,
        chat_id=chat_id,
        latency_ms=latency_ms,
    )


async def _log_interaction(
    chat_id: uuid.UUID,
    user_input: str,
    agent_output: str,
    routing_path: str,
    latency: float,
) -> None:
    """Write one interaction row in a short-lived session. Failure is logged, not raised."""
    try:
        async with AsyncLocalSession() as session:
            interaction = Agent_interact(
                chat_id=chat_id,
                user_input=user_input,
                agent_output=agent_output,
                routing_path=routing_path,
                token_metric=_estimate_tokens(user_input) + _estimate_tokens(agent_output),
                latency=round(latency, 4),
            )
            session.add(interaction)
            await session.commit()
    except Exception:
        logger.exception("Failed to log interaction for chat %s", chat_id)


def _build_trajectory(state: dict) -> str:
    """
    Reconstruct which nodes were visited based on the final state.
    This is a lightweight heuristic — not a full event trace.
    """
    steps = ["initial_retrieval"]

    # If search results exist, the planner triggered a search loop
    if state.get("search"):
        steps.append("Planner→search")

    steps.append("Answer generation")

    # If hallucination reasons exist, the checker ran and found issues
    if state.get("hallucination_reason"):
        steps.append("Hallucination checker→repair")
    else:
        steps.append("Hallucination checker→factual")

    return " → ".join(steps)


# ──────────────────────────────────────────────────────────────────────────────
# Interaction history
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/chats/{chat_id}/history", response_model=InteractionListResponse)
async def get_chat_history(
    chat_id: uuid.UUID,
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
