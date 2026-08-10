import time
import uuid
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update

from app.core.database import get_db
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
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all chats for the current user, newest first."""
    result = await db.execute(
        select(Chats)
        .where(Chats.user_id == current_user.user_id)
        .order_by(Chats.created_at.desc())
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

@router.post("/chats/{chat_id}/query", response_model=QueryResponse)
async def query_agent(
    chat_id: uuid.UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to the self-correcting RAG agent.

    1. Verifies the chat belongs to the user.
    2. Runs the full LangGraph pipeline (retrieve → plan → search → generate → verify).
    3. Logs the interaction to the agents table.
    4. Returns the final answer.
    """
    # ── Verify chat ownership ────────────────────────────────────────────
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    chat = result.scalar_one_or_none()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # ── Run the RAG graph ────────────────────────────────────────────────
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

    # ── Build a rough trajectory from the state ──────────────────────────
    trajectory = _build_trajectory(final_state)

    # ── Log the interaction ──────────────────────────────────────────────
    interaction = Agent_interact(
        chat_id=chat_id,
        user_input=body.message,
        agent_output=answer,
        routing_path=trajectory,
        token_metric=_estimate_tokens(body.message) + _estimate_tokens(answer),
        latency=round(elapsed, 4),
    )
    db.add(interaction)
    await db.commit()

    return QueryResponse(
        answer=answer,
        chat_id=chat_id,
        latency_ms=latency_ms,
    )


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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the interaction history for a chat, oldest first."""
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
    )
    interactions = result.scalars().all()
    return InteractionListResponse(interactions=interactions)
