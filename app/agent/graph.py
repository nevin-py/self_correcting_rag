"""
Business-ready self-correcting RAG graph.

Pipeline
--------
1. classify_and_plan   → intent + typed plan (single LLM call)
2. retrieve_documents  → knowledge-base evidence (conditional)
3. search_web          → public-web evidence (conditional)
4. assemble_evidence   → scoring, conflict detection, context assembly
5. extract_verify_claims → pre-verification (currently a no-op)
6. generate_answer     → cited answer + hard citation flags
7. verify_answer_claims → cascade or LLM claim-level check
8. repair_claims       → surgical patch (or legacy re-search) / terminate

When USE_VERIFY_CASCADE is on and a coverage-gap search runs:
  assemble → repair_claims (patch only, skip generate) → END
"""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import StateGraph, END

from app.agent.nodes import (
    classify_and_plan,
    retrieve_documents,
    search_web,
    assemble_evidence,
    extract_verify_claims,
    generate_answer,
    verify_answer_claims,
    repair_claims,
    should_retrieve_documents,
    should_search_web,
    should_post_assemble,
    hallucination_router,
)
from app.agent.state import RAGState, RepairDecision
from app.core.config import settings

logger = logging.getLogger(__name__)


def _new_state(
    query: str,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    provider: str = "auto",
    messages: list | None = None,
) -> RAGState:
    """Construct a fresh RAGState with guard counters initialized."""
    return RAGState(
        user_id=user_id,
        chat_id=chat_id,
        query=query,
        provider=provider,
        messages=messages or [],
        graph_steps=0,
        search_count=0,
        retrieval_count=0,
        regeneration_count=0,
        repair_pass_count=0,
        max_graph_steps=settings.MAX_GRAPH_STEPS,
        max_searches=settings.MAX_SEARCHES,
        max_retrievals=settings.MAX_RETRIEVALS,
        max_regenerations=settings.MAX_REGENERATIONS,
        evidence=[],
        claims=[],
        conflicts=[],
        citation_usage=[],
        cite_map={},
        coverage_gaps=[],
        repair_mode="",
        chunks=[],
        search=[],
        retrieval_queries=[],
        wiki_queries=[],
        tavily_queries=[],
        searxng_queries=[],
    )


# Build graph
builder = StateGraph(RAGState)

builder.add_node("classify_and_plan", classify_and_plan)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("search_web", search_web)
builder.add_node("assemble_evidence", assemble_evidence)
builder.add_node("extract_verify_claims", extract_verify_claims)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer_claims", verify_answer_claims)
builder.add_node("repair_claims", repair_claims)

builder.set_entry_point("classify_and_plan")

builder.add_conditional_edges(
    "classify_and_plan",
    should_retrieve_documents,
    {
        "retrieve_documents": "retrieve_documents",
        "assemble": "assemble_evidence",
    },
)

builder.add_conditional_edges(
    "retrieve_documents",
    should_search_web,
    {
        "search_web": "search_web",
        "assemble": "assemble_evidence",
    },
)

builder.add_edge("search_web", "assemble_evidence")
builder.add_conditional_edges(
    "assemble_evidence",
    should_post_assemble,
    {
        "extract_verify_claims": "extract_verify_claims",
        "repair_claims": "repair_claims",
    },
)
builder.add_edge("extract_verify_claims", "generate_answer")
builder.add_edge("generate_answer", "verify_answer_claims")

builder.add_conditional_edges(
    "verify_answer_claims",
    hallucination_router,
    {
        "satisfactory": END,
        "max_attempts": "repair_claims",
        "repair": "repair_claims",
    },
)


def _repair_next(state: RAGState) -> str:
    """Route repair: surgical coverage-gap search only; else end."""
    repair_state = state.get("repair_state")
    if repair_state in (
        RepairDecision.SATISFACTORY.value,
        RepairDecision.MAX_ATTEMPTS.value,
        "satisfactory",
        "max_attempts",
    ):
        return "end"

    plan = state.get("plan")
    retrieval_count = state.get("retrieval_count", 0)
    search_count = state.get("search_count", 0)
    max_retrievals = state.get("max_retrievals", settings.MAX_RETRIEVALS)
    max_searches = state.get("max_searches", settings.MAX_SEARCHES)

    retrieval_available = retrieval_count < max_retrievals
    search_available = search_count < max_searches

    # Cascade path: only follow search_web for coverage gaps (never blind retrieve)
    if settings.USE_VERIFY_CASCADE and state.get("repair_mode") == "surgical":
        if plan and plan.steps and search_available:
            for step in plan.steps:
                if step.action == "search_web":
                    return "search_web"
        return "end"

    if plan and plan.steps:
        for step in plan.steps:
            if step.action == "retrieve_documents" and retrieval_available:
                return "retrieve_documents"
            if step.action == "search_web" and search_available:
                return "search_web"

    logger.warning(
        "Repair requested but no remaining plan steps (retrievals %d/%d, searches %d/%d)",
        retrieval_count,
        max_retrievals,
        search_count,
        max_searches,
    )
    return "end"


builder.add_conditional_edges(
    "repair_claims",
    _repair_next,
    {
        "retrieve_documents": "retrieve_documents",
        "search_web": "search_web",
        "end": END,
    },
)

graph = builder.compile()
rag_app = graph.with_config({"recursion_limit": settings.MAX_GRAPH_STEPS})


def create_initial_state(
    query: str,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    provider: str = "auto",
    messages: list | None = None,
) -> RAGState:
    """Public helper used by router.py to build the initial state."""
    return _new_state(query, user_id, chat_id, provider, messages)
