"""
Business-ready self-correcting RAG graph.

Pipeline
--------
1. classify_query      → structured intent & source needs
2. build_plan          → typed plan with steps
3. retrieve_documents  → knowledge-base evidence (conditional)
4. search_web          → public-web evidence (conditional)
5. assemble_evidence   → scoring, conflict detection, context assembly
6. extract_verify_claims → pre-verification of expected claims
7. generate_answer     → cited answer generation
8. verify_answer_claims → claim-level hallucination check
9. repair_claims       → targeted repair or termination

The graph loops between (4-5-6-7-8-9) until claims are satisfied or guard
limits are reached.
"""

from __future__ import annotations

import uuid

from langgraph.graph import StateGraph, END

from app.agent.nodes import (
    classify_query,
    build_plan,
    retrieve_documents,
    search_web,
    assemble_evidence,
    extract_verify_claims,
    generate_answer,
    verify_answer_claims,
    repair_claims,
    should_retrieve_documents,
    should_search_web,
    hallucination_router,
)
from app.agent.state import RAGState
from app.core.config import settings


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
        max_graph_steps=settings.MAX_GRAPH_STEPS,
        max_searches=settings.MAX_SEARCHES,
        max_retrievals=settings.MAX_RETRIEVALS,
        max_regenerations=settings.MAX_REGENERATIONS,
        evidence=[],
        claims=[],
        conflicts=[],
        citation_usage=[],
        chunks=[],
        search=[],
        retrieval_queries=[],
        wiki_queries=[],
        tavily_queries=[],
    )


# Build graph
builder = StateGraph(RAGState)

builder.add_node("classify_query", classify_query)
builder.add_node("build_plan", build_plan)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("search_web", search_web)
builder.add_node("assemble_evidence", assemble_evidence)
builder.add_node("extract_verify_claims", extract_verify_claims)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer_claims", verify_answer_claims)
builder.add_node("repair_claims", repair_claims)

builder.set_entry_point("classify_query")

# classify → plan
builder.add_edge("classify_query", "build_plan")

# plan → conditional retrieval / web / assembly
builder.add_conditional_edges(
    "build_plan",
    should_retrieve_documents,
    {
        "retrieve_documents": "retrieve_documents",
        "assemble": "assemble_evidence",
    },
)

# retrieval → conditional web / assembly
builder.add_conditional_edges(
    "retrieve_documents",
    should_search_web,
    {
        "search_web": "search_web",
        "assemble": "assemble_evidence",
    },
)

# web → assembly
builder.add_edge("search_web", "assemble_evidence")

# assembly → extract/verify claims
builder.add_edge("assemble_evidence", "extract_verify_claims")

# pre-verified claims → generate answer
builder.add_edge("extract_verify_claims", "generate_answer")

# answer → claim-level verification
builder.add_edge("generate_answer", "verify_answer_claims")

# verification → repair, satisfactory, or max attempts
builder.add_conditional_edges(
    "verify_answer_claims",
    hallucination_router,
    {
        "satisfactory": END,
        "max_attempts": END,
        "repair": "repair_claims",
    },
)

def _repair_next(state: RAGState) -> str:
    """Route repair to the first action in the new repair plan."""
    plan = state.get("plan")
    if plan and plan.steps:
        for step in plan.steps:
            if step.action == "retrieve_documents":
                return "retrieve_documents"
    return "search_web"


# repair → either re-retrieve or re-search based on plan, then back to assembly
builder.add_conditional_edges(
    "repair_claims",
    _repair_next,
    {
        "retrieve_documents": "retrieve_documents",
        "search_web": "search_web",
    },
)

graph = builder.compile()
rag_app = graph


def create_initial_state(
    query: str,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    provider: str = "auto",
    messages: list | None = None,
) -> RAGState:
    """Public helper used by router.py to build the initial state."""
    return _new_state(query, user_id, chat_id, provider, messages)
