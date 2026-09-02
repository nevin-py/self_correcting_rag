"""LangGraph wiring for the lean self-correcting RAG agent.

    classify_and_plan ─┬─ conversational_response ─ END
                       ├─ ask_clarification ─────── END
                       └─ gather_evidence ─► generate_answer ─► verify_answer
                                                   ▲                │
                                                   └── repair ◄─────┘  (≤ MAX_REPAIR_PASSES)
                                                                    │ (else)
                                                                   END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    ask_clarification,
    classify_and_plan,
    conversational_response,
    gather_evidence,
    generate_answer,
    route_after_classify,
    route_after_verify,
    verify_answer,
)
from app.agent.state import RAGState
from app.core.config import settings

logger = logging.getLogger(__name__)


def _new_state(
    query: str,
    user_id: Any = None,
    chat_id: Any = None,
    provider: str = "auto",
    messages: list | None = None,
    document_inventory: list | None = None,
) -> RAGState:
    """Construct a fresh RAGState with guard counters initialized."""
    return RAGState(
        user_id=user_id,
        chat_id=chat_id,
        query=query,
        query_original=query,
        provider=provider or "auto",
        messages=messages or [],
        document_inventory=document_inventory or [],
        request_context={},
        user_credentials={},
        understanding=None,
        evidence=[],
        cite_map={},
        assembled_context="",
        claims=[],
        verification_errors=[],
        repair_queries=[],
        repair_count=0,
        prior_evidence_state=None,
        graph_steps=0,
        search_count=0,
        retrieval_count=0,
        answer="",
        final_status="answered",
        provider_used="",
        error=None,
    )


# Build graph
builder = StateGraph(RAGState)

builder.add_node("classify_and_plan", classify_and_plan)
builder.add_node("conversational_response", conversational_response)
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("gather_evidence", gather_evidence)
builder.add_node("generate_answer", generate_answer)
builder.add_node("verify_answer", verify_answer)

builder.set_entry_point("classify_and_plan")
builder.add_conditional_edges("classify_and_plan", route_after_classify, {
    "conversational_response": "conversational_response",
    "ask_clarification": "ask_clarification",
    "gather_evidence": "gather_evidence",
})
builder.add_edge("conversational_response", END)
builder.add_edge("ask_clarification", END)
builder.add_edge("gather_evidence", "generate_answer")
builder.add_edge("generate_answer", "verify_answer")
builder.add_conditional_edges("verify_answer", route_after_verify, {
    "gather_evidence": "gather_evidence",
    END: END,
})


# Compile graph lazily to avoid import-time errors
_graph: Any = None
_rag_app: Any = None


def get_graph() -> Any:
    """Get or compile the graph (lazy compilation)."""
    global _graph, _rag_app
    if _rag_app is None:
        _graph = builder.compile()
        _rag_app = _graph
    return _rag_app


rag_app = get_graph()


def create_initial_state(
    query: str,
    user_id=None,
    chat_id=None,
    provider: str = "auto",
    messages: list | None = None,
    user_credentials: dict | None = None,
    prior_evidence_state=None,
    request_context: dict | None = None,
    document_inventory: list | None = None,
) -> RAGState:
    """Single canonical entry point used by router.py."""
    state = _new_state(query, user_id, chat_id, provider, messages, document_inventory)
    state["user_credentials"] = user_credentials or {}
    state["prior_evidence_state"] = prior_evidence_state
    state["request_context"] = request_context or {}
    return state
