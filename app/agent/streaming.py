"""SSE streaming + provenance payload serialization for agent queries.

Split out of router.py. The SSE generator drives graph execution while
emitting node-status events; payloads feed the UI provenance panels.
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
from app.agent.chat_service import (  # noqa: F401
    _verify_chat_ownership,
    _load_history,
    _load_prior_evidence_state,
    _load_document_inventory,
    _finalize_evidence_state,
    _store_messages,
    _log_interaction,
)
from app.documents.service import estimate_tokens
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


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

def _build_initial_state(
    query: str,
    user_id: _uuid.UUID,
    chat_id: _uuid.UUID,
    provider: str = "auto",
    history: list | None = None,
    prior_evidence_state: EvidenceState | None = None,
    user_credentials: dict | None = None,
    document_inventory: list | None = None,
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
        document_inventory=document_inventory,
    )


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
    # Load history + prior cross-turn evidence state + document inventory
    # (short-lived session). History is ESSENTIAL: follow-ups like "talk about
    # its equations" are unresolvable without the prior turns — the SSE path
    # previously ran the graph with zero conversation context.
    prior_state: EvidenceState | None = None
    document_inventory: list[str] = []
    history: list = []
    async with session_factory() as _vs:
        await _verify_chat_ownership(_vs, chat_id, user_id)
        history = await _load_history(_vs, chat_id)
        prior_state = await _load_prior_evidence_state(_vs, chat_id)
        document_inventory = await _load_document_inventory(_vs, chat_id)
    turn = (prior_state.turn + 1) if prior_state else 1
    initial_state = _build_initial_state(
        query=message,
        user_id=user_id,
        chat_id=chat_id,
        provider=provider,
        history=history,
        prior_evidence_state=prior_state,
        user_credentials=user_credentials or {},
        document_inventory=document_inventory,
    )
    answer = ""
    accumulated_state: dict = {**initial_state}
    trajectory_nodes = []

    try:
        # astream_events(v2) intercepts chat-model token deltas at the callback
        # level — the generator streams its real tokens as they are produced,
        # WITHOUT changing any node logic or the repair flow. LangGraph tags
        # every event with metadata.langgraph_node, which we use to forward
        # only generate_answer tokens (planner/judge streams stay internal).
        HEARTBEAT_S = 5.0  # ping during silent stretches so clients never look stuck

        # Producer runs astream_events in its OWN task and pushes into a queue.
        # Never await/cancel the generator's frame from the consumer loop:
        # asyncio.wait_for(fut, timeout) on __anext__ cancels the generator's
        # frame mid-iteration, which kills the event stream the first time a
        # node runs silent longer than the heartbeat (routine on Cloud Run's
        # throttled CPU). The queue + independent task keep the 5s ping side-
        # effect free.
        _SENTINEL = object()
        events_q: asyncio.Queue = asyncio.Queue(maxsize=256)

        async def _producer():
            try:
                async for ev in rag_app.astream_events(initial_state, version="v2"):
                    await events_q.put(ev)
            finally:
                await events_q.put(_SENTINEL)

        producer_task: asyncio.Task = asyncio.ensure_future(_producer())
        answer_pass_has_tokens = False   # tokens emitted for the current pass?
        node_started: dict[str, float] = {}
        while True:
            try:
                ev = await asyncio.wait_for(events_q.get(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield f"event: ping\ndata: {json.dumps({'elapsed_ms': elapsed_ms})}\n\n"
                continue
            if ev is _SENTINEL:
                break
            etype = ev.get("event", "")
            name = ev.get("name", "")
            meta = ev.get("metadata") or {}
            data = ev.get("data") or {}
            node = meta.get("langgraph_node") or name
            total_ms = int((time.perf_counter() - start_time) * 1000)
    
            if settings.QUERY_TIMEOUT_SECONDS and settings.QUERY_TIMEOUT_SECONDS > 0 \
                    and time.perf_counter() - start_time > settings.QUERY_TIMEOUT_SECONDS:
                raise asyncio.TimeoutError()
    
            # ── Node/graph exceptions: surface for diagnosis; UI still gets a
            # well-formed done so the fallback path isn't mistaken for success. ──
            if etype == "on_chain_error":
                err = str(data.get("error") or data.get("exception") or ev.get("error") or "")
                node_err = meta.get("langgraph_node") or name
                logger.error("Graph %serror in node=%r name=%r: %s",
                             "substep " if node_err and node_err != name else "", node_err, name, err)
                continue
    
            # Graph-root terminal event. astream_events occasionally drops
            # intermediate node events under heavy asyncio scheduling (observed
            # on Cloud Run), but it ALWAYS closes with `on_chain_end` for the
            # root (name=="LangGraph") carrying the authoritative final state.
            # Without this merge the answer/evidence would be lost and the UI
            # show an empty fallback even though the graph succeeded.
            if etype == "on_chain_end" and name not in NODE_LABELS and isinstance(data.get("output"), dict):
                accumulated_state.update(data["output"])
                continue
    
            # ── Node lifecycle → status events (start + completion w/ timing) ──
            if etype == "on_chain_start" and name in NODE_LABELS:
                node_started[name] = time.perf_counter()
                yield f"event: status\ndata: {json.dumps({'node': name, 'label': NODE_LABELS[name], 'status': 'running', 'elapsed_ms': total_ms})}\n\n"
                continue
    
            if etype == "on_chain_end" and name in NODE_LABELS:
                out = data.get("output")
                started_at = node_started.pop(name, None)
                node_ms = int((time.perf_counter() - started_at) * 1000) if started_at else None
                if isinstance(out, dict):
                    accumulated_state.update(out)
    
                detail = ""
                if name == "classify_and_plan":
                    u = out.get("understanding") if isinstance(out, dict) else None
                    mode = getattr(u, "mode", None)
                    detail = f"Mode: {getattr(mode, 'value', mode)}"
                elif name == "gather_evidence":
                    ev_count = len((out.get("evidence") or []) if isinstance(out, dict) else [])
                    detail = f"Found {ev_count} evidence items"
                elif name == "verify_answer" and isinstance(out, dict):
                    claims = out.get("claims", []) or []
                    failed = [
                        c for c in claims
                        if (c.status.value if hasattr(c.status, "value") else str(c.status))
                        in ("unverified", "contradicted", "uncertain")
                    ]
                    detail = f"Claims: {len(claims)} total, {len(failed)} need attention"
    
                payload = {"node": name, "label": NODE_LABELS.get(name, name),
                           "status": "done", "elapsed_ms": total_ms}
                if node_ms is not None:
                    payload["node_ms"] = node_ms
                if detail:
                    payload["detail"] = detail
                yield f"event: status\ndata: {json.dumps(payload)}\n\n"
    
                if name in ("gather_evidence", "generate_answer") and isinstance(out, dict):
                    evid = accumulated_state.get("evidence") or []
                    if evid:
                        yield ("event: provenance\ndata: "
                               + json.dumps({"citations": _citations_payload(evid), "conflicts": []})
                               + "\n\n")
                continue
    
            # ── Real generator tokens ──
            if etype == "on_chat_model_start" and node == "generate_answer":
                if answer_pass_has_tokens:
                    # A NEW generation pass began after tokens were already sent
                    # (the bounded repair loop regenerating the answer): tell the
                    # client to discard what it has.
                    yield "event: answer_reset\ndata: {}\n\n"
                    answer = ""
                continue
    
            if etype == "on_chat_model_stream" and node == "generate_answer":
                chunk_obj = data.get("chunk")
                piece = getattr(chunk_obj, "content", "")
                if not piece or not isinstance(piece, str):
                    continue
                answer += piece
                answer_pass_has_tokens = True
                yield f"event: token\ndata: {json.dumps({'content': piece})}\n\n"
    
        if not producer_task.done():
            producer_task.cancel()
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
        if not producer_task.done():
            producer_task.cancel()
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

def _citations_payload(evidence: list, limit: int = 15) -> list[dict]:
    """Serialize evidence for SSE/API provenance panels."""
    out: list[dict] = []
    for ev in (evidence or [])[:limit]:
        out.append(
            {
                "evidence_id": ev.evidence_id,
                "cite_key": (ev.metadata or {}).get("cite_key"),
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
