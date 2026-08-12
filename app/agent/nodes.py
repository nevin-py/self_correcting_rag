"""
Agent nodes for the business-ready self-correcting RAG pipeline.

Architecture (each node has a single, observable responsibility):

    classify_and_plan    -> intent + retrieval plan (single LLM call)
    retrieve_documents   -> vector + BM25 retrieval over the knowledge base
    search_web           -> web search (Tavily / SearXNG / Wikipedia)
    assemble_evidence    -> SOURCE RANKING + conflict classification + context assembly
    extract_verify_claims -> claim extraction (disabled for cost; post-gen verify covers it)
    generate_answer      -> cited answer generation + hard citation flagging
    verify_answer_claims -> hard citation check + claim-level verification
    repair_claims        -> targeted repair or termination

Evidence normalization, source authority, ranking, conflict classification,
cross-turn state and verification now live in dedicated modules
(normalization / source_authority / ranking / conflicts / evidence_state /
verification) so the heuristics are small, pure, and unit-testable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import concurrent.futures
from datetime import datetime, timezone
from typing import Any, get_args, get_origin
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from app.agent.conflicts import detect_conflicts, is_genuine_contradiction
from app.agent.citation_validator import (
    extract_citation_tokens,
    flag_uncited_in_answer,
    resolve_citation_token,
    validate_answer_citations,
)
from app.agent.evidence_state import to_context_block
from app.agent.verify_cascade import (
    dedupe_search_queries,
    grade_coverage,
    normalize_query,
    patch_flagged_sentences,
    run_verify_cascade,
)
from app.agent.debug_session import agent_debug_log
from app.agent.normalization import (
    compose_search_query,
    detect_metric_type,
    detect_place_mentions,
    detect_price_basis,
    detect_scope_cues,
    detect_temporal_qualifier,
    extract_year_period,
    expand_queries,
    decompose_query_text,
    get_retrieval_queries_for_subqueries,
    SubQuery,
)
from app.agent.ranking import (
    combined_score,
    evidence_fits_classification,
    filter_evidence_by_classification,
    rank_evidence,
)
from app.agent.reranker import rerank
from app.agent.search_tool import search_structured
from app.agent.source_authority import (
    authority_score,
    classify_source_quality,
    official_search_variants,
)
from app.agent.state import (
    Claim,
    ClaimStatus,
    ClaimType,
    CitationUsage,
    Evidence,
    GeographicScope,
    MetricType,
    PlannerDecision,
    PlannerOutput,
    PlanStep,
    PriceBasis,
    QueryClassification,
    QueryNeed,
    RepairDecision,
    SourceQuality,
    SourceType,
    TemporalQualifier,
    utc_now,
)
from app.agent.verification import audit_claims
from app.documents.clients import resolve_llms
from app.documents.clients import get_chroma_client
from app.documents.service import retrieve_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _normalize_messages_for_gemini(messages: list) -> list:
    """Gemini rejects system-only payloads ('contents are required').

    Convert a lone SystemMessage into a HumanMessage, and ensure there is at
    least one user turn when a SystemMessage is followed by nothing else.
    """
    if not messages:
        return [HumanMessage(content="Continue.")]

    out: list[BaseMessage] = []
    has_human = False
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Keep system if other providers; for Gemini path we fold into human
            out.append(msg)
        elif isinstance(msg, HumanMessage):
            has_human = True
            out.append(msg)
        else:
            out.append(msg)

    if not has_human:
        # Fold system instructions into a user turn Gemini will accept
        system_parts = [m.content for m in out if isinstance(m, SystemMessage)]
        other = [m for m in out if not isinstance(m, SystemMessage)]
        merged = "\n\n".join(str(p) for p in system_parts if p)
        return other + [HumanMessage(content=merged or "Follow the instructions.")]
    return out


def _is_google_llm(llm: Any) -> bool:
    return llm is not None and llm.__class__.__name__ == "ChatGoogleGenerativeAI"


def _prepare_messages(llm: Any, messages: list) -> list:
    if _is_google_llm(llm):
        return _normalize_messages_for_gemini(messages)
    return messages


def _strip_json_markers(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict:
    """Best-effort JSON object extraction from model prose / markdown fences."""
    if not text:
        return {}
    cleaned = _strip_json_markers(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(cleaned[start : i + 1])
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _safe_json_loads(text: str) -> dict:
    return _extract_json_object(text)


def _is_openai_compatible(llm: Any) -> bool:
    name = llm.__class__.__name__ if llm is not None else ""
    return name in {"ChatOpenAI", "AzureChatOpenAI"}


def _base_model_type(annotation: Any) -> type[BaseModel] | None:
    """Resolve a field annotation to a nested BaseModel type, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _coerce_null_strings(data: Any, model: type[BaseModel]) -> Any:
    """Coerce JSON nulls to schema defaults before Pydantic validation."""
    if not isinstance(data, dict):
        return data
    fixed = dict(data)
    for key, finfo in model.model_fields.items():
        if key not in fixed:
            continue
        val = fixed[key]
        if val is None and finfo.annotation is str:
            fixed[key] = finfo.default if finfo.default is not PydanticUndefined else ""
        elif isinstance(val, dict):
            nested = _base_model_type(finfo.annotation)
            if nested is not None:
                fixed[key] = _coerce_null_strings(val, nested)
    return fixed


def _validate_structured(data: dict, output_schema: type[BaseModel]) -> BaseModel:
    try:
        return output_schema.model_validate(data)
    except ValidationError:
        coerced = _coerce_null_strings(data, output_schema)
        return output_schema.model_validate(coerced)


def _response_text(response: Any) -> str:
    """Extract visible text from an LLM response (handles list / reasoning-only)."""
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(getattr(block, "text", None) or block))
        content = "".join(parts)
    text = str(content or "").strip()
    if text:
        return text

    extra = getattr(response, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "text", "output"):
        val = extra.get(key)
        if isinstance(val, dict):
            val = val.get("text") or val.get("content") or ""
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Some OpenRouter wrappers stash the message on response.response_metadata
    meta = getattr(response, "response_metadata", None) or {}
    for key in ("reasoning", "content"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _structured_invoke(llm: Any, messages: list, output_schema: Any):
    """Invoke with structured output; one raw JSON retry only if needed.

    Avoids burning a second full LLM round-trip when the first response is empty.
    """
    prepared = _prepare_messages(llm, messages)

    # 1) Native structured output (json_schema when available)
    try:
        try:
            bound = llm.with_structured_output(output_schema, method="json_schema")
        except TypeError:
            bound = llm.with_structured_output(output_schema)
        result = bound.invoke(prepared)
        if result is not None:
            return result
    except Exception as structured_exc:
        logger.debug("Structured output failed (%s); trying raw JSON parse", structured_exc)

    # 2) Single raw completion + parse (no further retries here)
    parse_hint = HumanMessage(
        content=(
            "Respond with ONLY a single valid JSON object matching the required schema. "
            "No markdown, no commentary, no code fences. Never return an empty response."
        )
    )
    response = llm.invoke([*prepared, parse_hint])
    text = _response_text(response)
    if not text:
        # #region agent log
        agent_debug_log(
            "D",
            "nodes.py:_structured_invoke",
            "empty_structured_response",
            {
                "schema": getattr(output_schema, "__name__", str(output_schema)),
                "content_type": type(getattr(response, "content", None)).__name__,
                "raw_preview": str(getattr(response, "content", ""))[:120],
            },
        )
        # #endregion
        raise ValueError("Could not parse JSON from model output: (empty response)")
    data = _extract_json_object(text)
    if not data:
        raise ValueError(f"Could not parse JSON from model output: {text[:200]}")
    return _validate_structured(data, output_schema)


def _llm_with_fallback(primary: Any, fallbacks: Any, messages: list, output_schema: Any):
    """Call primary LLM; on failure walk fallbacks with structured output."""
    chain = [primary]
    if fallbacks is None:
        pass
    elif isinstance(fallbacks, (list, tuple)):
        chain.extend(fallbacks)
    else:
        chain.append(fallbacks)

    last_exc: Exception | None = None
    for idx, llm in enumerate(chain):
        if llm is None:
            continue
        try:
            return _structured_invoke(llm, messages, output_schema)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM call failed (%s)%s",
                exc,
                ", trying next" if idx < len(chain) - 1 else "",
            )
    if last_exc:
        raise last_exc
    raise RuntimeError("No LLM clients available")


def _invoke_chat(primary: Any, fallbacks: tuple[Any, ...] | list[Any], messages: list) -> tuple[str, str]:
    """Invoke chat models with multi-fallback. Returns (answer, used_label_suffix)."""
    chain = [primary, *list(fallbacks or ())]
    last_exc: Exception | None = None
    for idx, llm in enumerate(chain):
        if llm is None:
            continue
        try:
            response = llm.invoke(_prepare_messages(llm, messages))
            text = _response_text(response)
            if not text:
                raise ValueError("Generator returned empty response")
            label = "primary" if idx == 0 else f"fallback-{idx}"
            return text, label
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Generator failed (%s)%s",
                exc,
                ", trying next" if idx < len(chain) - 1 else "",
            )
    if last_exc:
        raise last_exc
    raise RuntimeError("No generator clients available")


# ── Backward-compatible extraction wrappers (delegate to normalization) ───────

def _extract_metric_type(text: str) -> MetricType:
    return detect_metric_type(text)


def _extract_temporal_qualifier(text: str) -> TemporalQualifier:
    return detect_temporal_qualifier(text)


def _extract_year_period(text: str) -> str:
    return extract_year_period(text)


def _extract_metric_value(text: str) -> str:
    """Extract the primary numeric value from text (for citation display)."""
    m = re.search(r"[₹$]?[\d,.]+\s*(?:%|per\s*cent|lakh|crore|billion|million|trillion)", text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    m = re.search(r"\b[\d,]+(?:\.\d+)?\b", text)
    if m:
        return m.group(0)
    return ""


def _classify_source_quality(source_name: str, source_url: str | None, source_type: SourceType) -> SourceQuality:
    return classify_source_quality(source_name, source_url, source_type)


def _score_domain_authority(url: str | None, source_type: SourceType) -> float:
    return authority_score(url, source_type)


def _extract_geographic_scope(geography: str, text: str) -> tuple[GeographicScope, str]:
    """Generic geography extraction (NO hardcoded place lists).

    - If a query-level `geography` (LLM-classified) is provided, its scope is
      derived generically and returned (this is the query's intended scope).
    - Otherwise the evidence's own scope + place mention is discovered from text.
    Place discovery is a general Title-Case recognizer, so any place (e.g.
    Maharashtra, Karnataka, Tamil Nadu, USA) is captured without enumeration.
    """
    if geography:
        scope = detect_scope_cues(geography)
        if scope == GeographicScope.UNKNOWN:
            # Infer a generic scope for a bare place name if possible.
            if any(w in geography.lower() for w in ("national", "country", "nation")):
                scope = GeographicScope.NATIONAL
            elif any(w in geography.lower() for w in ("state", "province")):
                scope = GeographicScope.STATE
        return scope, geography

    scope = detect_scope_cues(text)
    places = detect_place_mentions(text)
    place = places[0] if places else ""
    return scope, place


def _enrich_evidence_metadata(ev: Evidence, classification: QueryClassification | None = None) -> Evidence:
    """Populate metric, price-basis, geographic, temporal and source fields.

    Rules:
    - A specific metric acronym (GDP/GSDP/GVA/...) wins over a generic modifier
      (growth rate / inflation). A generic modifier is overridden by a specific
      classification hint.
    - Evidence geography is discovered from its OWN text; it only inherits the
      query geography when it names no place of its own (so a source about a
      *different* place is still caught as a geographic mismatch).
    - Price basis (current vs constant) is detected explicitly.
    """
    # ── Metric ──
    detected = detect_metric_type(ev.text)
    if detected != MetricType.UNKNOWN and detected not in (MetricType.GROWTH_RATE, MetricType.INFLATION):
        ev.metric_type = detected
    elif classification and classification.metric_hint != MetricType.UNKNOWN:
        ev.metric_type = classification.metric_hint
    elif detected != MetricType.UNKNOWN:
        ev.metric_type = detected

    # ── Price basis ──
    if ev.price_basis == PriceBasis.UNKNOWN:
        ev.price_basis = detect_price_basis(ev.text)
    if ev.price_basis == PriceBasis.UNKNOWN and classification and getattr(classification, "price_basis", PriceBasis.UNKNOWN) != PriceBasis.UNKNOWN:
        ev.price_basis = classification.price_basis  # type: ignore[attr-defined]

    # ── Temporal ──
    if ev.temporal_qualifier == TemporalQualifier.UNKNOWN:
        ev.temporal_qualifier = detect_temporal_qualifier(ev.text)
    if ev.temporal_qualifier == TemporalQualifier.UNKNOWN and classification and classification.temporal_qualifier != TemporalQualifier.UNKNOWN:
        ev.temporal_qualifier = classification.temporal_qualifier

    # ── Year / period ──
    if not ev.year_period:
        ev.year_period = extract_year_period(ev.text)

    # ── Metric value ──
    if not ev.metric_value:
        ev.metric_value = _extract_metric_value(ev.text)

    # ── Geography ──
    text_place = detect_place_mentions(ev.text)
    if text_place:
        ev.geography = text_place[0]
        scope = detect_scope_cues(ev.text)
        if scope != GeographicScope.UNKNOWN:
            ev.geographic_scope = scope
    elif classification and classification.geography:
        ev.geography = classification.geography
        if classification.geographic_scope != GeographicScope.UNKNOWN:
            ev.geographic_scope = classification.geographic_scope

    # ── Source quality / authority ──
    if ev.source_quality == SourceQuality.UNKNOWN:
        ev.source_quality = classify_source_quality(ev.source_name, ev.source_url, ev.source_type)
    if ev.authority_score == 0.0:
        ev.authority_score = authority_score(ev.source_url, ev.source_type)

    return ev


# ── Authority / recency scoring ──────────────────────────────────────────────


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    m = re.search(r"(?i)(january|february|march|april|may|june|july|august|september|october|november|december)[\s,]+(20\d{2})", text)
    if m:
        return datetime(int(m.group(2)), months[m.group(1).lower()], 1, tzinfo=timezone.utc)
    return None


def _recency_score(evidence_date: datetime | None, temporal_focus: datetime | str | None = None) -> float:
    if evidence_date is None:
        return 0.5
    now = utc_now()
    reference = temporal_focus
    if isinstance(reference, str):
        reference = _parse_date(reference)
    reference = reference or now
    age_days = max(0, (reference - evidence_date).days)
    return 0.9 if age_days < 365 else max(0.3, 0.9 - age_days / 3650.0)


def _combined_score(ev: Evidence) -> float:
    """Backward-compatible combined score (no classification context)."""
    return combined_score(ev, None)


# ── Deterministic contradiction detection (delegates to conflicts) ────────────

def _normalize_claim(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_contradiction(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    """Backward-compatible contradiction check (genuine contradiction or disagreement)."""
    return is_genuine_contradiction(claim_text, evidence_text)


def _detect_evidence_conflicts(evidence: list[Evidence]) -> list[dict]:
    """Pairwise conflict detection (returns structured, classified records)."""
    return detect_conflicts(evidence)


# ── Nodes: classification & planning (single LLM call) ───────────────────────


_CLASSIFY_AND_PLAN_PROMPT = """You are the classifier+planner for a business-ready self-correcting RAG agent.
In ONE response, classify the user query AND produce a retrieval/verification plan.

## Classification fields
- primary_need: factual, procedural, comparative, temporal, exploratory
- needs_documents: true if internal docs likely contain the answer
- needs_web: true if public/web info is needed (latest news, external facts, updates beyond a textbook)
- needs_calculation: true if arithmetic, dates, or aggregation required
- temporal_focus: ISO date if about a specific date / "latest"
- temporal_qualifier: actual / estimate / preliminary / revised / projected / advance / unknown
- geographic_scope: global / national / state / district / city / region / unknown
- geography: specific place name if mentioned; use "" if none
- metric_hint: gdp / gsdp / gva / gva_share / output_share / employment / revenue / population / growth_rate / inflation / other / unknown
  → CRITICAL: GSDP, GVA, GVA_SHARE, OUTPUT_SHARE, and GDP are DIFFERENT metrics. Never conflate them.
- domain_hints: list of relevant domains
- ambiguity: low / medium / high
- rewrite: clearer, disambiguated version of the query

## Plan steps
Each step must have:
- action: one of retrieve_documents, search_web, calculate, synthesize
- queries: concrete search/retrieval queries (include EXACT metric acronym + geography)
- expected_claims: factual claims evidence should support
- rationale: why this step is needed

Rules:
- If metric_hint is set, every query MUST use that exact metric (never substitute GDP for GSDP, etc.).
- Include geography and temporal qualifier in queries when known.
- Prefer retrieve_documents first when needs_documents is true; add search_web when needs_web is true.

User query: {query}
"""


def _query_implies_web(query: str) -> bool:
    """Heuristic: recency / compare-to-textbook questions need public web evidence."""
    q = (query or "").lower()
    keys = (
        "latest",
        "most recent",
        "recent",
        "current",
        "today",
        "this year",
        "updated",
        "still hold",
        "still holds",
        "compare",
        "how does that picture compare",
    )
    return any(k in q for k in keys)


def _heuristic_classification(query: str) -> QueryClassification:
    """Build a classification from the query text when the planner LLM fails.

    Metric/geo agnostic: uses existing detectors only — no named places or agencies.
    Produces a short rewrite suitable for search (not the full user paragraph).
    """
    places = detect_place_mentions(query)
    geography = places[0] if places else ""
    metric = detect_metric_type(query)
    want_web = _query_implies_web(query)
    rewrite = compose_search_query(
        geography=geography,
        metric=metric,
        temporal=TemporalQualifier.UNKNOWN,
        base="",
    )
    # Keep rewrite short; append a compact topic cue from the first clause if needed
    if not rewrite:
        rewrite = re.split(r"[.?!\n]", query.strip())[0].strip()[:120]
    else:
        # Optional short topic from query without dumping the whole paragraph
        first = re.split(r"[.?!\n]", query.strip())[0].strip()
        if first and len(rewrite) < 40:
            rewrite = f"{rewrite} {first[:80]}".strip()
    rewrite = rewrite[:160]
    return QueryClassification(
        primary_need=QueryNeed.TEMPORAL if want_web else QueryNeed.FACTUAL,
        needs_documents=True,
        needs_web=want_web,
        geography=geography,
        metric_hint=metric,
        rewrite=rewrite or query[:120],
    )


def classify_and_plan(state: dict) -> dict:
    """Classify intent and build a retrieval plan in a single structured LLM call."""
    t0 = time.perf_counter()
    query = state["query"]
    llms = resolve_llms(state.get("provider", "auto"), user_credentials=state.get("user_credentials") or {})
    used_fallback = False

    try:
        plan = _llm_with_fallback(
            llms.planner,
            llms.planner_fallbacks,
            [SystemMessage(content=_CLASSIFY_AND_PLAN_PROMPT.format(query=query))],
            PlannerOutput,
        )
        classification = plan.classification or QueryClassification(rewrite=query)
    except Exception as exc:
        used_fallback = True
        logger.exception("classify_and_plan failed: %s", exc)
        classification = _heuristic_classification(query)
        short_q = classification.rewrite or query[:120]
        steps = [PlanStep(
            action="retrieve_documents",
            queries=[short_q],
            expected_claims=[],
            rationale="Fallback retrieve due to classify_and_plan failure",
        )]
        if classification.needs_web:
            steps.append(PlanStep(
                action="search_web",
                queries=[short_q],
                expected_claims=[],
                rationale="Fallback web search from heuristic rewrite",
            ))
        plan = PlannerOutput(classification=classification, steps=steps)

    if classification.primary_need in (QueryNeed.TEMPORAL, QueryNeed.EXPLORATORY):
        classification.needs_web = True
    # Recency / compare cues must force web even when the model (or fallback) missed it
    if _query_implies_web(query):
        classification.needs_web = True
    # Keep plan.classification in sync after mutation
    plan.classification = classification
    if not plan.steps:
        steps = []
        if classification.needs_documents:
            steps.append(PlanStep(
                action="retrieve_documents",
                queries=[classification.rewrite or query],
                expected_claims=[],
                rationale="Default document retrieval",
            ))
        if classification.needs_web:
            steps.append(PlanStep(
                action="search_web",
                queries=[classification.rewrite or query],
                expected_claims=[],
                rationale="Default web search",
            ))
        plan.steps = steps or [PlanStep(
            action="retrieve_documents",
            queries=[classification.rewrite or query],
            expected_claims=[],
            rationale="Fallback retrieve",
        )]
    elif classification.needs_web and not any(s.action == "search_web" for s in plan.steps):
        plan.steps = list(plan.steps) + [PlanStep(
            action="search_web",
            queries=[classification.rewrite or query],
            expected_claims=[],
            rationale="Added web search for recency/comparison query",
        )]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    # #region agent log
    qlow = (query or "").lower()
    agent_debug_log(
        "A",
        "nodes.py:classify_and_plan",
        "classify_plan_result",
        {
            "elapsed_ms": round(elapsed_ms),
            "used_fallback": used_fallback,
            "needs_web": bool(classification.needs_web),
            "needs_documents": bool(classification.needs_documents),
            "primary_need": getattr(classification.primary_need, "value", classification.primary_need),
            "plan_actions": [s.action for s in (plan.steps or [])],
            "rewrite": (classification.rewrite or "")[:120],
            "query_has_recent": any(k in qlow for k in ("recent", "latest", "most recent", "current")),
            "query_has_compare": "compare" in qlow or "textbook" in qlow,
        },
    )
    # #endregion
    logger.info("Node classify_and_plan completed in %.0fms", elapsed_ms)
    return {
        "classification": classification,
        "plan": plan,
        "planner_state": PlannerDecision.NOT_ENOUGH.value,
        "provider_used": llms.label,
    }


def classify_query(state: dict) -> dict:
    """Backward-compatible wrapper — prefer classify_and_plan in the live graph."""
    return classify_and_plan(state)


def build_plan(state: dict) -> dict:
    """No-op if classify_and_plan already produced a plan; otherwise run combined call."""
    if state.get("plan") and state.get("classification"):
        logger.info("Node build_plan skipped (already planned)")
        return {
            "plan": state["plan"],
            "classification": state["classification"],
            "planner_state": state.get("planner_state") or PlannerDecision.NOT_ENOUGH.value,
            "provider_used": state.get("provider_used"),
        }
    return classify_and_plan(state)


# ── Precise query composition (LLM-driven explicit searches) ─────────────────

def _precise_query(state: dict) -> str:
    """Build one precise, disambiguated retrieval/search query from classification."""
    classification = state.get("classification")
    query = state["query"]
    if not classification:
        return query
    return compose_search_query(
        geography=classification.geography,
        metric=classification.metric_hint,
        temporal=classification.temporal_qualifier,
        base=classification.rewrite or query,
    )


# ── Nodes: retrieval & web search ────────────────────────────────────────────


def _chunks_to_evidence(chunks: list[dict], source_type: SourceType, classification: QueryClassification | None = None) -> list[Evidence]:
    evidence = []
    for idx, ch in enumerate(chunks):
        text = ch.get("text", "")
        meta = ch.get("metadata", {}) or {}
        # Add parent context to metadata if available
        if ch.get("parent_context"):
            meta["parent_context"] = ch["parent_context"]
        date = _parse_date(meta.get("date") or meta.get("source_date"))
        raw_name = (
            meta.get("source_name")
            or meta.get("filename")
            or meta.get("title")
            or meta.get("source")
            or ""
        )
        source_name = str(raw_name).strip()
        if not source_name or source_name.lower() == "unknown":
            source_name = "document"
        ev = Evidence(
            text=text,
            source_type=source_type,
            source_name=source_name,
            source_url=meta.get("source_url"),
            source_date=date,
            retrieval_score=float(ch.get("score", 0.0)),
            chunk_index=idx,
            metadata=meta,
        )
        ev.authority_score = authority_score(ev.source_url, source_type)
        ev.recency_score = _recency_score(date)
        ev = _enrich_evidence_metadata(ev, classification)
        ev.combined_score = combined_score(ev, classification)
        evidence.append(ev)
    return evidence


async def retrieve_documents(state: dict) -> dict:
    """Retrieve chunks from the knowledge base and wrap them as Evidence."""
    t0 = time.perf_counter()
    if state.get("retrieval_count", 0) >= state.get("max_retrievals", settings.MAX_RETRIEVALS):
        logger.warning("Max retrievals reached; skipping document retrieval")
        return {
            "evidence": list(state.get("evidence", [])),
            "chunks": [ev.text for ev in state.get("evidence", []) if ev.source_type == SourceType.DOCUMENT],
            "retrieval_count": state.get("retrieval_count", 0),
        }

    query = state["query"]
    classification = state.get("classification")
    precise = _precise_query(state)
    
    # Use query decomposition for complex multi-part questions
    decomposition = decompose_query_text(query)
    
    # Use query expansion for better recall
    if classification:
        retrieval_queries = expand_queries(
            base_query=precise,
            metric=classification.metric_hint,
            geography=classification.geography,
            temporal=classification.temporal_qualifier,
            price_basis=getattr(classification, 'price_basis', PriceBasis.UNKNOWN),
        )
    else:
        retrieval_queries = [precise]
    
    # Add decomposition-based queries if applicable
    if decomposition.needs_decomposition and decomposition.sub_queries:
        decomp_queries = get_retrieval_queries_for_subqueries(
            base_query=query,
            sub_queries=decomposition.sub_queries,
            classification=classification,
        )
        for q in decomp_queries:
            if q.lower() not in {x.lower() for x in retrieval_queries}:
                retrieval_queries.append(q)
    
    # Add LLM rewrite if available
    if classification and classification.rewrite:
        if classification.rewrite not in retrieval_queries:
            retrieval_queries.append(classification.rewrite)
    
    # Add plan queries if available
    if state.get("plan"):
        for step in state["plan"].steps:
            if step.action == "retrieve_documents":
                for q in step.queries:
                    if q not in retrieval_queries:
                        retrieval_queries.append(q)

    user_id = state["user_id"]
    chat_id = state["chat_id"]
    all_evidence: list[Evidence] = list(state.get("evidence", []))

    try:
        tasks = [
            retrieve_chunks(q, user_id=user_id, top_k=30, scope="chat", chat_id=chat_id)
            for q in retrieval_queries[:3]
        ]
        results = await asyncio.gather(*tasks)

        chunks: list[dict] = []
        for query_result in results:
            for ch in query_result:
                dist = ch.get("distance")
                if dist is None:
                    score = 0.5
                else:
                    score = 1.0 - min(1.0, max(0.0, float(dist)))
                chunks.append({"text": ch["text"], "metadata": ch.get("metadata", {}), "score": score})

        if chunks:
            ranked = rerank(query, [("chunk", c["text"]) for c in chunks], top_k=len(chunks))
            text_to_score = {r.text: r.score for r in ranked}
            for ch in chunks:
                ev = _chunks_to_evidence([ch], SourceType.DOCUMENT, classification)[0]
                rs = text_to_score.get(ch["text"])
                ev.rerank_score = float(rs) if rs is not None else None
                ev = _enrich_evidence_metadata(ev, classification)
                ev.combined_score = combined_score(ev, classification)
                all_evidence.append(ev)
    except Exception as exc:
        logger.exception("Document retrieval failed: %s", exc)

    logger.info("Node retrieve_documents completed in %.0fms", (time.perf_counter() - t0) * 1000)
    return {
        "evidence": all_evidence,
        "chunks": [ev.text for ev in all_evidence if ev.source_type == SourceType.DOCUMENT],
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


async def search_web(state: dict) -> dict:
    """Run web search and wrap results as Evidence."""
    t0 = time.perf_counter()
    if state.get("search_count", 0) >= state.get("max_searches", settings.MAX_SEARCHES):
        logger.warning("Max searches reached; skipping web search")
        return {
            "evidence": list(state.get("evidence", [])),
            "search": state.get("search", []),
            "search_count": state.get("search_count", 0),
        }

    classification = state.get("classification")
    precise = _precise_query(state)
    search_queries = [precise]
    if classification:
        search_queries.append(classification.rewrite or state["query"])
    if state.get("plan"):
        for step in state["plan"].steps:
            if step.action == "search_web":
                search_queries.extend(step.queries)

    # Prefer one primary query; allow a second official-TLD variant (or gap-fill).
    max_keep = 2 if state.get("repair_mode") == "surgical" else 1
    queries = dedupe_search_queries(search_queries, max_keep=max_keep)
    if classification and (classification.needs_web or (classification.geography or "").strip()):
        extras: list[str] = []
        for q in queries:
            extras.extend(official_search_variants(q))
        queries = dedupe_search_queries(queries + extras, max_keep=2)
    prior_norms = {normalize_query(q) for q in (state.get("searxng_queries") or [])}
    if prior_norms:
        filtered = [q for q in queries if normalize_query(q) not in prior_norms]
        queries = filtered or queries[:1]

    all_evidence: list[Evidence] = list(state.get("evidence", []))
    search_strings: list[str] = []

    allow_tavily = True
    user_id = state.get("user_id")
    if user_id is not None:
        try:
            from app.core.database import AsyncLocalSession
            from app.core.usage import enforce_tavily_budget

            async with AsyncLocalSession() as session:
                await enforce_tavily_budget(session, user_id)
        except Exception as exc:
            from fastapi import HTTPException

            if isinstance(exc, HTTPException) and exc.status_code == 429:
                logger.warning("Tavily daily budget exhausted for user=%s; SearXNG only", user_id)
                allow_tavily = False
            else:
                logger.exception("Tavily budget check failed")

    async def _one_search(q: str) -> list[dict]:
        try:
            return await search_structured(
                q, max_results=6, user_id=user_id, allow_tavily=allow_tavily
            )
        except Exception as exc:
            logger.warning("Web search failed for query %r: %s", q, exc)
            return []

    results_lists = await asyncio.gather(*[_one_search(q) for q in queries])
    for results in results_lists:
        for r in results:
            text = r.get("content", "").strip()
            if not text:
                continue
            url = r.get("url")
            date = _parse_date(r.get("published_date") or r.get("date"))
            ev = Evidence(
                text=text,
                source_type=SourceType.WEB,
                source_name=r.get("title") or r.get("source") or urlparse(url or "").netloc,
                source_url=url,
                source_date=date,
                retrieval_score=float(r.get("score", 0.5)),
                metadata=r,
            )
            ev.authority_score = authority_score(url, SourceType.WEB)
            ev.recency_score = _recency_score(date, classification.temporal_focus if classification else None)
            ev = _enrich_evidence_metadata(ev, classification)
            ev.combined_score = combined_score(ev, classification)
            if not evidence_fits_classification(ev, classification):
                continue
            all_evidence.append(ev)
            search_strings.append(text)

    logger.info(
        "Node search_web completed in %.0fms (queries=%s)",
        (time.perf_counter() - t0) * 1000,
        queries,
    )
    return {
        "evidence": all_evidence,
        "search": search_strings,
        "search_count": state.get("search_count", 0) + 1,
        "searxng_queries": list(state.get("searxng_queries") or []) + list(queries),
    }


# ── Node: evidence assembly (ranking + conflict classification + context) ─────


def _count_tokens(text: str) -> int:
    return len(text) // 4


def assemble_evidence(state: dict) -> dict:
    """Deduplicate, rank (geo/metric-aware), classify conflicts, build context.

    Also merges the PERSISTENT cross-turn evidence state: established facts from
    prior turns are carried forward (provenance preserved) and ranked alongside
    freshly retrieved evidence, without dumping the whole conversation.
    """
    evidence: list[Evidence] = list(state.get("evidence", []))
    classification: QueryClassification | None = state.get("classification")
    prior: Any = state.get("prior_evidence_state") or state.get("evidence_state")

    # Merge prior established evidence (clearly marked, slightly discounted).
    merged: list[Evidence] = list(evidence)
    if prior and prior.established:
        seen_ids = {ev.evidence_id for ev in evidence}
        for ev in prior.established:
            if ev.evidence_id not in seen_ids:
                ev.combined_score = 0.6
                merged.append(ev)
                seen_ids.add(ev.evidence_id)

    # Deduplicate by near-duplicate text, then rank.
    unique: list[Evidence] = []
    seen_hashes = set()
    for ev in merged:
        h = hash(_normalize_claim(ev.text)[:200])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        ev = _enrich_evidence_metadata(ev, classification)
        ev.combined_score = combined_score(ev, classification)
        unique.append(ev)

    unique = rank_evidence(unique, classification)
    unique = filter_evidence_by_classification(unique, classification)

    # Classify conflicts (not all disagreements are contradictions).
    conflicts = detect_conflicts(unique)

    # Penalize ONLY genuine contradictions / source disagreements, never updates.
    for c in conflicts:
        if c.get("is_contradiction"):
            loser = next((e for e in unique if e.evidence_id == c["loser"]), None)
            if loser:
                loser.authority_score *= 0.7
                loser.combined_score = combined_score(loser, classification)
    unique = rank_evidence(unique, classification)

    # Build token-budgeted context with full structured metadata + E1..En cite keys.
    # Cap hard — large contexts dominate generate_answer latency on OpenRouter.
    CONTEXT_TOKEN_BUDGET = 8000
    context_parts: list[str] = []
    token_count = 0
    cite_map: dict[str, str] = {}
    kept: list[Evidence] = []

    cross_turn = to_context_block(prior)
    if cross_turn:
        context_parts.append(cross_turn)
        token_count += _count_tokens(cross_turn)

    cite_idx = 0
    for ev in unique:
        header = f"SOURCE: {ev.source_type.value} | {ev.source_name}"
        if ev.source_url:
            header += f" | {ev.source_url}"
        if ev.source_date:
            header += f" | {ev.source_date.date().isoformat()}"
        meta_parts = []
        if ev.metric_type != MetricType.UNKNOWN:
            meta_parts.append(f"metric={ev.metric_type.value}")
        if ev.metric_value:
            meta_parts.append(f"value={ev.metric_value}")
        if ev.price_basis != PriceBasis.UNKNOWN:
            meta_parts.append(f"price={ev.price_basis.value}")
        if ev.geographic_scope != GeographicScope.UNKNOWN:
            meta_parts.append(f"scope={ev.geographic_scope.value}")
        if ev.geography:
            meta_parts.append(f"geo={ev.geography}")
        if ev.year_period:
            meta_parts.append(f"period={ev.year_period}")
        if ev.temporal_qualifier != TemporalQualifier.UNKNOWN:
            meta_parts.append(f"temporal={ev.temporal_qualifier.value}")
        if ev.source_quality != SourceQuality.UNKNOWN:
            meta_parts.append(f"quality={ev.source_quality.value}")
        if meta_parts:
            header += f" | {' '.join(meta_parts)}"

        # Tentative cite key for budget check
        tentative_key = f"E{cite_idx + 1}"
        entry = f"[{tentative_key}] {header}\n{ev.text}"

        parent_ctx = ev.metadata.get("parent_context")
        if parent_ctx:
            entry += f"\n\n[CONTEXT] {parent_ctx[:500]}"

        entry_tokens = _count_tokens(entry)
        if token_count + entry_tokens > CONTEXT_TOKEN_BUDGET and context_parts:
            break

        cite_idx += 1
        cite_key = f"E{cite_idx}"
        ev.metadata = {**(ev.metadata or {}), "cite_key": cite_key}
        cite_map[cite_key] = ev.evidence_id
        entry = f"[{cite_key}] {header}\n{ev.text}"
        if parent_ctx:
            entry += f"\n\n[CONTEXT] {parent_ctx[:500]}"
        context_parts.append(entry)
        token_count += entry_tokens
        kept.append(ev)

    key_legend = "Cite keys: " + ", ".join(
        f"[{k}]→{v}" for k, v in cite_map.items()
    ) if cite_map else ""
    assembled = (key_legend + "\n\n" if key_legend else "") + "\n\n---\n\n".join(context_parts)

    gaps = grade_coverage(
        state.get("query", ""),
        kept or unique,
        classification,
    )
    logger.info(
        "Assembled context: %d items, ~%d tokens, cite_keys=%d, coverage_gaps=%d",
        len(kept),
        token_count,
        len(cite_map),
        len(gaps),
    )

    return {
        "evidence": unique,
        "conflicts": conflicts,
        "assembled_context": assembled,
        "cite_map": cite_map,
        "coverage_gaps": gaps,
    }


# ── Node: claim extraction & verification ────────────────────────────────────


_CLAIM_PROMPT = """You are a fact-checking assistant. Given the query, evidence context, and source metadata, extract 1-8 atomic factual claims that an answer should make.

For each claim provide:
- text: the claim text
- status: one of verified / partial / contradicted / unverified / uncertain
- claim_type: fact / inference / speculation
  -> "fact" if directly supported by a single evidence item
  -> "inference" if deduced from combining multiple evidence items (label it as inference!)
  -> "speculation" if extrapolating beyond evidence
- evidence_ids: list of evidence IDs that support it
- contradicting_evidence_ids: list of evidence IDs that contradict it
- reasoning: one-sentence justification
- repair_action: none / search_web / retrieve_documents / reject / rephrase

IMPORTANT economic-data checks:
- Verify the EXACT metric (GDP vs GSDP vs GVA vs GVA_SHARE vs OUTPUT_SHARE are DIFFERENT)
- Verify price basis (current/nominal vs constant/real)
- Verify geographic scope matches (national vs state vs district)
- Verify temporal qualifiers match (actual vs estimate vs revised vs advance vs projected)
- Clearly separate facts from inferences; never present an inference as a hard fact.

Evidence context:
{context}

User query: {query}
"""


class _ClaimList(BaseModel):
    claims: list[Claim]


def extract_verify_claims(state: dict) -> dict:
    """Pre-generation claim extraction (disabled for cost; post-gen verify covers it)."""
    return {"claims": []}


# ── Node: answer generation ──────────────────────────────────────────────────


_ANSWER_PROMPT = """You are a precise research assistant with structured reasoning. Answer the user's query using ONLY the evidence below.

## YOUR THINKING PROCESS (internal - don't show to user):
1. UNDERSTAND: What specific metric, geography, and time period is the user asking about?
2. GATHER: Which evidence items are most relevant? List their cite keys (E1, E2, ...).
3. VERIFY: Do sources agree? Any conflicts? Which source is more authoritative?
4. SYNTHESIZE: What's the direct answer? What caveats apply?

## ANSWER FORMAT (must follow this structure):

### Direct Answer
[One sentence answering the exact question asked, ending with an inline [E#] citation]

### Supporting Evidence
- **Fact 1** [E#]: [specific evidence from source]
- **Fact 2** [E#]: [specific evidence from source]
- ...

### Analysis & Caveats
- **Confidence**: High/Medium/Low (based on source quality & agreement)
- **Limitations**: [Any missing data, conflicts, or uncertainties]
- **Inference**: [If combining multiple sources, clearly label as inference]

## CRITICAL RULES (mandatory):
- GSDP ≠ GDP ≠ GVA ≠ GVA_SHARE ≠ OUTPUT_SHARE (DIFFERENT metrics)
- State ≠ National (Maharashtra GSDP ≠ India GDP)
- Current prices ≠ Constant prices (nominal vs real)
- Advance estimate ≠ Revised estimate ≠ Actual (different accuracy)
- EVERY factual sentence MUST end with a cite key from the evidence list, e.g. [E3]
- Use ONLY keys listed in the Cite keys legend / evidence headers — never invent IDs
- Do NOT put a bibliography-only list of citations; cite inline on the claim sentence
- If sources conflict, explain WHY (different years? different statuses?)
- If evidence is insufficient, say "Insufficient data" — NEVER guess
- Do not introduce numbers, totals, or percentages that do not appear in the cited evidence snippets
- If cited items use different metric types, name each metric; do not treat them as interchangeable

## CROSS-TURN EVIDENCE
If a [CROSS-TURN EVIDENCE STATE] block is present:
- Treat ESTABLISHED FACTS as verified but re-verify against current question
- Treat SUPERSEDED items as OUTDATED (don't present as current)
- Treat OPEN CONFLICTS as UNRESOLVED (don't assert either side as fact)

Evidence context:
{context}

Conflicts detected:
{conflicts}

User query: {query}
"""


def generate_answer(state: dict) -> dict:
    """Generate a cited answer from assembled evidence."""
    t0 = time.perf_counter()
    query = state["query"]
    context = state.get("assembled_context", "")
    conflicts = json.dumps(state.get("conflicts", []), indent=2, default=str)

    if not context.strip():
        return {
            "answer": "I don't have enough reliable information to answer this question.",
            "regeneration_count": state.get("regeneration_count", 0) + 1,
        }

    messages = [
        SystemMessage(content=_ANSWER_PROMPT.format(context=context, conflicts=conflicts, query=query)),
        HumanMessage(content=query),
    ]

    llms = resolve_llms(state.get("provider", "auto"), user_credentials=state.get("user_credentials") or {})
    try:
        answer, used_suffix = _invoke_chat(llms.generator, llms.generator_fallbacks, messages)
    except Exception as exc:
        logger.warning("generate_answer LLM failed: %s", exc)
        answer, used_suffix = "", "failed"
    used = llms.label if used_suffix == "primary" else f"{llms.label}+{used_suffix}"

    if not (answer or "").strip():
        logger.warning("generate_answer produced empty text; returning fallback")
        return {
            "answer": (
                "I retrieved evidence but could not generate a readable answer "
                "(empty model response). Please try again."
            ),
            "regeneration_count": state.get("regeneration_count", 0) + 1,
            "provider_used": used,
            "verification_errors": [],
        }

    evidence: list[Evidence] = state.get("evidence", [])
    cite_map: dict[str, str] = state.get("cite_map") or {}
    citation_check = validate_answer_citations(answer, evidence, cite_map=cite_map)
    answer = flag_uncited_in_answer(answer, citation_check)

    logger.info(
        "Node generate_answer completed in %.0fms (uncited=%d invalid_cites=%d answer_chars=%d)",
        (time.perf_counter() - t0) * 1000,
        len(citation_check.uncited_sentences),
        len(citation_check.invalid_citation_ids),
        len(answer),
    )
    return {
        "answer": answer,
        "regeneration_count": state.get("regeneration_count", 0) + 1,
        "provider_used": used,
        "verification_errors": [e.to_dict() for e in citation_check.errors],
    }


# ── Node: claim-level verification / hallucination check ─────────────────────


_VERIFY_PROMPT = """You are a strict claim-level hallucination checker.
Given the generated answer, the evidence context, and the extracted claims, verify each claim.

For each claim, decide:
- status: verified / partial / contradicted / unverified / uncertain
- claim_type: fact / inference / speculation
- evidence_ids: supporting evidence IDs
- contradicting_evidence_ids: contradicting evidence IDs
- reasoning: concise justification
- repair_action: none / search_web / retrieve_documents / reject / rephrase

CHECK LIST (fail the claim if any mismatch):
1. SUPPORT: Is every important factual claim backed by cited evidence?
2. EVIDENCE MATCH: Does the cited evidence actually support the claim?
3. METRIC: metric in claim == metric in evidence (GDP != GSDP != GVA != GVA_SHARE != OUTPUT_SHARE)
4. GEOGRAPHY: geography in claim == geography in evidence (national != state)
5. DATE: period in claim == period in evidence
6. STATUS: estimate status in claim == status in evidence (actual vs advance vs revised)
7. PRICE BASIS: current vs constant prices are not interchangeable
8. AUTHORITY: is the cited source authoritative enough for the claim?
9. INFERENCE: is the conclusion stronger than the evidence? (correlation != causation)
10. MIXING: are sources being mixed incorrectly (different geos/metrics)?
11. CONTRADICTIONS: are apparent conflicts genuine, or just different years/statuses?

Evidence context:
{context}

Answer:
{answer}

Existing claims:
{claims}
"""


_ANSWER_CITATION_RE = re.compile(r"\[([Ee]\d{1,3}|[a-fA-F0-9]{6,12})\]")


def _claims_from_answer_citations(
    answer: str,
    evidence: list[Evidence],
    cite_map: dict[str, str] | None = None,
) -> list[Claim]:
    """Heuristic claim extraction when the verifier LLM fails or times out."""
    if not answer or not evidence:
        return []
    known = {ev.evidence_id: ev for ev in evidence}
    claims: list[Claim] = []
    parts = re.split(r"(?<=[.!?])\s+", answer)
    for part in parts:
        text = part.strip()
        if len(text) < 20:
            continue
        ids = []
        for tok in _ANSWER_CITATION_RE.findall(text):
            eid = resolve_citation_token(tok, evidence, cite_map)
            if eid and eid in known:
                ids.append(eid)
        if not ids:
            continue
        claims.append(
            Claim(
                text=text[:500],
                status=ClaimStatus.PARTIALLY_VERIFIED,
                claim_type=ClaimType.FACT,
                evidence_ids=list(dict.fromkeys(ids)),
                reasoning="Extracted from answer citations (verifier skipped/failed)",
            )
        )
        if len(claims) >= 8:
            break
    return claims


def _verify_context_budget(context: str, max_chars: int = 6000) -> str:
    """Keep verify prompts small — large contexts dominate verifier latency."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n\n[... context truncated for verification ...]"


def _cascade_llm_residual(
    sentences: list[str],
    evidence: list[Evidence],
    state: dict,
) -> list[Claim]:
    """Batched structured verify for escalated sentences only."""
    if not sentences:
        return []
    cite_map = state.get("cite_map") or {}
    known = {ev.evidence_id: ev for ev in evidence}
    snippets = []
    for s in sentences:
        for tok in extract_citation_tokens(s):
            eid = resolve_citation_token(tok, evidence, cite_map)
            if eid and eid in known:
                snippets.append(f"[{eid}] {known[eid].text[:350]}")
    snippets = list(dict.fromkeys(snippets))[:12]
    prompt = (
        "Verify each claim against its cited evidence. Return JSON claims list.\n"
        "For each claim: status (verified/partial/contradicted/unverified/uncertain), "
        "evidence_ids, reasoning, repair_action (none/rephrase/search_web).\n\n"
        f"Evidence snippets:\n{chr(10).join(snippets)}\n\n"
        f"Claims to verify:\n" + "\n".join(f"- {s}" for s in sentences)
    )
    llms = resolve_llms(state.get("provider", "auto"), user_credentials=state.get("user_credentials") or {})
    result = _llm_with_fallback(
        llms.verifier,
        llms.verifier_fallbacks,
        [SystemMessage(content=prompt)],
        _ClaimList,
    )
    claims = list(result.claims) if result and result.claims else []
    if not claims:
        raise ValueError("Cascade LLM residual returned empty claims")
    return claims


def verify_answer_claims(state: dict) -> dict:
    """Hard citation check + cascade (or legacy LLM) claim verification."""
    t0 = time.perf_counter()
    answer = state.get("answer", "")
    context = _verify_context_budget(state.get("assembled_context", ""))
    evidence: list[Evidence] = state.get("evidence", [])
    cite_map: dict[str, str] = state.get("cite_map") or {}
    prior: Any = state.get("prior_evidence_state") or state.get("evidence_state")
    prior_claims: list[Claim] = state.get("claims", [])

    citation_check = validate_answer_citations(answer, evidence, cite_map=cite_map)
    citation_errors = list(citation_check.errors)

    claims: list[Claim] = []

    if settings.USE_VERIFY_CASCADE:
        try:
            cascade = run_verify_cascade(
                answer,
                evidence,
                cite_map=cite_map,
                llm_invoke=lambda sents, ev: _cascade_llm_residual(sents, ev, state),
            )
            claims = cascade.claims
        except Exception as exc:
            logger.warning("Verify cascade failed (%s); using citation heuristics", exc)
            claims = _claims_from_answer_citations(answer, evidence, cite_map)
    else:
        try:
            llms = resolve_llms(state.get("provider", "auto"), user_credentials=state.get("user_credentials") or {})
            result = _llm_with_fallback(
                llms.verifier,
                llms.verifier_fallbacks,
                [SystemMessage(content=_VERIFY_PROMPT.format(
                    context=context,
                    answer=answer[:4000],
                    claims=json.dumps([c.model_dump() for c in prior_claims], default=str)[:2000],
                ))],
                _ClaimList,
            )
            claims = result.claims
        except Exception as exc:
            logger.warning("Claim verification failed (%s); using citation heuristics", exc)
            claims = _claims_from_answer_citations(answer, evidence, cite_map)

    if not claims:
        claims = list(citation_check.claims) or _claims_from_answer_citations(answer, evidence, cite_map)
    else:
        existing_texts = {c.text.strip().lower() for c in claims}
        for c in citation_check.claims:
            if c.status == ClaimStatus.UNVERIFIED and c.text.strip().lower() not in existing_texts:
                claims.append(c)

    known_ids = {ev.evidence_id for ev in evidence}
    for claim in claims:
        bad = [i for i in claim.evidence_ids if i not in known_ids]
        if bad:
            claim.status = ClaimStatus.UNVERIFIED
            claim.reasoning = (claim.reasoning + " " if claim.reasoning else "") + (
                f"Invalid citation ids: {bad}"
            )
            claim.repair_action = claim.repair_action or "rephrase"
        elif not claim.evidence_ids and claim.status not in (
            ClaimStatus.VERIFIED,
            ClaimStatus.PARTIALLY_VERIFIED,
        ):
            claim.status = ClaimStatus.UNVERIFIED
            claim.repair_action = claim.repair_action or "rephrase"

    for claim in claims:
        for ev in evidence:
            is_contra, reason = is_genuine_contradiction(claim.text, ev.text)
            if is_contra:
                claim.status = ClaimStatus.CONTRADICTED
                claim.contradicting_evidence_ids = list(set(claim.contradicting_evidence_ids + [ev.evidence_id]))
                claim.reasoning = f"Deterministic contradiction: {reason}"
                claim.repair_action = claim.repair_action or "rephrase"

    errors = audit_claims(claims, evidence, prior)
    all_errors = [e.to_dict() for e in citation_errors] + [e.to_dict() for e in errors]

    citation_usage = [
        CitationUsage(claim_id=claim.claim_id, evidence_ids=claim.evidence_ids)
        for claim in claims
    ]

    logger.info(
        "Node verify_answer_claims completed in %.0fms (claims=%d citation_errors=%d cascade=%s)",
        (time.perf_counter() - t0) * 1000,
        len(claims),
        len(citation_errors),
        settings.USE_VERIFY_CASCADE,
    )
    # #region agent log
    agent_debug_log(
        "E",
        "nodes.py:verify_answer_claims",
        "verify_complete",
        {
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "claims": len(claims),
            "citation_errors": len(citation_errors),
            "uncited": len(citation_check.uncited_sentences),
            "failed": sum(
                1
                for c in claims
                if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)
            ),
            "cascade": settings.USE_VERIFY_CASCADE,
        },
    )
    # #endregion
    return {
        "claims": claims,
        "citation_usage": citation_usage,
        "verification_errors": all_errors,
    }


# ── Node: repair ─────────────────────────────────────────────────────────────


def repair_claims(state: dict) -> dict:
    """Surgical patch (cascade) or legacy re-search/regenerate repair."""
    claims: list[Claim] = state.get("claims", [])
    failed = [
        c for c in claims
        if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)
    ]

    if not failed:
        return {"repair_state": RepairDecision.SATISFACTORY.value, "final_status": "answered"}

    if settings.USE_VERIFY_CASCADE:
        return _repair_claims_surgical(state, failed)

    # Legacy path
    if state.get("regeneration_count", 0) >= state.get("max_regenerations", settings.MAX_REGENERATIONS):
        return {
            "repair_state": RepairDecision.MAX_ATTEMPTS.value,
            "final_status": "max_attempts",
            "answer": _add_caveats(state.get("answer", ""), failed),
        }

    new_steps: list[PlanStep] = []
    for claim in failed:
        if claim.repair_action in ("search_web", ""):
            new_steps.append(PlanStep(
                action="search_web",
                queries=[claim.text],
                expected_claims=[claim.text],
                rationale=f"Repair unverified claim: {claim.text[:80]}",
            ))
        elif claim.repair_action == "retrieve_documents":
            new_steps.append(PlanStep(
                action="retrieve_documents",
                queries=[claim.text],
                expected_claims=[claim.text],
                rationale=f"Repair contradicted claim: {claim.text[:80]}",
            ))

    seen = set()
    deduped = []
    for step in new_steps:
        key = (step.action, tuple(step.queries))
        if key not in seen:
            seen.add(key)
            deduped.append(step)

    retrieval_available = state.get("retrieval_count", 0) < state.get("max_retrievals", settings.MAX_RETRIEVALS)
    search_available = state.get("search_count", 0) < state.get("max_searches", settings.MAX_SEARCHES)
    actionable = [
        s for s in deduped
        if (s.action == "retrieve_documents" and retrieval_available)
        or (s.action == "search_web" and search_available)
    ]
    if not actionable:
        logger.warning(
            "Repair needed but no actionable steps (retrieval=%s search=%s failed=%d)",
            retrieval_available,
            search_available,
            len(failed),
        )
        return {
            "repair_state": RepairDecision.MAX_ATTEMPTS.value,
            "final_status": "max_attempts",
            "answer": _add_caveats(state.get("answer", ""), failed),
        }

    if state.get("regeneration_count", 0) >= 1 and (
        state.get("search_count", 0) >= 1 or state.get("retrieval_count", 0) >= 1
    ):
        if state.get("final_status") == "repairing":
            return {
                "repair_state": RepairDecision.MAX_ATTEMPTS.value,
                "final_status": "max_attempts",
                "answer": _add_caveats(state.get("answer", ""), failed),
            }

    return {
        "repair_state": RepairDecision.REPAIR.value,
        "plan": PlannerOutput(
            classification=state.get("classification") or QueryClassification(),
            steps=actionable,
        ),
        "final_status": "repairing",
    }


def _repair_claims_surgical(state: dict, failed: list[Claim]) -> dict:
    """One-pass surgical sentence patch; optional single coverage-gap search."""
    repair_pass = int(state.get("repair_pass_count") or 0)
    max_passes = int(getattr(settings, "MAX_REPAIR_PASSES", 1) or 1)
    # #region agent log
    agent_debug_log(
        "E",
        "nodes.py:_repair_claims_surgical",
        "surgical_repair_entry",
        {
            "failed_count": len(failed),
            "repair_pass": repair_pass,
            "coverage_gaps": list(state.get("coverage_gaps") or [])[:3],
            "repair_mode": state.get("repair_mode"),
            "final_status": state.get("final_status"),
        },
    )
    # #endregion

    # After gap search: patch with refreshed evidence and end
    if state.get("repair_mode") == "surgical" and state.get("final_status") == "repairing":
        return _do_surgical_patch(state, failed, repair_pass)

    if repair_pass >= max_passes:
        return {
            "repair_state": RepairDecision.MAX_ATTEMPTS.value,
            "final_status": "max_attempts",
            "answer": _add_caveats(state.get("answer", ""), failed),
            "repair_mode": "surgical",
        }

    gaps = list(state.get("coverage_gaps") or [])
    search_available = state.get("search_count", 0) < state.get("max_searches", settings.MAX_SEARCHES)
    prior_norms = {normalize_query(q) for q in (state.get("searxng_queries") or [])}

    if gaps and search_available:
        # One targeted gap query only
        gap_q = gaps[0]
        # Prefer a short search query derived from the gap + original query
        query = state.get("query", "")
        targeted = gap_q if len(gap_q) < 160 else f"{query} {gap_q[:80]}"
        if normalize_query(targeted) in prior_norms:
            # Already searched this — patch with existing evidence
            return _do_surgical_patch(state, failed, repair_pass)
        return {
            "repair_state": RepairDecision.REPAIR.value,
            "repair_mode": "surgical",
            "final_status": "repairing",
            "plan": PlannerOutput(
                classification=state.get("classification") or QueryClassification(),
                steps=[PlanStep(
                    action="search_web",
                    queries=[targeted],
                    expected_claims=[c.text for c in failed[:3]],
                    rationale=f"Coverage gap fill: {gap_q[:120]}",
                )],
            ),
        }

    return _do_surgical_patch(state, failed, repair_pass)


def _do_surgical_patch(state: dict, failed: list[Claim], repair_pass: int) -> dict:
    """Patch flagged sentences inline and force end via max_attempts."""
    evidence: list[Evidence] = state.get("evidence", [])
    cite_map: dict[str, str] = state.get("cite_map") or {}
    answer = state.get("answer", "")

    llms = resolve_llms(state.get("provider", "auto"), user_credentials=state.get("user_credentials") or {})

    def _rewrite(prompt: str) -> str:
        text, _ = _invoke_chat(
            llms.generator,
            llms.generator_fallbacks,
            [SystemMessage(content=prompt), HumanMessage(content="Rewrite the target sentence only.")],
        )
        return text

    try:
        patched = patch_flagged_sentences(answer, failed, evidence, cite_map, _rewrite)
    except Exception as exc:
        logger.warning("Surgical patch failed: %s", exc)
        patched = answer

    failed_ids = {c.claim_id for c in failed}
    updated = []
    for c in state.get("claims", []):
        if c.claim_id in failed_ids:
            nc = c.model_copy(deep=True)
            if nc.status == ClaimStatus.CONTRADICTED:
                updated.append(nc)
            elif nc.status != ClaimStatus.VERIFIED:
                nc.status = ClaimStatus.UNCERTAIN
                nc.reasoning = (nc.reasoning or "") + " | surgical patch attempted"
                updated.append(nc)
            else:
                updated.append(nc)
        else:
            updated.append(c)

    still_hard = [
        c for c in updated
        if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED)
    ]
    uncertain = [c for c in updated if c.status == ClaimStatus.UNCERTAIN]

    if still_hard:
        final_status = "max_attempts"
        repair_state = RepairDecision.MAX_ATTEMPTS.value
        patched = _add_caveats(patched, still_hard)
    elif uncertain:
        final_status = "partial"
        repair_state = RepairDecision.SATISFACTORY.value
        patched = _add_caveats(patched, uncertain)
    else:
        final_status = "answered"
        repair_state = RepairDecision.SATISFACTORY.value

    logger.info(
        "Surgical repair pass %d patched %d failed claims status=%s",
        repair_pass + 1,
        len(failed),
        final_status,
    )
    return {
        "answer": patched,
        "claims": updated,
        "repair_pass_count": repair_pass + 1,
        "repair_state": repair_state,
        "final_status": final_status,
        "repair_mode": "surgical",
        "coverage_gaps": [],
    }


def _add_caveats(answer: str, failed: list[Claim]) -> str:
    if not failed:
        return answer
    caveat = "\n\nNote: The following claims could not be fully verified: " + "; ".join(
        f'"{c.text}"' for c in failed
    )
    if "could not be fully verified" in (answer or ""):
        return answer
    return (answer or "") + caveat


# ── Legacy compatibility wrappers (deprecated) ───────────────────────────────


def Planner(state: dict) -> dict:
    return build_plan(state)


def Hallucination_Check(state: dict) -> dict:
    return verify_answer_claims(state)


def search_tool(state: dict) -> dict:
    return search_web(state)


async def Initial_Chunks(state: dict) -> dict:
    return await retrieve_documents(state)


def should_retrieve_documents(state: dict) -> str:
    classification = state.get("classification")
    if classification and classification.needs_documents:
        return "retrieve_documents"
    return "assemble"


def should_search_web(state: dict) -> str:
    classification = state.get("classification")
    decision = "search_web" if (classification and classification.needs_web) else "assemble"
    # #region agent log
    agent_debug_log(
        "C",
        "nodes.py:should_search_web",
        "search_routing",
        {
            "decision": decision,
            "needs_web": bool(classification.needs_web) if classification else None,
            "search_count": state.get("search_count", 0),
        },
    )
    # #endregion
    return decision


def should_post_assemble(state: dict) -> str:
    """After assemble: surgical repair loop skips generate; first pass continues."""
    if (
        settings.USE_VERIFY_CASCADE
        and state.get("repair_mode") == "surgical"
        and state.get("final_status") == "repairing"
        and int(state.get("repair_pass_count") or 0) == 0
    ):
        return "repair_claims"
    return "extract_verify_claims"


def hallucination_router(state: dict) -> str:
    claims = state.get("claims", [])
    failed = [
        c for c in claims
        if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)
    ]
    if not failed:
        decision = "satisfactory"
    elif settings.USE_VERIFY_CASCADE:
        if int(state.get("repair_pass_count") or 0) >= int(settings.MAX_REPAIR_PASSES or 1):
            decision = "max_attempts"
        else:
            decision = "repair"
    elif state.get("regeneration_count", 0) >= state.get("max_regenerations", settings.MAX_REGENERATIONS):
        decision = "max_attempts"
    else:
        decision = "repair"
    # #region agent log
    agent_debug_log(
        "E",
        "nodes.py:hallucination_router",
        "verify_route",
        {
            "decision": decision,
            "failed_count": len(failed),
            "failed_statuses": [getattr(c.status, "value", c.status) for c in failed[:8]],
            "failed_previews": [c.text[:80] for c in failed[:4]],
            "repair_pass_count": state.get("repair_pass_count", 0),
            "citation_error_count": len(state.get("verification_errors") or []),
        },
    )
    # #endregion
    return decision


def planner_router(state: dict) -> str:
    plan = state.get("plan")
    if plan:
        actions = {s.action for s in plan.steps}
        if "retrieve_documents" in actions:
            return "retrieve_documents"
        if "search_web" in actions:
            return "search_web"
    return "generate"


def Hallucination_Check_router(state: dict) -> str:
    return hallucination_router(state)
