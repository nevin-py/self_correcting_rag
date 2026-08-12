"""
Agent nodes for the business-ready self-correcting RAG pipeline.

Architecture (each node has a single, observable responsibility):

    classify_query       -> structured query intent & source needs
    build_plan           -> typed plan with metric/geo-disambiguated queries
    retrieve_documents   -> vector + BM25 retrieval over the knowledge base
    search_web           -> web search (Tavily / SearXNG / Wikipedia)
    assemble_evidence    -> SOURCE RANKING + conflict classification + context assembly
    extract_verify_claims -> claim extraction (disabled for cost; post-gen verify covers it)
    generate_answer      -> cited answer generation (consults cross-turn evidence state)
    verify_answer_claims -> claim-level verification + deterministic structured errors
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
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.agent.conflicts import detect_conflicts, is_genuine_contradiction
from app.agent.evidence_state import to_context_block
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
from app.agent.ranking import combined_score, rank_evidence
from app.agent.reranker import rerank
from app.agent.search_tool import search_structured
from app.agent.source_authority import authority_score, classify_source_quality
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
from app.documents.clients import chat_llm, routing_llm
from app.documents.clients import openrouter_planner_llm, openrouter_generator_llm, openrouter_hallucination_llm
from app.documents.clients import get_chroma_client
from app.documents.service import retrieve_chunks
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── LLM helpers ──────────────────────────────────────────────────────────────


def _llm_with_fallback(primary: Any, fallback: Any | None, messages: list, output_schema: Any):
    """Call primary LLM; on failure invoke fallback with structured output."""
    try:
        bound = primary.with_structured_output(output_schema)
        return bound.invoke(messages)
    except Exception as exc:
        logger.warning("Primary LLM failed (%s), trying fallback", exc)
        if fallback is None:
            raise
        bound = fallback.with_structured_output(output_schema)
        return bound.invoke(messages)


def _strip_json_markers(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(_strip_json_markers(text))
    except json.JSONDecodeError:
        return {}


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


# ── Nodes: classification & planning ─────────────────────────────────────────


_CLASSIFICATION_PROMPT = """You are a query classifier for a business-ready RAG system.
Analyze the user's question and emit a structured classification.

Guidelines:
- primary_need: factual, procedural, comparative, temporal, exploratory
- needs_documents: true if internal docs likely contain the answer
- needs_web: true if public/web info is needed (latest news, external facts)
- needs_calculation: true if arithmetic, dates, or aggregation required
- temporal_focus: ISO date if the question is about a specific date / "latest"
- temporal_qualifier: actual / estimate / preliminary / revised / projected / advance / unknown
- geographic_scope: global / national / state / district / city / region / unknown
- geography: the specific place name if mentioned (e.g. "Karnataka", "India", "Maharashtra", "USA")
- metric_hint: gdp / gsdp / gva / gva_share / output_share / employment / revenue / population / growth_rate / inflation / other / unknown
  → CRITICAL: GSDP, GVA, GVA_SHARE, OUTPUT_SHARE, and GDP are DIFFERENT metrics. Never conflate them.
- domain_hints: list of relevant domains
- ambiguity: low / medium / high
- rewrite: a clearer, disambiguated version of the query

User query: {query}
"""


def classify_query(state: dict) -> dict:
    """Classify user intent and source requirements."""
    query = state["query"]
    try:
        classification = _llm_with_fallback(
            openrouter_planner_llm or routing_llm,
            routing_llm,
            [SystemMessage(content=_CLASSIFICATION_PROMPT.format(query=query))],
            QueryClassification,
        )
    except Exception as exc:
        logger.exception("Query classification failed: %s", exc)
        classification = QueryClassification(
            primary_need=QueryNeed.FACTUAL,
            needs_documents=True,
            needs_web=False,
            rewrite=query,
        )

    if classification.primary_need in (QueryNeed.TEMPORAL, QueryNeed.EXPLORATORY):
        classification.needs_web = True

    return {"classification": classification}


_PLANNER_PROMPT = """You are a planner for a self-correcting RAG agent.
Given the user query and its classification, produce a structured plan with explicit steps.

Each step must have:
- action: one of retrieve_documents, search_web, calculate, synthesize
- queries: concrete search/retrieval queries to run (include the EXACT metric acronym
  and geography from the classification, e.g. "Maharashtra GSDP advance estimate")
- expected_claims: what factual claims you expect evidence to support
- rationale: why this step is needed

CRITICAL classification handling:
- metric_hint: if set (not "unknown"), queries MUST reference that exact metric
  (gsdp, gva, gva_share, output_share, gdp). NEVER substitute one for another.
- geographic_scope + geography: include the specific geography in every query.
- temporal_qualifier: include it in queries (e.g. "advance estimate", "actual").
- price basis: if the query distinguishes current vs constant prices, include it.

Classification: {classification}
User query: {query}
"""


def build_plan(state: dict) -> dict:
    """Build a structured retrieval and verification plan."""
    query = state["query"]
    classification = state.get("classification") or QueryClassification(rewrite=query)

    try:
        plan = _llm_with_fallback(
            openrouter_planner_llm or routing_llm,
            routing_llm,
            [SystemMessage(content=_PLANNER_PROMPT.format(
                query=query,
                classification=classification.model_dump_json(),
            ))],
            PlannerOutput,
        )
    except Exception as exc:
        logger.exception("Planner failed: %s", exc)
        plan = PlannerOutput(
            classification=classification,
            steps=[PlanStep(
                action="retrieve_documents",
                queries=[classification.rewrite or query],
                expected_claims=[],
                rationale="Fallback retrieve due to planner failure",
            )],
        )

    return {"plan": plan, "planner_state": PlannerDecision.NOT_ENOUGH.value}


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
        source_name = meta.get("source_name") or meta.get("filename") or meta.get("title") or "unknown"
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
                ev.rerank_score = text_to_score.get(ch["text"])
                ev = _enrich_evidence_metadata(ev, classification)
                ev.combined_score = combined_score(ev, classification)
                all_evidence.append(ev)
    except Exception as exc:
        logger.exception("Document retrieval failed: %s", exc)

    return {
        "evidence": all_evidence,
        "chunks": [ev.text for ev in all_evidence if ev.source_type == SourceType.DOCUMENT],
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


async def search_web(state: dict) -> dict:
    """Run web search and wrap results as Evidence."""
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

    all_evidence: list[Evidence] = list(state.get("evidence", []))
    search_strings: list[str] = []

    for q in search_queries[:3]:
        try:
            results = await search_structured(q, max_results=10)
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
                all_evidence.append(ev)
                search_strings.append(text)
        except Exception as exc:
            logger.warning("Web search failed for query %r: %s", q, exc)

    return {
        "evidence": all_evidence,
        "search": search_strings,
        "search_count": state.get("search_count", 0) + 1,
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

    # Build token-budgeted context with full structured metadata.
    CONTEXT_TOKEN_BUDGET = 12000
    context_parts: list[str] = []
    token_count = 0

    cross_turn = to_context_block(prior)
    if cross_turn:
        context_parts.append(cross_turn)
        token_count += _count_tokens(cross_turn)

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
        entry = f"{header}\n[{ev.evidence_id}] {ev.text}"
        
        # Include parent context if available for better understanding
        parent_ctx = ev.metadata.get("parent_context")
        if parent_ctx:
            entry += f"\n\n[CONTEXT] {parent_ctx[:500]}"  # Limit to prevent overflow
        
        entry_tokens = _count_tokens(entry)
        if token_count + entry_tokens > CONTEXT_TOKEN_BUDGET and context_parts:
            break
        context_parts.append(entry)
        token_count += entry_tokens

    assembled = "\n\n---\n\n".join(context_parts)
    logger.info("Assembled context: %d items, ~%d tokens", len(context_parts), token_count)

    return {
        "evidence": unique,
        "conflicts": conflicts,
        "assembled_context": assembled,
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
2. GATHER: Which evidence items are most relevant? List their evidence IDs.
3. VERIFY: Do sources agree? Any conflicts? Which source is more authoritative?
4. SYNTHESIZE: What's the direct answer? What caveats apply?

## ANSWER FORMAT (must follow this structure):

### Direct Answer
[One sentence answering the exact question asked, with inline citations]

### Supporting Evidence
- **Fact 1** [citation]: [specific evidence from source]
- **Fact 2** [citation]: [specific evidence from source]
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
- ALWAYS cite EVERY fact with evidence ID: [a1b2c3d4]
- If sources conflict, explain WHY (different years? different statuses?)
- If evidence is insufficient, say "Insufficient data" — NEVER guess

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

    try:
        response = chat_llm.invoke(messages)
        answer = response.content
    except Exception as exc:
        logger.warning("Primary generator failed: %s", exc)
        if openrouter_generator_llm:
            response = openrouter_generator_llm.invoke(messages)
            answer = response.content
        else:
            raise

    return {
        "answer": answer,
        "regeneration_count": state.get("regeneration_count", 0) + 1,
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


def verify_answer_claims(state: dict) -> dict:
    """Verify every claim in the generated answer + run deterministic structured checks."""
    query = state["query"]
    answer = state.get("answer", "")
    context = state.get("assembled_context", "")
    evidence: list[Evidence] = state.get("evidence", [])
    prior: Any = state.get("prior_evidence_state") or state.get("evidence_state")
    prior_claims: list[Claim] = state.get("claims", [])

    try:
        bound = (openrouter_hallucination_llm or routing_llm).with_structured_output(_ClaimList)
        result = bound.invoke([
            SystemMessage(content=_VERIFY_PROMPT.format(context=context, answer=answer, claims=json.dumps([c.model_dump() for c in prior_claims], default=str)))
        ])
        claims = result.claims
    except Exception as exc:
        logger.exception("Claim verification failed: %s", exc)
        claims = []

    # Merge deterministic contradiction checks.
    for claim in claims:
        for ev in evidence:
            is_contra, reason = is_genuine_contradiction(claim.text, ev.text)
            if is_contra:
                claim.status = ClaimStatus.CONTRADICTED
                claim.contradicting_evidence_ids = list(set(claim.contradicting_evidence_ids + [ev.evidence_id]))
                claim.reasoning = f"Deterministic contradiction: {reason}"
                claim.repair_action = claim.repair_action or "search_web"

    # Deterministic structured verification (metric/geo/date/status/authority/inference/causation).
    errors = audit_claims(claims, evidence, prior)

    citation_usage = [
        CitationUsage(claim_id=claim.claim_id, evidence_ids=claim.evidence_ids)
        for claim in claims
    ]

    return {
        "claims": claims,
        "citation_usage": citation_usage,
        "verification_errors": errors,
    }


# ── Node: repair ─────────────────────────────────────────────────────────────


def repair_claims(state: dict) -> dict:
    """Decide whether to repair, give up, or accept the current answer."""
    claims: list[Claim] = state.get("claims", [])
    failed = [c for c in claims if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)]

    if not failed:
        return {"repair_state": RepairDecision.SATISFACTORY.value, "final_status": "answered"}

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

    return {
        "repair_state": RepairDecision.REPAIR.value,
        "plan": PlannerOutput(
            classification=state.get("classification") or QueryClassification(),
            steps=deduped,
        ),
        "final_status": "repairing",
    }


def _add_caveats(answer: str, failed: list[Claim]) -> str:
    caveat = "\n\nNote: The following claims could not be fully verified: " + "; ".join(
        f'"{c.text}"' for c in failed
    )
    return answer + caveat


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
    if classification and classification.needs_web:
        return "search_web"
    return "assemble"


def hallucination_router(state: dict) -> str:
    claims = state.get("claims", [])
    failed = [c for c in claims if c.status in (ClaimStatus.UNVERIFIED, ClaimStatus.CONTRADICTED, ClaimStatus.UNCERTAIN)]
    if not failed:
        return "satisfactory"
    if state.get("regeneration_count", 0) >= state.get("max_regenerations", settings.MAX_REGENERATIONS):
        return "max_attempts"
    return "repair"


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
