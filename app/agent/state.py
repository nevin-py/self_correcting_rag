"""
Typed state and structured schemas for the business-ready self-correcting RAG agent.

Design principles
-----------------
1. Provenance first: every piece of evidence knows where it came from.
2. Document vs. web separation: first-class source types with different scoring heuristics.
3. Claim-level verification: answers are decomposed into claims that are individually
   checked against evidence.
4. Deterministic conflict detection: evidence contradictions are surfaced with explicit
   reasoning, not hidden inside an LLM call.
5. Structured planning: the planner emits a typed plan (query class, sub-queries,
   expected claims, source strategy) rather than a single enum.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ── Enums ────────────────────────────────────────────────────────────────────


class SourceType(str, enum.Enum):
    DOCUMENT = "document"      # User-uploaded / internal knowledge base
    WEB = "web"                # Search results / public web
    LLM_KNOWLEDGE = "llm"      # Unverifiable world knowledge fallback
    UNKNOWN = "unknown"


class ClaimStatus(str, enum.Enum):
    VERIFIED = "verified"              # Supported by evidence
    PARTIALLY_VERIFIED = "partial"     # Supported but with caveats / low authority
    CONTRADICTED = "contradicted"      # Directly contradicted by evidence
    UNVERIFIED = "unverified"          # No evidence found
    UNCERTAIN = "uncertain"            # Conflicting or insufficient evidence


class PlannerDecision(str, enum.Enum):
    """Backward-compatible high-level routing decision."""
    EVIDENT = "evident"
    NOT_ENOUGH = "not_enough"


class RepairDecision(str, enum.Enum):
    """Backward-compatible repair routing decision."""
    SATISFACTORY = "satisfactory"
    REPAIR = "repair"
    MAX_ATTEMPTS = "max_attempts"


class QueryNeed(str, enum.Enum):
    """Why does the user need an answer? Informs source strategy."""
    FACTUAL = "factual"          # Seeking a verifiable fact
    PROCEDURAL = "procedural"    # How-to / step-by-step
    COMPARATIVE = "comparative"  # Compare options
    TEMPORAL = "temporal"        # Time-sensitive / latest info
    EXPLORATORY = "exploratory"  # Open-ended brainstorming
    UNKNOWN = "unknown"


class MetricType(str, enum.Enum):
    """Economic / statistical metric category to prevent cross-metric confusion."""
    GDP = "gdp"
    GSDP = "gsdp"                      # Gross State Domestic Product
    GVA = "gva"                        # Gross Value Added
    GVA_SHARE = "gva_share"            # Sector share of GVA (NOT the same as share of output)
    OUTPUT_SHARE = "output_share"      # Sector share of output (NOT the same as GVA share)
    EMPLOYMENT = "employment"
    REVENUE = "revenue"
    POPULATION = "population"
    GROWTH_RATE = "growth_rate"
    INFLATION = "inflation"
    OTHER = "other"
    UNKNOWN = "unknown"


class PriceBasis(str, enum.Enum):
    """Nominal vs real prices — conflating these is a classic error.

    CURRENT  = nominal / current-price figures (absolute rupee/dollar values)
    CONSTANT = real / constant-price figures (inflation-adjusted, e.g. 2011-12 prices)
    """
    CURRENT = "current"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class ConflictType(str, enum.Enum):
    """Classification of an apparent evidence conflict (not all are contradictions)."""
    NONE = "none"
    GENUINE_CONTRADICTION = "genuine_contradiction"          # same metric/geo/period, opposite facts
    SOURCE_DISAGREEMENT = "source_disagreement"              # same facts, different sources disagree
    DIFFERENT_YEARS = "different_years"                      # different periods, not a contradiction
    DIFFERENT_ESTIMATE_STATUS = "different_estimate_status"  # advance vs revised vs actual (an update)
    DIFFERENT_METRICS = "different_metrics"                  # e.g. GSDP vs GVA
    DIFFERENT_GEOGRAPHIC_SCOPES = "different_geographic_scopes"  # national vs state
    REVISED_VS_UNREVISED = "revised_vs_unrevised"            # updated figure replaces older one
    INSUFFICIENT_OVERLAP = "insufficient_overlap"            # unrelated, not a conflict


class GeographicScope(str, enum.Enum):
    """Geographic level to prevent scope confusion (national vs state vs city)."""
    GLOBAL = "global"
    NATIONAL = "national"
    STATE = "state"
    DISTRICT = "district"
    CITY = "city"
    REGION = "region"            # Multi-state / sub-continent
    UNKNOWN = "unknown"


class TemporalQualifier(str, enum.Enum):
    """Distinguishes actuals from estimates/projections to prevent temporal confusion."""
    ACTUAL = "actual"
    ESTIMATE = "estimate"
    PRELIMINARY = "preliminary"
    REVISED = "revised"
    PROJECTED = "projected"
    ADVANCE = "advance"          # Advance estimate (before full data)
    UNKNOWN = "unknown"


class SourceQuality(str, enum.Enum):
    """Is the source primary (original data) or secondary (citing primary)?"""
    PRIMARY = "primary"          # Original report / data release (e.g. RBI report, Census)
    SECONDARY = "secondary"      # News article or analysis citing primary source
    TERTIARY = "tertiary"        # Wikipedia, aggregators, blogs
    UNKNOWN = "unknown"


class ClaimType(str, enum.Enum):
    """Distinguishes verbatim facts from inferences drawn by the LLM."""
    FACT = "fact"                # Directly stated in evidence
    INFERENCE = "inference"      # Deduced from combining multiple evidence items
    SPECULATION = "speculation"  # Extrapolation or opinion not grounded in evidence


# ── Evidence / Claim / Plan models ───────────────────────────────────────────


class Evidence(BaseModel):
    """A single piece of retrieved or searched evidence."""

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    source_type: SourceType
    source_name: str = ""                    # Document name or site name
    source_url: str | None = None
    source_date: datetime | None = None      # Publication / retrieval date
    source_quality: SourceQuality = SourceQuality.UNKNOWN   # Primary / secondary / tertiary
    retrieval_score: float = 0.0             # Vector/BM25 score (0-1)
    rerank_score: float | None = None        # Cross-encoder rerank score
    authority_score: float = 0.0             # Domain/doc authority (0-1)
    recency_score: float = 0.0               # Temporal relevance (0-1)
    combined_score: float = 0.0              # Aggregated ranking score
    chunk_index: int | None = None           # Position in source document
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Structured metric / geographic / temporal fields ──
    metric_type: MetricType = MetricType.UNKNOWN     # GDP, GSDP, GVA, etc.
    metric_value: str = ""                           # Extracted numeric value as string
    geographic_scope: GeographicScope = GeographicScope.UNKNOWN  # National / state / district
    geography: str = ""                              # Specific place name (e.g. "Karnataka")
    year_period: str = ""                            # e.g. "2022-23", "FY2023"
    temporal_qualifier: TemporalQualifier = TemporalQualifier.UNKNOWN  # Actual / estimate / projected
    price_basis: PriceBasis = PriceBasis.UNKNOWN          # Current (nominal) vs Constant (real)

    model_config = {"extra": "ignore"}

    def to_citation(self) -> str:
        """Short citation string for inclusion in generated answers."""
        parts = []
        if self.source_type == SourceType.WEB and self.source_url:
            parts.append(f"[{self.source_name or self.source_url}]")
        elif self.source_name:
            parts.append(f"[{self.source_name}]")
        else:
            parts.append(f"[{self.evidence_id}]")
        # Append metric and geographic scope for disambiguation
        if self.metric_type != MetricType.UNKNOWN:
            parts.append(self.metric_type.value.upper())
        if self.price_basis != PriceBasis.UNKNOWN:
            parts.append(f"{self.price_basis.value}-price")
        if self.geography:
            parts.append(self.geography)
        if self.year_period:
            parts.append(self.year_period)
        if self.temporal_qualifier != TemporalQualifier.UNKNOWN:
            parts.append(f"({self.temporal_qualifier.value})")
        return " ".join(parts)


class Claim(BaseModel):
    """An atomic factual statement extracted from the answer."""

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    claim_type: ClaimType = ClaimType.FACT          # fact / inference / speculation
    evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""                     # Why this status was assigned
    repair_action: str = ""                 # e.g. "search_web", "reject", "rephrase"


class QueryClassification(BaseModel):
    """Structured understanding of the user query."""

    primary_need: QueryNeed = QueryNeed.UNKNOWN
    needs_documents: bool = True            # Should we search the knowledge base?
    needs_web: bool = False                 # Should we search the public web?
    needs_calculation: bool = False
    temporal_focus: str | None = None       # ISO date string or "latest"
    temporal_qualifier: TemporalQualifier = TemporalQualifier.UNKNOWN  # estimate / actual / projected
    geographic_scope: GeographicScope = GeographicScope.UNKNOWN        # national / state / city
    geography: str = ""                      # Specific place name (e.g. "Karnataka")
    domain_hints: list[str] = Field(default_factory=list)
    ambiguity: str = "low"                  # low / medium / high
    rewrite: str = ""                       # Disambiguated / expanded query
    metric_hint: MetricType = MetricType.UNKNOWN  # If query is about a specific metric

    @field_validator("geography", "rewrite", "ambiguity", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        """OpenRouter models often emit null instead of \"\" for optional strings."""
        return "" if v is None else v


class PlanStep(BaseModel):
    """One step in a retrieval/verification plan."""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str                             # retrieve_documents, search_web, calculate, synthesize
    queries: list[str] = Field(default_factory=list)
    expected_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


class PlannerOutput(BaseModel):
    """Structured plan produced by the planner."""

    classification: QueryClassification
    steps: list[PlanStep]
    fallback_strategy: str = "answer_with_caveats"


class RepairOutput(BaseModel):
    """Structured repair instructions for failed claims."""

    decision: RepairDecision
    failed_claims: list[Claim] = Field(default_factory=list)
    new_steps: list[PlanStep] = Field(default_factory=list)
    explanation: str = ""


class EvidenceState(BaseModel):
    """Structured cross-turn evidence memory.

    Replaces the previous ad-hoc 'prepend prior summary to query' hack. Evidence is
    carried forward as typed records so later turns can build on established facts while
    still re-verifying, superseding, or flagging conflicts. It deliberately does NOT store
    raw conversation text — only structured, provenance-preserving evidence.
    """

    turn: int = 0                                 # conversation turn this state represents
    established: list[Evidence] = Field(default_factory=list)   # verified facts from prior turns
    inferences: list[Evidence] = Field(default_factory=list)    # inference-level evidence
    superseded: list[Evidence] = Field(default_factory=list)   # older evidence replaced by newer
    conflicts: list[dict] = Field(default_factory=list)        # structured conflict records
    unresolved: list[str] = Field(default_factory=list)        # claims that could not be resolved

    model_config = {"extra": "ignore"}

    def all_evidence(self) -> list[Evidence]:
        """All non-superseded evidence carried forward."""
        return list(self.established) + list(self.inferences)

    def is_empty(self) -> bool:
        return not (self.established or self.inferences or self.conflicts or self.unresolved)


class CitationUsage(BaseModel):
    """Tracks which citations were actually used by the generator."""

    claim_id: str
    evidence_ids: list[str]


# ── LangGraph state ──────────────────────────────────────────────────────────


def _add_to_list(existing: list | None, new: list | None) -> list:
    """Reduce function: append lists for Annotated state fields."""
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


def _keep_latest(_existing: Any, new: Any) -> Any:
    """Reduce function: overwrite with latest value."""
    return new


class RAGState(TypedDict, total=False):
    """LangGraph state for the business-ready RAG agent.

    Backward-compatible fields (``chunks``, ``search``) are retained as plain strings
    for callers that still pass them, but the canonical structured data lives in
    ``evidence`` and ``claims``.
    """

    # ── identity / request ──
    user_id: uuid.UUID
    chat_id: uuid.UUID
    query: str
    provider: str
    user_credentials: dict  # provider → {api_key, fallback, models}; never log
    messages: Annotated[list[dict], _add_to_list]

    # ── control / guard counters ──
    graph_steps: int
    search_count: int
    retrieval_count: int
    regeneration_count: int
    repair_pass_count: int
    max_graph_steps: int
    max_searches: int
    max_retrievals: int
    max_regenerations: int

    # ── structured agent memory ──
    classification: Annotated[QueryClassification | None, _keep_latest]
    plan: Annotated[PlannerOutput | None, _keep_latest]
    evidence: Annotated[list[Evidence], _keep_latest]
    claims: Annotated[list[Claim], _keep_latest]
    conflicts: Annotated[list[dict], _add_to_list]
    citation_usage: Annotated[list[CitationUsage], _add_to_list]
    assembled_context: Annotated[str, _keep_latest]
    cite_map: Annotated[dict[str, str], _keep_latest]  # E1 → evidence_id
    coverage_gaps: Annotated[list[str], _keep_latest]
    repair_mode: Annotated[str, _keep_latest]  # "" | "surgical"
    evidence_state: Annotated[EvidenceState | None, _keep_latest]  # persistent cross-turn state
    prior_evidence_state: Annotated[EvidenceState | None, _keep_latest]  # loaded from DB at entry
    verification_errors: Annotated[list[dict], _keep_latest]  # structured verifier errors

    # ── outputs ──
    answer: Annotated[str, _keep_latest]
    final_status: Annotated[str, _keep_latest]
    error: Annotated[str | None, _keep_latest]

    # ── backward-compatible flat-string buffers ──
    # Deprecated: prefer structured ``evidence`` for new logic.
    chunks: Annotated[list[str], _add_to_list]
    search: Annotated[list[str], _add_to_list]

    # ── backward-compatible planner legacy ──
    # Deprecated: prefer ``plan`` and ``classification``.
    planner_state: Annotated[str, _keep_latest]
    retrieval_queries: Annotated[list[str], _add_to_list]
    wiki_queries: Annotated[list[str], _add_to_list]
    tavily_queries: Annotated[list[str], _add_to_list]
    searxng_queries: Annotated[list[str], _add_to_list]
    repair_state: Annotated[str, _keep_latest]

    # ── legacy outputs / counters used by the router / tests ──
    provider_used: Annotated[str, _keep_latest]
    need_repair: Annotated[str, _keep_latest]
    hallucination_reason: Annotated[list[str], _add_to_list]
    max_tries_planner: Annotated[int, _keep_latest]
    max_tries_hallucinator: Annotated[int, _keep_latest]
    steps_taken: Annotated[int, _keep_latest]
    searches_done: Annotated[int, _keep_latest]
    retrievals_done: Annotated[int, _keep_latest]
    regenerations_done: Annotated[int, _keep_latest]
    cross_chat_enabled: Annotated[bool, _keep_latest]


# ── Helpers ──────────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def evidence_by_id(evidence: list[Evidence], evidence_id: str) -> Evidence | None:
    for ev in evidence:
        if ev.evidence_id == evidence_id:
            return ev
    return None
