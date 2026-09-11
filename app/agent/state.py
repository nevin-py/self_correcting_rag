"""Typed state and schemas for the lean self-correcting RAG agent.

Design principles
-----------------
1. One LLM call per job: classify, generate, verify. No heuristic stand-ins.
2. Provenance first: every piece of evidence knows where it came from.
3. Domain-agnostic: no keyword lists, no magic thresholds, no hardcoded places.
4. Chat-aware: conversation history and cross-turn evidence memory are
   first-class inputs to every LLM call.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ── Enums ────────────────────────────────────────────────────────────────────


class SourceType(str, enum.Enum):
    DOCUMENT = "document"      # User-uploaded / internal knowledge base
    WEB = "web"                # Public web search
    UNKNOWN = "unknown"


class ClaimStatus(str, enum.Enum):
    VERIFIED = "verified"              # Supported by cited evidence
    CONTRADICTED = "contradicted"      # Evidence says otherwise
    UNVERIFIED = "unverified"          # No supporting evidence found
    UNCERTAIN = "uncertain"            # Conflicting or insufficient evidence


class QueryMode(str, enum.Enum):
    """What the agent should do with this turn."""
    RESEARCH = "research"                  # Retrieve/search, then answer with citations
    CONVERSATIONAL = "conversational"      # Small talk / meta — reply directly
    CLARIFICATION = "clarification"        # Ambiguous — ask the user which they mean


# ── Core records ─────────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """A single piece of retrieved or searched evidence."""

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    source_type: SourceType = SourceType.UNKNOWN
    source_name: str = ""                    # Document name or site name
    source_url: str | None = None
    source_date: datetime | None = None      # Publication / retrieval date
    retrieval_score: float = 0.0             # Vector/BM25/search score (0-1)
    rerank_score: float | None = None        # Cross-encoder rerank score
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class Claim(BaseModel):
    """An atomic factual statement from the answer, with its verdict."""

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""                      # Why this status was assigned


class QueryUnderstanding(BaseModel):
    """Structured output of the classify_and_plan node (single LLM call)."""

    mode: QueryMode = QueryMode.RESEARCH
    rewritten_query: str = ""                # Standalone, context-bound query for retrieval
    needs_documents: bool = True             # Search the user's knowledge base?
    needs_web: bool = False                  # Search the public web?
    search_queries: list[str] = Field(default_factory=list)  # 1-3 targeted queries
    temporal_focus: str = ""                 # e.g. "latest", "2023", "Q1 2024"; "" if unspecified
    geography: str = ""                      # Place the question is about; "" if none
    clarification_question: str = ""

    model_config = {"extra": "ignore"}

    @field_validator("rewritten_query", "temporal_focus", "geography", "clarification_question", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        """Models often emit null instead of "" for optional strings."""
        return "" if v is None else v


class Verdict(BaseModel):
    """Structured output of the verify_answer node (single LLM judge call)."""

    claims: list[Claim] = Field(default_factory=list)
    overall: Literal["supported", "partial", "unsupported"] = "supported"
    repair_queries: list[str] = Field(default_factory=list)  # Searches that would fix gaps
    explanation: str = ""
    # When contradictions stem from an ambiguous term (different people,
    # organizations, or expansions of an acronym), the judge writes ONE
    # question listing the interpretations so the agent can ask instead of guess.
    clarification_question: str = ""
    # Relevance-to-question verdict (Self-RAG style): does the answer address
    # what was actually asked? A fluent, fully-cited answer to the wrong
    # question must not pass as "supported".
    addresses_question: bool = True
    question_gaps: list[str] = Field(default_factory=list)  # Aspects of the question left unanswered
    # Entity triage: the query names an entity but evidence shows several
    # distinct real-world referents sharing that name (people, orgs, acronyms).
    referent_ambiguity: bool = False
    referents: list[str] = Field(default_factory=list)  # Short identities found in evidence

    model_config = {"extra": "ignore"}

    @field_validator("clarification_question", "explanation", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        return "" if v is None else v


class Computation(BaseModel):
    """One planned arithmetic step, executed by the sandboxed evaluator."""

    label: str = ""
    expression: str = ""

    model_config = {"extra": "ignore"}

    @field_validator("label", "expression", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v: Any) -> Any:
        return "" if v is None else v


class ComputationPlan(BaseModel):
    """Structured output of the compute-planner call (pure arithmetic only)."""

    computations: list[Computation] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class EvidenceState(BaseModel):
    """Structured cross-turn evidence memory.

    Carries verified facts forward so later turns can build on established
    results while still re-verifying against fresh evidence. Stores typed,
    provenance-preserving records — never raw conversation text.
    """

    turn: int = 0
    established: list[Evidence] = Field(default_factory=list)   # Verified facts from prior turns
    unresolved: list[str] = Field(default_factory=list)         # Claims that could not be resolved
    last_final_status: str = ""                                 # needs_clarification etc.
    model_config = {"extra": "ignore"}

    def all_evidence(self) -> list[Evidence]:
        return list(self.established)

    def is_empty(self) -> bool:
        return not (self.established or self.unresolved)


# ── LangGraph state ──────────────────────────────────────────────────────────


def _add_to_list(existing: list | None, new: list | None) -> list:
    return (existing or []) + (new or [])


def _keep_latest(_existing: Any, new: Any) -> Any:
    return new


class RAGState(TypedDict, total=False):
    """LangGraph state for the lean RAG agent."""

    # ── identity / request ──
    user_id: uuid.UUID
    chat_id: uuid.UUID
    query: str                                # Context-bound query (after classify_and_plan)
    query_original: str                       # Raw user message, never mutated
    provider: str
    user_credentials: dict                    # provider → {api_key, fallback, models}; never log
    messages: Annotated[list, _keep_latest]   # Conversation history (LangChain messages)
    request_context: dict                     # {timezone, location, device} from the client
    document_inventory: list                  # Filenames of docs ingested into this chat

    # ── routing (set by classify_and_plan) ──
    understanding: Annotated[QueryUnderstanding | None, _keep_latest]

    # ── working memory ──
    evidence: Annotated[list[Evidence], _keep_latest]
    cite_map: Annotated[dict[str, str], _keep_latest]     # E1 → evidence_id
    assembled_context: Annotated[str, _keep_latest]
    claims: Annotated[list[Claim], _keep_latest]
    verification_errors: Annotated[list[dict], _keep_latest]
    repair_queries: Annotated[list[str], _keep_latest]
    repair_count: Annotated[int, _keep_latest]
    prior_evidence_state: Annotated[EvidenceState | None, _keep_latest]
    # Deterministic calculator: planned expressions evaluated by the sandboxed
    # evaluator — authoritative numeric ground truth for generate + judge.
    computations: Annotated[list[dict], _keep_latest]
    # Self-correction loop (Reflexion-style): the judge's serialized critique
    # and the previous draft are fed back into the regeneration pass so the
    # repair actually fixes the flagged problems instead of repeating them.
    critique: Annotated[str, _keep_latest]
    draft_answer: Annotated[str, _keep_latest]
    revise_requested: Annotated[bool, _keep_latest]       # Revise-only pass (no new search)
    history_summary: Annotated[str, _keep_latest]         # Compaction of older messages

    # ── guard counters ──
    graph_steps: int
    search_count: int
    retrieval_count: int

    # ── outputs ──
    answer: Annotated[str, _keep_latest]
    final_status: Annotated[str, _keep_latest]            # answered | answered_with_caveats | needs_clarification | conversational
    provider_used: Annotated[str, _keep_latest]
    error: Annotated[str | None, _keep_latest]


# ── Helpers ──────────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(timezone.utc)



