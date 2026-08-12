"""
Layer 5: Graph correctness tests (unit-level, mocked LLM).

Covers: deterministic routing, execution guards, fallback branches,
        enum validation, structured state, claim-level verification.

These tests mock the LLM calls so they run without API keys.
"""

import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.agent.state import (
    Claim,
    ClaimStatus,
    Evidence,
    PlannerOutput,
    PlanStep,
    QueryClassification,
    QueryNeed,
    RepairDecision,
    RepairOutput,
    SourceType,
)
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
    _detect_contradiction,
    _detect_evidence_conflicts,
)
from app.core.config import settings


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    """Build a minimal valid RAGState with sane defaults."""
    state = {
        "user_id": uuid.uuid4(),
        "chat_id": uuid.uuid4(),
        "query": "test query",
        "provider": "auto",
        "messages": [],
        "chunks": [],
        "search": [],
        "evidence": [],
        "claims": [],
        "conflicts": [],
        "citation_usage": [],
        "classification": None,
        "plan": None,
        "answer": "",
        "final_status": "",
        "assembled_context": "",
        "graph_steps": 0,
        "search_count": 0,
        "retrieval_count": 0,
        "regeneration_count": 0,
        "max_graph_steps": settings.MAX_GRAPH_STEPS,
        "max_searches": settings.MAX_SEARCHES,
        "max_retrievals": settings.MAX_RETRIEVALS,
        "max_regenerations": settings.MAX_REGENERATIONS,
    }
    state.update(overrides)
    return state


def _evidence(text: str, source_type: SourceType = SourceType.DOCUMENT, **kwargs) -> Evidence:
    return Evidence(text=text, source_type=source_type, **kwargs)


# ── Query classification router / node ───────────────────────────────────────

class TestClassification:
    """Deterministic classification and source routing."""

    def test_should_retrieve_when_documents_needed(self):
        state = _base_state(
            classification=QueryClassification(needs_documents=True, needs_web=False)
        )
        assert should_retrieve_documents(state) == "retrieve_documents"

    def test_should_skip_retrieve_when_documents_not_needed(self):
        state = _base_state(
            classification=QueryClassification(needs_documents=False, needs_web=True)
        )
        assert should_retrieve_documents(state) == "assemble"

    def test_should_search_web_when_web_needed(self):
        state = _base_state(
            classification=QueryClassification(needs_documents=False, needs_web=True)
        )
        assert should_search_web(state) == "search_web"

    def test_should_skip_web_when_web_not_needed(self):
        state = _base_state(
            classification=QueryClassification(needs_documents=True, needs_web=False)
        )
        assert should_search_web(state) == "assemble"

    @pytest.mark.asyncio
    async def test_classify_query_returns_classification(self):
        mock_response = QueryClassification(
            primary_need=QueryNeed.FACTUAL,
            needs_documents=True,
            needs_web=False,
            rewrite="what is the test query",
        )
        with patch("app.agent.nodes.openrouter_planner_llm", None):
            with patch("app.agent.nodes.routing_llm") as mock_llm:
                mock_llm.with_structured_output = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_response)))
                result = classify_query(_base_state(query="test query"))

        assert result["classification"].primary_need == QueryNeed.FACTUAL
        assert result["classification"].needs_documents is True

    @pytest.mark.asyncio
    async def test_classify_query_fallback_on_failure(self):
        with patch("app.agent.nodes.routing_llm") as mock_llm:
            mock_llm.with_structured_output = MagicMock(return_value=MagicMock(invoke=MagicMock(side_effect=ValueError("bad"))))
            with patch("app.agent.nodes.openrouter_planner_llm", None):
                result = classify_query(_base_state(query="test query"))
        assert result["classification"].rewrite == "test query"


# ── Planner node ─────────────────────────────────────────────────────────────

class TestPlanner:
    """Structured planner node."""

    def test_build_plan_returns_steps(self):
        classification = QueryClassification(needs_documents=True, needs_web=False, rewrite="test query")
        mock_response = PlannerOutput(
            classification=classification,
            steps=[
                PlanStep(action="retrieve_documents", queries=["test query"], expected_claims=["answer"], rationale="retrieve"),
            ],
        )
        with patch("app.agent.nodes.routing_llm") as mock_llm:
            mock_llm.with_structured_output = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=mock_response)))
            result = build_plan(_base_state(query="test query", classification=classification))

        assert result["plan"].steps[0].action == "retrieve_documents"

    def test_build_plan_fallback_on_failure(self):
        classification = QueryClassification(needs_documents=True, rewrite="test query")
        with patch("app.agent.nodes.routing_llm") as mock_llm:
            mock_llm.with_structured_output = MagicMock(return_value=MagicMock(invoke=MagicMock(side_effect=ValueError("bad"))))
            with patch("app.agent.nodes.openrouter_planner_llm", None):
                result = build_plan(_base_state(query="test query", classification=classification))
        assert result["plan"].steps[0].action == "retrieve_documents"


# ── Hallucination / verification router ──────────────────────────────────────

class TestHallucinationRouter:
    """Deterministic routing for claim-level verification."""

    def test_no_failed_claims_routes_to_satisfactory(self):
        state = _base_state(claims=[Claim(text="ok", status=ClaimStatus.VERIFIED)])
        assert hallucination_router(state) == "satisfactory"

    def test_unverified_claim_routes_to_repair(self):
        state = _base_state(claims=[Claim(text="bad", status=ClaimStatus.UNVERIFIED)])
        assert hallucination_router(state) == "repair"

    def test_contradicted_claim_routes_to_repair(self):
        state = _base_state(claims=[Claim(text="bad", status=ClaimStatus.CONTRADICTED)])
        assert hallucination_router(state) == "repair"

    def test_max_regenerations_forces_max_attempts(self):
        state = _base_state(
            claims=[Claim(text="bad", status=ClaimStatus.UNVERIFIED)],
            regeneration_count=settings.MAX_REGENERATIONS,
        )
        assert hallucination_router(state) == "max_attempts"

    def test_falls_back_to_satisfactory_with_no_claims(self):
        state = _base_state(claims=[])
        assert hallucination_router(state) == "satisfactory"


# ── Repair node ──────────────────────────────────────────────────────────────

class TestRepair:
    """Repair node behavior."""

    def test_no_failed_claims_returns_satisfactory(self):
        state = _base_state(claims=[Claim(text="ok", status=ClaimStatus.VERIFIED)])
        result = repair_claims(state)
        assert result["repair_state"] == RepairDecision.SATISFACTORY.value

    def test_failed_claim_returns_repair_plan(self):
        state = _base_state(
            claims=[Claim(text="bad", status=ClaimStatus.UNVERIFIED, repair_action="search_web")],
            classification=QueryClassification(),
        )
        result = repair_claims(state)
        assert result["repair_state"] == RepairDecision.REPAIR.value
        assert result["plan"].steps[0].action == "search_web"

    def test_max_regenerations_returns_max_attempts(self):
        state = _base_state(
            claims=[Claim(text="bad", status=ClaimStatus.UNVERIFIED)],
            regeneration_count=settings.MAX_REGENERATIONS,
            classification=QueryClassification(),
        )
        result = repair_claims(state)
        assert result["repair_state"] == RepairDecision.MAX_ATTEMPTS.value


# ── Deterministic contradiction detection ────────────────────────────────────

class TestContradictionDetection:
    """Deterministic conflict detection heuristics."""

    def test_negation_contradiction(self):
        is_contra, reason = _detect_contradiction("The product launched", "The product did not launch")
        assert is_contra is True
        assert "negation" in reason

    def test_antonym_contradiction(self):
        is_contra, reason = _detect_contradiction("Profits increased", "Profits decreased")
        assert is_contra is True
        assert "antonym" in reason

    def test_numeric_contradiction(self):
        is_contra, reason = _detect_contradiction("Revenue was 100 million", "Revenue was 200 million")
        assert is_contra is True
        assert "numeric" in reason

    def test_no_contradiction_unrelated(self):
        is_contra, _ = _detect_contradiction("The sky is blue", "Stocks rose today")
        assert is_contra is False

    def test_evidence_conflict_detection(self):
        ev_a = _evidence("The company hired 500 people", SourceType.WEB, source_name="Reuters")
        ev_b = _evidence("The company fired 500 people", SourceType.WEB, source_name="Blog")
        conflicts = _detect_evidence_conflicts([ev_a, ev_b])
        assert len(conflicts) == 1


# ── Evidence assembly ────────────────────────────────────────────────────────

class TestEvidenceAssembly:
    """Evidence scoring, deduplication, and conflict handling."""

    def test_deduplicates_near_duplicate_evidence(self):
        ev1 = _evidence("The company reported earnings.")
        ev2 = _evidence("The company reported earnings.")
        result = assemble_evidence(_base_state(evidence=[ev1, ev2]))
        assert len(result["evidence"]) == 1

    def test_detects_and_penalizes_conflicts(self):
        ev_a = _evidence("Revenue increased", SourceType.WEB, source_name="Reuters", authority_score=0.9)
        ev_b = _evidence("Revenue decreased", SourceType.WEB, source_name="Blog", authority_score=0.4)
        result = assemble_evidence(_base_state(evidence=[ev_a, ev_b]))
        assert len(result["conflicts"]) == 1
        # Higher authority source should rank first
        assert result["evidence"][0].source_name == "Reuters"


# ── Graph compilation ────────────────────────────────────────────────────────

class TestGraphCompilation:
    """Verify the graph compiles and has the expected structure."""

    def test_graph_compiles(self):
        from app.agent.graph import rag_app
        assert rag_app is not None

    def test_graph_has_expected_nodes(self):
        from app.agent.graph import rag_app
        node_names = set(rag_app.get_graph().nodes.keys())
        expected = {
            "classify_query",
            "build_plan",
            "retrieve_documents",
            "search_web",
            "assemble_evidence",
            "extract_verify_claims",
            "generate_answer",
            "verify_answer_claims",
            "repair_claims",
        }
        assert expected.issubset(node_names)
