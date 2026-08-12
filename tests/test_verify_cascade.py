"""Unit tests for verify cascade: mechanical, numeric, short-circuit."""

from __future__ import annotations

from unittest.mock import patch

from app.agent.state import ClaimStatus, Evidence, SourceType
from app.agent.verify_cascade import (
    dedupe_search_queries,
    extract_numbers,
    grade_coverage,
    mechanical_check_claim,
    numeric_check_claim,
    numeric_compare,
    run_verify_cascade,
)


def _ev(eid: str, text: str, cite_key: str | None = None) -> Evidence:
    meta = {"cite_key": cite_key} if cite_key else {}
    return Evidence(
        evidence_id=eid,
        text=text,
        source_type=SourceType.WEB,
        source_name="test",
        metadata=meta,
        combined_score=0.8,
        rerank_score=0.8,
    )


class TestDedupeAndCoverage:
    def test_dedupe_near_duplicate_queries(self):
        qs = [
            "Maharashtra GSDP services share 2024-25",
            "maharashtra gsdp services share 2024-25 official",
            "completely different topic GDP China",
        ]
        out = dedupe_search_queries(qs, max_keep=2)
        assert len(out) == 2
        assert "China" in out[1]

    def test_grade_coverage_empty(self):
        gaps = grade_coverage("query", [])
        assert gaps


class TestMechanical:
    def test_valid_e_key_with_overlap_passes(self):
        ev = _ev(
            "a1b2c3d4",
            "Maharashtra services contributed 54.5 percent of GSDP in 2024-25.",
            "E1",
        )
        claim = mechanical_check_claim(
            "Maharashtra services contributed 54.5% of GSDP in 2024-25 [E1].",
            [ev],
            {"E1": "a1b2c3d4"},
        )
        assert claim is not None
        assert claim.status == ClaimStatus.VERIFIED
        assert claim.evidence_ids == ["a1b2c3d4"]

    def test_invalid_e_key_fails(self):
        ev = _ev("a1b2c3d4", "Some evidence about GSDP services share.", "E1")
        claim = mechanical_check_claim(
            "Maharashtra services were 54.5% of GSDP [E9].",
            [ev],
            {"E1": "a1b2c3d4"},
        )
        assert claim is not None
        assert claim.status == ClaimStatus.UNVERIFIED

    def test_weak_overlap_fails(self):
        ev = _ev(
            "a1b2c3d4",
            "The monsoon arrived early across coastal districts this year.",
            "E1",
        )
        claim = mechanical_check_claim(
            "Maharashtra services contributed 54.5% of GSDP in 2024-25 [E1].",
            [ev],
            {"E1": "a1b2c3d4"},
            min_overlap=0.25,
        )
        assert claim is not None
        assert claim.status == ClaimStatus.UNVERIFIED


class TestNumeric:
    def test_extract_percent(self):
        nums = extract_numbers("share was 54.5% of GSDP")
        assert any(abs(v - 54.5) < 1e-6 and u == "%" for v, u in nums)

    def test_percent_vs_bare_match(self):
        assert numeric_compare("share was 54.5% [E1]", "services were 54.5 of GSDP") == "pass"

    def test_number_mismatch_contradicts(self):
        assert numeric_compare("share was 54.5% [E1]", "services were 45.5% of GSDP") == "contradict"

    def test_fy_match(self):
        assert (
            numeric_compare(
                "In FY 2024-25 the share was reported [E1]",
                "For 2024-25 Maharashtra DES reported figures",
            )
            in ("pass", "skip")
        )

    def test_numeric_check_claim_contradict(self):
        ev = _ev("a1b2c3d4", "Services share of GSDP was 45.5 percent in 2024-25.", "E1")
        claim = numeric_check_claim(
            "Maharashtra services contributed 54.5% of GSDP in 2024-25 [E1].",
            [ev],
            {"E1": "a1b2c3d4"},
        )
        assert claim is not None
        assert claim.status == ClaimStatus.CONTRADICTED

    def test_numeric_check_claim_pass(self):
        ev = _ev("a1b2c3d4", "Services share of GSDP was 54.5 percent in 2024-25.", "E1")
        claim = numeric_check_claim(
            "Maharashtra services contributed 54.5% of GSDP in 2024-25 [E1].",
            [ev],
            {"E1": "a1b2c3d4"},
        )
        assert claim is not None
        assert claim.status == ClaimStatus.VERIFIED


    def test_invented_magnitude_contradicts_even_with_overlap(self):
        """A number that does not appear in cited evidence fails closed."""
        ev = _ev(
            "a1b2c3d4",
            "The tertiary sector is the largest contributor to the economy according to the survey.",
            "E1",
        )
        claim = numeric_check_claim(
            "The tertiary sector accounts for 394225000 lakh of output [E1].",
            [ev],
            {"E1": "a1b2c3d4"},
        )
        assert claim is not None
        assert claim.status == ClaimStatus.CONTRADICTED


class TestCascadeShortCircuit:
    def test_llm_not_called_when_mechanical_and_numeric_resolve(self):
        ev = _ev(
            "a1b2c3d4",
            "Maharashtra services contributed 54.5 percent of GSDP in FY 2024-25 according to DES.",
            "E1",
        )
        answer = (
            "### Direct Answer\n"
            "Maharashtra services contributed 54.5% of GSDP in 2024-25 [E1].\n\n"
            "### Supporting Evidence\n"
            "- **Fact** [E1]: Maharashtra services contributed 54.5 percent of GSDP in FY 2024-25 according to DES.\n"
        )
        called = {"n": 0}

        def boom(sents, evidence):
            called["n"] += 1
            raise AssertionError("LLM residual should not be invoked")

        with patch("app.agent.verify_cascade._get_nli_pipeline", return_value=None):
            result = run_verify_cascade(
                answer,
                [ev],
                cite_map={"E1": "a1b2c3d4"},
                llm_invoke=boom,
            )

        assert called["n"] == 0
        assert result.stats.escalated == 0
        assert result.stats.mechanical + result.stats.numeric >= 1
        assert all(c.status == ClaimStatus.VERIFIED for c in result.claims)

    def test_empty_llm_residual_marks_uncertain_not_pass(self):
        # Mid token overlap so mechanical escalates; empty LLM residual → UNCERTAIN
        answer = "Maharashtra GSDP services share reached unprecedented levels recently [E1]."
        ev = _ev(
            "a1b2c3d4",
            "Maharashtra published an indicators bulletin for districts.",
            "E1",
        )

        def empty(sents, evidence):
            return []

        with patch("app.agent.verify_cascade._get_nli_pipeline", return_value=None):
            result = run_verify_cascade(
                answer,
                [ev],
                cite_map={"E1": "a1b2c3d4"},
                llm_invoke=empty,
            )

        assert result.stats.escalated >= 1
        assert any(c.status == ClaimStatus.UNCERTAIN for c in result.claims)
        assert not any(c.status == ClaimStatus.VERIFIED for c in result.claims)

    def test_cascade_numeric_overrides_mechanical_invented_number(self):
        ev = _ev(
            "a1b2c3d4",
            "The tertiary sector is the largest contributor according to the survey of the economy.",
            "E1",
        )
        answer = (
            "The tertiary sector is the largest contributor totaling 394225000 lakh [E1]."
        )
        with patch("app.agent.verify_cascade._get_nli_pipeline", return_value=None):
            result = run_verify_cascade(
                answer,
                [ev],
                cite_map={"E1": "a1b2c3d4"},
                llm_invoke=lambda s, e: (_ for _ in ()).throw(AssertionError("no llm")),
            )
        assert any(c.status == ClaimStatus.CONTRADICTED for c in result.claims)
        assert result.stats.numeric >= 1
