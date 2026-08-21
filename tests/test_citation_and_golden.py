"""Tests for the hard citation validator."""

import pytest

from app.agent.citation_validator import (
    flag_uncited_in_answer,
    resolve_citation_token,
    split_checkable_sentences,
    validate_answer_citations,
)
from app.agent.state import Evidence, SourceType


def _ev(eid: str, text: str = "Renewable energy accounted for 30% of generation in 2024") -> Evidence:
    return Evidence(
        evidence_id=eid,
        text=text,
        source_type=SourceType.WEB,
        source_name="example.org",
    )


class TestCitationValidator:
    def test_uncited_factual_flagged(self):
        result = validate_answer_citations(
            "Renewable energy contributed 30% of total generation in 2024.",
            [_ev("a1b2c3d4")],
        )
        assert not result.ok
        assert result.uncited_sentences
        assert any(e.issue == "UNCITED_ASSERTION" for e in result.errors)

    def test_valid_citation_passes(self):
        result = validate_answer_citations(
            "Renewable energy contributed 30% of generation in 2024 [a1b2c3d4].",
            [_ev("a1b2c3d4")],
        )
        assert result.ok
        assert not result.uncited_sentences

    def test_invalid_citation_id_flagged(self):
        result = validate_answer_citations(
            "Solar grew 22% last year [e99].",
            [_ev("a1b2c3d4")],
        )
        assert "e99" in result.invalid_citation_ids
        assert any(e.issue == "INVALID_CITATION" for e in result.errors)

    def test_cite_key_resolution_via_map(self):
        ev = _ev("a1b2c3d4")
        ev.metadata["cite_key"] = "E1"
        result = validate_answer_citations(
            "Wind added 12 GW of capacity in 2024 [E1].",
            [ev],
            cite_map={"E1": "a1b2c3d4"},
        )
        assert result.ok
        assert result.cited_ids == ["a1b2c3d4"]

    def test_hedged_statements_do_not_need_citations(self):
        result = validate_answer_citations(
            "I could not find sufficient evidence to answer this question fully.",
            [],
        )
        assert result.ok
        assert not result.uncited_sentences

    def test_resolve_token_hex_and_ekey(self):
        ev = _ev("a1b2c3d4")
        ev.metadata["cite_key"] = "E2"
        assert resolve_citation_token("a1b2c3d4", [ev]) == "a1b2c3d4"
        assert resolve_citation_token("E2", [ev], {"E2": "a1b2c3d4"}) == "a1b2c3d4"
        assert resolve_citation_token("E9", [ev]) is None

    def test_split_skips_caveats_body(self):
        answer = (
            "## Direct Answer\n\nCapacity grew by 5 GW in 2024 [E1].\n\n"
            "Caveats:\n- This figure could not be cross-checked.\n"
        )
        sentences = split_checkable_sentences(answer)
        assert any("[E1]" in s for s in sentences)
        assert all("cross-checked" not in s for s in sentences)

    def test_flag_only_invalid_ids(self):
        answer = "Solar capacity reached 1 TW worldwide [e42]."
        result = validate_answer_citations(answer, [_ev("a1b2c3d4")])
        flagged = flag_uncited_in_answer(answer, result)
        assert "Citation check" in flagged
        # No invalid ids -> unchanged
        ok_result = validate_answer_citations(
            "Generation rose 4% in 2024 [a1b2c3d4].", [_ev("a1b2c3d4")]
        )
        assert flag_uncited_in_answer("Generation rose 4% in 2024 [a1b2c3d4].", ok_result) == (
            "Generation rose 4% in 2024 [a1b2c3d4]."
        )
