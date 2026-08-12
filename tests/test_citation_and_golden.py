"""Tests for hard citation validation and golden eval checks."""

from app.agent.citation_validator import flag_uncited_in_answer, validate_answer_citations
from app.agent.eval_checks import load_golden_set, score_case
from app.agent.state import Evidence, SourceType


def _ev(eid: str, text: str = "Maharashtra services 64.27% of GSDP") -> Evidence:
    return Evidence(
        evidence_id=eid,
        text=text,
        source_type=SourceType.WEB,
        source_name="DES",
    )


class TestCitationValidator:
    def test_uncited_factual_flagged(self):
        result = validate_answer_citations(
            "Maharashtra services contributed 64.27% of GSDP in 2024-25.",
            [_ev("a1b2c3d4")],
        )
        assert not result.ok
        assert result.uncited_sentences
        assert any(e.issue == "UNCITED_ASSERTION" for e in result.errors)

    def test_valid_citation_passes(self):
        result = validate_answer_citations(
            "Maharashtra services contributed 64.27% of GSDP in 2024-25 [a1b2c3d4].",
            [_ev("a1b2c3d4")],
        )
        assert result.ok
        assert not result.uncited_sentences
        assert result.cited_ids == ["a1b2c3d4"]

    def test_valid_e_key_citation(self):
        ev = _ev("a1b2c3d4")
        ev.metadata["cite_key"] = "E1"
        result = validate_answer_citations(
            "Maharashtra services contributed 64.27% of GSDP in 2024-25 [E1].",
            [ev],
            cite_map={"E1": "a1b2c3d4"},
        )
        assert result.ok
        assert result.cited_ids == ["a1b2c3d4"]

    def test_analysis_section_skipped(self):
        answer = (
            "### Direct Answer\n"
            "Services were 64% of GSDP [a1b2c3d4].\n\n"
            "### Analysis & Caveats\n"
            "- **Confidence**: Medium based on single source without triangulation.\n"
            "- Maharashtra economy grew faster than national GDP without any citation here.\n"
        )
        result = validate_answer_citations(answer, [_ev("a1b2c3d4")])
        assert result.ok

    def test_fact_label_cite_counts_for_quote(self):
        ev = _ev("a1b2c3d4")
        ev.metadata["cite_key"] = "E1"
        result = validate_answer_citations(
            '- **Fact 1 (Service Sector) [E1]**: "Services were 54.5% of GSDP in 2017-18."',
            [ev],
            cite_map={"E1": "a1b2c3d4"},
        )
        assert result.ok
        assert not result.uncited_sentences

    def test_invalid_citation_id(self):
        result = validate_answer_citations(
            "Services were 64.27% of GSDP [deadbeef].",
            [_ev("a1b2c3d4")],
        )
        assert not result.ok
        assert "deadbeef" in result.invalid_citation_ids

    def test_flag_appends_note(self):
        result = validate_answer_citations(
            "Maharashtra GSDP services share was 64%.",
            [_ev("a1b2c3d4")],
        )
        flagged = flag_uncited_in_answer("Maharashtra GSDP services share was 64%.", result)
        assert "Citation check:" in flagged


class TestGoldenEval:
    def test_golden_set_loads(self):
        cases = load_golden_set()
        assert len(cases) >= 15
        assert all("id" in c and "query" in c for c in cases)

    def test_good_answers_pass(self):
        cases = {c["id"]: c for c in load_golden_set()}
        for case_id, case in cases.items():
            good = case.get("good_answer_example")
            if not good:
                continue
            score = score_case(case, good)
            assert score.passed, f"{case_id} failed: {[c.name + ':' + c.detail for c in score.failed]}"

    def test_bad_uncited_fails(self):
        case = next(c for c in load_golden_set() if c["id"] == "uncited-should-fail")
        score = score_case(case, case["bad_answer_example"])
        assert not score.passed
        assert any(c.name == "citation_uncited_budget" for c in score.failed)

    def test_bad_invalid_citation_fails(self):
        case = next(c for c in load_golden_set() if c["id"] == "invalid-citation-should-fail")
        score = score_case(case, case["bad_answer_example"])
        assert not score.passed
        assert any(c.name == "citation_ids_resolve" for c in score.failed)

    def test_metric_trap_prefers_gsdp(self):
        case = next(c for c in load_golden_set() if c["id"] == "mh-gsdp-vs-gdp-trap")
        score = score_case(case, case["good_answer_example"])
        assert score.passed
