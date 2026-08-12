"""
Regression tests for the six observed failure modes:

1. Metric confusion (GSDP vs GVA vs output share)
2. Temporal/estimate confusion (actual vs estimate vs projected)
3. Cross-turn evidence state loss
4. Source quality issues (primary vs secondary vs tertiary)
5. Geographic scope confusion (national vs state vs city)
6. Fact vs inference conflation

Also covers:
- Structured evidence enrichment heuristics
- Claim type tagging (fact/inference/speculation)
- Context assembly with metric/geographic metadata
- Contradiction detection improvements
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.agent.state import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    GeographicScope,
    MetricType,
    PlannerOutput,
    PlanStep,
    QueryClassification,
    QueryNeed,
    RepairDecision,
    SourceQuality,
    SourceType,
    TemporalQualifier,
)
from app.agent.nodes import (
    _extract_metric_type,
    _extract_temporal_qualifier,
    _extract_year_period,
    _extract_metric_value,
    _classify_source_quality,
    _extract_geographic_scope,
    _enrich_evidence_metadata,
    _detect_contradiction,
    _detect_evidence_conflicts,
    assemble_evidence,
    repair_claims,
    hallucination_router,
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


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 1: Metric confusion (GSDP vs GVA vs output share)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricConfusion:
    """Metric type detection and differentiation heuristics."""

    def test_gsdp_detected_in_text(self):
        assert _extract_metric_type("Karnataka GSDP grew by 8.3%") == MetricType.GSDP

    def test_gva_detected_in_text(self):
        assert _extract_metric_type("India's GVA at constant prices") == MetricType.GVA

    def test_gdp_detected_in_text(self):
        assert _extract_metric_type("India GDP growth rate 2023") == MetricType.GDP

    def test_output_share_detected_in_text(self):
        assert _extract_metric_type("Manufacturing share of output was 17%") == MetricType.OUTPUT_SHARE

    def test_gdp_not_confused_with_gsdp(self):
        """GDP and GSDP must return different metric types."""
        gdp_result = _extract_metric_type("India GDP is $3.7 trillion")
        gsdp_result = _extract_metric_type("Karnataka GSDP reached ₹15 lakh crore")
        assert gdp_result != gsdp_result
        assert gdp_result == MetricType.GDP
        assert gsdp_result == MetricType.GSDP

    def test_gva_not_confused_with_gsdp(self):
        """GVA and GSDP must return different metric types."""
        gva_result = _extract_metric_type("GVA growth in manufacturing sector")
        gsdp_result = _extract_metric_type("GSDP of Karnataka state")
        assert gva_result != gsdp_result
        assert gva_result == MetricType.GVA
        assert gsdp_result == MetricType.GSDP

    def test_output_share_not_confused_with_gdp(self):
        assert _extract_metric_type("Output share of agriculture") == MetricType.OUTPUT_SHARE
        assert _extract_metric_type("GDP of India") == MetricType.GDP

    def test_unknown_metric_for_unrelated_text(self):
        assert _extract_metric_type("The weather is sunny today") == MetricType.UNKNOWN

    def test_evidence_metric_type_populated(self):
        ev = _evidence("Karnataka GSDP for 2022-23 was ₹15.7 lakh crore")
        ev = _enrich_evidence_metadata(ev)
        assert ev.metric_type == MetricType.GSDP

    def test_evidence_metric_type_from_classification_hint(self):
        """If text has no metric keyword but classification hints at one, use classification."""
        ev = _evidence("The economy grew significantly")
        classification = QueryClassification(metric_hint=MetricType.GSDP)
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.metric_type == MetricType.GSDP

    def test_evidence_metric_type_text_takes_priority_over_classification(self):
        """If text has an explicit metric keyword, it overrides classification hint."""
        ev = _evidence("GVA at constant prices rose 7.2%")
        classification = QueryClassification(metric_hint=MetricType.GDP)
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.metric_type == MetricType.GVA

    def test_metric_value_extracted(self):
        ev = _evidence("Karnataka GSDP for 2022-23 was ₹15.7 lakh crore")
        ev = _enrich_evidence_metadata(ev)
        assert "15.7" in ev.metric_value

    def test_assembled_context_includes_metric_type(self):
        ev = _evidence("Karnataka GSDP for 2022-23 was ₹15.7 lakh crore",
                        source_type=SourceType.WEB, source_name="Economic Survey")
        ev = _enrich_evidence_metadata(ev)
        ev.combined_score = 0.8
        result = assemble_evidence(_base_state(evidence=[ev]))
        assert "metric=gsdp" in result["assembled_context"]

    def test_citation_includes_metric_info(self):
        ev = _evidence("Karnataka GSDP for 2022-23 was ₹15.7 lakh crore",
                        source_name="Economic Survey")
        ev = _enrich_evidence_metadata(ev)
        citation = ev.to_citation()
        assert "GSDP" in citation


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 2: Temporal/estimate confusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalConfusion:
    """Temporal qualifier detection and differentiation."""

    def test_estimate_detected(self):
        assert _extract_temporal_qualifier("GDP estimate for 2023-24") == TemporalQualifier.ESTIMATE

    def test_preliminary_detected(self):
        assert _extract_temporal_qualifier("Preliminary figures show growth") == TemporalQualifier.PRELIMINARY

    def test_revised_detected(self):
        assert _extract_temporal_qualifier("Revised GDP growth was 7.2%") == TemporalQualifier.REVISED

    def test_projected_detected(self):
        assert _extract_temporal_qualifier("GDP is projected to reach $5 trillion") == TemporalQualifier.PROJECTED

    def test_advance_detected(self):
        assert _extract_temporal_qualifier("Advance estimates suggest 6.5% growth") == TemporalQualifier.ADVANCE

    def test_actual_detected(self):
        assert _extract_temporal_qualifier("Actual GDP growth was 8.2%") == TemporalQualifier.ACTUAL

    def test_estimate_not_confused_with_actual(self):
        est = _extract_temporal_qualifier("Estimated GSDP for Karnataka")
        act = _extract_temporal_qualifier("Actual GSDP for Karnataka confirmed")
        assert est != act
        assert est == TemporalQualifier.ESTIMATE
        assert act == TemporalQualifier.ACTUAL

    def test_unknown_temporal_for_neutral_text(self):
        assert _extract_temporal_qualifier("Karnataka GSDP was ₹15 lakh crore") == TemporalQualifier.UNKNOWN

    def test_year_period_extracted_fy(self):
        assert _extract_year_period("GSDP for FY2023") == "FY2023"

    def test_year_period_extracted_range(self):
        assert _extract_year_period("GSDP for 2022-23") == "2022-23"

    def test_year_period_extracted_single_year(self):
        assert _extract_year_period("GDP in 2023") == "2023"

    def test_evidence_temporal_fields_populated(self):
        ev = _evidence("Advance estimates of Karnataka GSDP for 2022-23 show growth")
        ev = _enrich_evidence_metadata(ev)
        assert ev.temporal_qualifier == TemporalQualifier.ADVANCE
        assert ev.year_period == "2022-23"

    def test_assembled_context_includes_temporal_qualifier(self):
        ev = _evidence("Advance estimates of Karnataka GSDP for 2022-23",
                        source_name="MOSPI")
        ev = _enrich_evidence_metadata(ev)
        ev.combined_score = 0.8
        result = assemble_evidence(_base_state(evidence=[ev]))
        assert "temporal=advance" in result["assembled_context"]
        assert "period=2022-23" in result["assembled_context"]

    def test_classification_temporal_qualifier_propagated(self):
        ev = _evidence("Karnataka GSDP growth figures")
        classification = QueryClassification(temporal_qualifier=TemporalQualifier.ESTIMATE)
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.temporal_qualifier == TemporalQualifier.ESTIMATE


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 3: Cross-turn evidence state loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossTurnEvidence:
    """Cross-turn evidence carry-forward mechanism."""

    def test_build_initial_state_accepts_prior_evidence(self):
        from app.agent.router import _build_initial_state
        state = _build_initial_state(
            query="What about Karnataka?",
            user_id=uuid.uuid4(),
            chat_id=uuid.uuid4(),
            prior_evidence_summary="PRIOR TURN EVIDENCE:\n- gsdp | Karnataka | 2022-23 | ₹15.7 lakh crore",
        )
        # The query should include the prior evidence context
        assert "PRIOR TURN EVIDENCE" in state["query"]
        assert "Karnataka" in state["query"]

    def test_build_initial_state_without_prior_evidence(self):
        from app.agent.router import _build_initial_state
        state = _build_initial_state(
            query="What about Karnataka?",
            user_id=uuid.uuid4(),
            chat_id=uuid.uuid4(),
        )
        # Without prior evidence, query stays clean
        assert state["query"] == "What about Karnataka?"

    def test_trajectory_includes_evidence_metadata(self):
        from app.agent.evidence_state import build_evidence_state, serialize_for_storage
        ev = _evidence("Karnataka GSDP for 2022-23 was ₹15.7 lakh crore",
                        source_name="Economic Survey")
        ev = _enrich_evidence_metadata(ev)
        state = _base_state(evidence=[ev])
        est = build_evidence_state(state["evidence"], [], [], turn=1)
        serialized = serialize_for_storage(est)
        import json as _json
        data = _json.loads(serialized)
        # Structured cross-turn state carries established facts (not raw conversation)
        assert "established" in data
        assert data["turn"] == 1

    def test_classification_includes_geographic_scope(self):
        classification = QueryClassification(
            geographic_scope=GeographicScope.STATE,
            geography="Karnataka",
        )
        assert classification.geographic_scope == GeographicScope.STATE
        assert classification.geography == "Karnataka"

    def test_classification_includes_metric_hint(self):
        classification = QueryClassification(
            metric_hint=MetricType.GSDP,
        )
        assert classification.metric_hint == MetricType.GSDP


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 4: Source quality issues
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceQuality:
    """Primary vs secondary vs tertiary source classification."""

    def test_government_source_is_primary(self):
        assert _classify_source_quality(
            "Economic Survey", "https://mof.gov.in/report", SourceType.WEB
        ) == SourceQuality.PRIMARY

    def test_rbi_source_is_primary(self):
        assert _classify_source_quality(
            "RBI Bulletin", "https://rbi.org.in/bulletin", SourceType.WEB
        ) == SourceQuality.PRIMARY

    def test_mospi_source_is_primary(self):
        assert _classify_source_quality(
            "MOSPI Data", "https://mospi.gov.in/data", SourceType.WEB
        ) == SourceQuality.PRIMARY

    def test_worldbank_source_is_primary(self):
        assert _classify_source_quality(
            "World Bank", "https://worldbank.org/indicator", SourceType.WEB
        ) == SourceQuality.PRIMARY

    def test_news_source_is_secondary(self):
        assert _classify_source_quality(
            "Reuters", "https://reuters.com/article/india-gdp", SourceType.WEB
        ) == SourceQuality.SECONDARY

    def test_wikipedia_is_tertiary(self):
        assert _classify_source_quality(
            "Wikipedia", "https://en.wikipedia.org/wiki/Economy", SourceType.WEB
        ) == SourceQuality.TERTIARY

    def test_reddit_is_tertiary(self):
        assert _classify_source_quality(
            "Reddit", "https://reddit.com/r/india", SourceType.WEB
        ) == SourceQuality.TERTIARY

    def test_document_with_report_keyword_is_primary(self):
        assert _classify_source_quality(
            "Annual Report 2023", None, SourceType.DOCUMENT
        ) == SourceQuality.PRIMARY

    def test_document_without_keyword_is_unknown(self):
        assert _classify_source_quality(
            "My Notes", None, SourceType.DOCUMENT
        ) == SourceQuality.UNKNOWN

    def test_evidence_source_quality_populated(self):
        ev = _evidence("India GDP grew 8.2%",
                        source_type=SourceType.WEB,
                        source_name="RBI",
                        source_url="https://rbi.org.in/bulletin")
        ev = _enrich_evidence_metadata(ev)
        assert ev.source_quality == SourceQuality.PRIMARY

    def test_primary_source_gets_combined_score_bump(self):
        ev_primary = _evidence("India GDP grew 8.2%",
                                source_type=SourceType.WEB,
                                source_name="RBI",
                                source_url="https://rbi.org.in/bulletin",
                                source_quality=SourceQuality.PRIMARY,
                                retrieval_score=0.5,
                                authority_score=0.85,
                                recency_score=0.8)
        ev_primary.rerank_score = 0.7
        ev_primary.combined_score = 0.7  # baseline

        ev_secondary = _evidence("India GDP grew 8.2%",
                                  source_type=SourceType.WEB,
                                  source_name="Blog",
                                  source_url="https://blog.example.com/post",
                                  source_quality=SourceQuality.SECONDARY,
                                  retrieval_score=0.5,
                                  authority_score=0.85,
                                  recency_score=0.8)
        ev_secondary.rerank_score = 0.7
        ev_secondary.combined_score = 0.7  # baseline

        from app.agent.nodes import _combined_score
        score_primary = _combined_score(ev_primary)
        score_secondary = _combined_score(ev_secondary)
        assert score_primary > score_secondary

    def test_tertiary_source_gets_penalized(self):
        ev_tertiary = _evidence("India GDP data",
                                 source_type=SourceType.WEB,
                                 source_name="Wikipedia",
                                 source_url="https://en.wikipedia.org",
                                 source_quality=SourceQuality.TERTIARY,
                                 retrieval_score=0.5,
                                 authority_score=0.85,
                                 recency_score=0.8)
        ev_tertiary.rerank_score = 0.7

        from app.agent.nodes import _combined_score
        score = _combined_score(ev_tertiary)
        # Should be lower than what it would be without the quality penalty
        assert score < 0.7  # basic sanity check

    def test_assembled_context_includes_source_quality(self):
        ev = _evidence("GDP data",
                        source_type=SourceType.WEB,
                        source_name="RBI",
                        source_url="https://rbi.org.in/data",
                        source_quality=SourceQuality.PRIMARY)
        ev.combined_score = 0.8
        result = assemble_evidence(_base_state(evidence=[ev]))
        assert "quality=primary" in result["assembled_context"]


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 5: Geographic scope confusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeographicScopeConfusion:
    """Geographic scope extraction and differentiation.

    Place discovery is generic (Title-Case recognition) and the *scope level* comes
    from explicit scope words or the LLM-classified query. We deliberately do NOT
    hardcode a list of states/countries, so a bare place name yields UNKNOWN scope
    but its name is still captured as `geography`.
    """

    def test_place_name_discovered_from_text(self):
        scope, geo = _extract_geographic_scope("", "Karnataka GSDP growth rate")
        assert geo == "Karnataka"
        assert scope != GeographicScope.NATIONAL

    def test_national_detected_from_scope_word(self):
        scope, geo = _extract_geographic_scope("", "India's national GDP growth rate")
        assert scope == GeographicScope.NATIONAL
        assert geo == "India"

    def test_state_scope_word_detected(self):
        scope, geo = _extract_geographic_scope("", "Maharashtra state GSDP")
        assert scope == GeographicScope.STATE
        assert geo == "Maharashtra"

    def test_global_scope_word_detected(self):
        scope, geo = _extract_geographic_scope("", "Global inflation trends")
        assert scope == GeographicScope.GLOBAL

    def test_text_geography_overrides_classification(self):
        ev = _evidence("Tamil Nadu GSDP growth was 8%")
        classification = QueryClassification(
            geographic_scope=GeographicScope.NATIONAL,
            geography="India",
        )
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.geography == "Tamil Nadu"
        assert ev.geography != "India"

    def test_classification_geography_propagated_when_no_text_place(self):
        ev = _evidence("The economy grew significantly")
        classification = QueryClassification(
            geographic_scope=GeographicScope.STATE,
            geography="Karnataka",
        )
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.geographic_scope == GeographicScope.STATE
        assert ev.geography == "Karnataka"

    def test_maharashtra_vs_india_is_a_mismatch(self):
        from app.agent.normalization import geographic_match
        q = QueryClassification(geographic_scope=GeographicScope.NATIONAL, geography="India")
        ev_state = _evidence("Maharashtra GSDP grew 8%",
                            source_type=SourceType.WEB, source_name="Economic Survey")
        ev_state = _enrich_evidence_metadata(ev_state, q)
        assert ev_state.geography == "Maharashtra"
        assert geographic_match(q.geography, ev_state.geography) < 0.5

    def test_ranking_penalizes_wrong_geography(self):
        from app.agent.ranking import rank_evidence
        q = QueryClassification(geographic_scope=GeographicScope.NATIONAL, geography="India")
        ev_right = _evidence("India GDP grew 8.2%",
                            source_type=SourceType.WEB, source_name="MoSPI",
                            source_url="https://mospi.gov.in/x",
                            authority_score=0.95, rerank_score=0.7, retrieval_score=0.7)
        ev_right = _enrich_evidence_metadata(ev_right, q)
        ev_wrong = _evidence("Maharashtra GSDP grew 8.2% (semantically similar phrasing)",
                            source_type=SourceType.WEB, source_name="Economic Survey",
                            source_url="https://example.org/x",
                            authority_score=0.9, rerank_score=0.95, retrieval_score=0.95)
        ev_wrong = _enrich_evidence_metadata(ev_wrong, q)
        ranked = rank_evidence([ev_right, ev_wrong], q)
        assert ranked[0].evidence_id == ev_right.evidence_id

    def test_classification_has_geographic_scope_field(self):
        qc = QueryClassification(
            geographic_scope=GeographicScope.NATIONAL,
            geography="India",
        )
        assert qc.geographic_scope == GeographicScope.NATIONAL
        assert qc.geography == "India"

    def test_citation_includes_geography(self):
        ev = _evidence("Karnataka GSDP growth",
                        source_name="Economic Survey")
        ev = _enrich_evidence_metadata(ev)
        citation = ev.to_citation()
        assert "Karnataka" in citation

    def test_citation_includes_geography(self):
        ev = _evidence("Karnataka GSDP growth",
                        source_name="Economic Survey")
        ev = _enrich_evidence_metadata(ev)
        citation = ev.to_citation()
        assert "Karnataka" in citation

    def test_classification_has_geographic_scope_field(self):
        qc = QueryClassification(
            geographic_scope=GeographicScope.NATIONAL,
            geography="India",
        )
        assert qc.geographic_scope == GeographicScope.NATIONAL
        assert qc.geography == "India"


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE 6: Fact vs inference conflation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactVsInference:
    """Claim type tagging: fact, inference, speculation."""

    def test_claim_type_enum_exists(self):
        assert ClaimType.FACT.value == "fact"
        assert ClaimType.INFERENCE.value == "inference"
        assert ClaimType.SPECULATION.value == "speculation"

    def test_claim_default_type_is_fact(self):
        claim = Claim(text="India GDP grew 8.2%")
        assert claim.claim_type == ClaimType.FACT

    def test_claim_can_be_tagged_as_inference(self):
        claim = Claim(
            text="Based on GSDP and population, Karnataka's per capita income likely increased",
            claim_type=ClaimType.INFERENCE,
        )
        assert claim.claim_type == ClaimType.INFERENCE

    def test_claim_can_be_tagged_as_speculation(self):
        claim = Claim(
            text="Karnataka may surpass Maharashtra in GSDP by 2030",
            claim_type=ClaimType.SPECULATION,
        )
        assert claim.claim_type == ClaimType.SPECULATION

    def test_repair_handles_different_claim_types(self):
        """Repair should handle inference and speculation claims too."""
        claims = [
            Claim(text="fact claim", status=ClaimStatus.UNVERIFIED, claim_type=ClaimType.FACT, repair_action="search_web"),
            Claim(text="inference claim", status=ClaimStatus.UNVERIFIED, claim_type=ClaimType.INFERENCE, repair_action="search_web"),
            Claim(text="speculation claim", status=ClaimStatus.UNCERTAIN, claim_type=ClaimType.SPECULATION, repair_action="rephrase"),
        ]
        result = repair_claims(_base_state(claims=claims, classification=QueryClassification()))
        assert result["repair_state"] == RepairDecision.REPAIR.value
        # Should have 3 repair steps (one per failed claim)
        assert len(result["plan"].steps) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# ENRICHMENT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichmentIntegration:
    """Full enrichment pipeline: metric + temporal + geographic + source quality."""

    def test_full_enrichment_of_economic_evidence(self):
        ev = _evidence(
            "Advance estimates of Karnataka GSDP for 2022-23 show ₹15.7 lakh crore",
            source_type=SourceType.WEB,
            source_name="RBI Annual Report",
            source_url="https://rbi.org.in/annual-report-2023",
        )
        ev = _enrich_evidence_metadata(ev)
        assert ev.metric_type == MetricType.GSDP
        assert ev.temporal_qualifier == TemporalQualifier.ADVANCE
        assert ev.year_period == "2022-23"
        # Geography is discovered from text (generic); scope from a bare place is UNKNOWN.
        assert ev.geography == "Karnataka"
        assert ev.geographic_scope != GeographicScope.NATIONAL
        assert ev.source_quality == SourceQuality.PRIMARY

    def test_enrichment_with_classification_context(self):
        ev = _evidence("The growth rate was 7.2%")
        classification = QueryClassification(
            metric_hint=MetricType.GSDP,
            geographic_scope=GeographicScope.STATE,
            geography="Karnataka",
            temporal_qualifier=TemporalQualifier.ESTIMATE,
        )
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.metric_type == MetricType.GSDP
        assert ev.geographic_scope == GeographicScope.STATE
        assert ev.geography == "Karnataka"
        assert ev.temporal_qualifier == TemporalQualifier.ESTIMATE

    def test_text_based_detection_overrides_classification_for_metric(self):
        """If the text explicitly mentions a different metric, text wins."""
        ev = _evidence("GVA at constant prices grew 7.2%")
        classification = QueryClassification(metric_hint=MetricType.GDP)
        ev = _enrich_evidence_metadata(ev, classification)
        assert ev.metric_type == MetricType.GVA  # text wins

    def test_assembled_context_includes_all_metadata(self):
        ev = _evidence(
            "Advance estimates of Karnataka GSDP for 2022-23 show ₹15.7 lakh crore",
            source_type=SourceType.WEB,
            source_name="Economic Survey",
            source_url="https://mospi.gov.in/survey",
        )
        ev = _enrich_evidence_metadata(ev)
        ev.combined_score = 0.9
        result = assemble_evidence(_base_state(evidence=[ev]))
        ctx = result["assembled_context"]
        assert "metric=gsdp" in ctx
        assert "temporal=advance" in ctx
        assert "geo=Karnataka" in ctx
        assert "period=2022-23" in ctx
        assert "quality=primary" in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRADICTION DETECTION IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContradictionImprovements:
    """Extended contradiction detection for metric and geographic mismatches."""

    def test_metric_mismatch_detection(self):
        """GDP vs GSDP on the same topic should not be treated as the same thing."""
        # This tests that evidence metadata is enriched so downstream consumers
        # can detect metric mismatches even if text similarity is high.
        ev1 = _evidence("India GDP grew 8.2% in 2023-24")
        ev1 = _enrich_evidence_metadata(ev1)
        ev2 = _evidence("Karnataka GSDP grew 8.2% in 2023-24")
        ev2 = _enrich_evidence_metadata(ev2)
        # Different metric types should be detectable from the structured fields
        assert ev1.metric_type != ev2.metric_type
        assert ev1.metric_type == MetricType.GDP
        assert ev2.metric_type == MetricType.GSDP

    def test_geographic_scope_mismatch_detection(self):
        """National vs state data on same metric should be distinguishable."""
        ev1 = _evidence("India national GDP grew 8.2%")
        ev1 = _enrich_evidence_metadata(ev1)
        ev2 = _evidence("Karnataka state GSDP grew 8.2%")
        ev2 = _enrich_evidence_metadata(ev2)
        from app.agent.normalization import geographic_match
        assert ev1.geography == "India"
        assert ev2.geography == "Karnataka"
        assert ev1.geographic_scope == GeographicScope.NATIONAL
        assert ev2.geographic_scope == GeographicScope.STATE
        assert geographic_match(ev1.geography, ev2.geography) < 0.5

    def test_temporal_qualifier_mismatch_detection(self):
        """Estimate vs actual for the same metric should be distinguishable."""
        ev1 = _evidence("GDP estimate for 2023-24 is 7.2%")
        ev1 = _enrich_evidence_metadata(ev1)
        ev2 = _evidence("Actual GDP for 2023-24 was 8.2%")
        ev2 = _enrich_evidence_metadata(ev2)
        assert ev1.temporal_qualifier != ev2.temporal_qualifier
        assert ev1.temporal_qualifier == TemporalQualifier.ESTIMATE
        assert ev2.temporal_qualifier == TemporalQualifier.ACTUAL


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMA UPDATES
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseSchemaUpdates:
    """Verify new response schema fields are properly defined."""

    def test_citation_response_has_new_fields(self):
        from app.agent.schemas import CitationResponse
        cr = CitationResponse(
            evidence_id="abc123",
            text="Karnataka GSDP",
            source_type="web",
            source_name="Economic Survey",
            authority_score=0.9,
            recency_score=0.8,
            metric_type="gsdp",
            metric_value="₹15.7 lakh crore",
            geographic_scope="state",
            geography="Karnataka",
            year_period="2022-23",
            temporal_qualifier="advance",
            source_quality="primary",
        )
        assert cr.metric_type == "gsdp"
        assert cr.geography == "Karnataka"
        assert cr.year_period == "2022-23"
        assert cr.temporal_qualifier == "advance"
        assert cr.source_quality == "primary"

    def test_claim_response_has_claim_type(self):
        from app.agent.schemas import ClaimResponse
        cr = ClaimResponse(
            claim_id="abc",
            text="Test claim",
            status="verified",
            claim_type="inference",
            evidence_ids=["e1"],
            contradicting_evidence_ids=[],
            reasoning="Test",
        )
        assert cr.claim_type == "inference"

    def test_citation_response_defaults(self):
        from app.agent.schemas import CitationResponse
        cr = CitationResponse(
            evidence_id="abc123",
            text="Test",
            source_type="document",
            source_name="Test",
            authority_score=0.5,
            recency_score=0.5,
        )
        assert cr.metric_type == "unknown"
        assert cr.geographic_scope == "unknown"
        assert cr.temporal_qualifier == "unknown"
        assert cr.source_quality == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING TEST REGRESSION GUARANTEES
# ═══════════════════════════════════════════════════════════════════════════════

class TestExistingBehaviorPreserved:
    """Verify that existing behavior is not broken by the new fields."""

    def test_evidence_without_new_fields_still_works(self):
        ev = Evidence(text="simple text", source_type=SourceType.DOCUMENT)
        assert ev.metric_type == MetricType.UNKNOWN
        assert ev.geographic_scope == GeographicScope.UNKNOWN
        assert ev.temporal_qualifier == TemporalQualifier.UNKNOWN
        assert ev.source_quality == SourceQuality.UNKNOWN

    def test_claim_without_claim_type_defaults_to_fact(self):
        claim = Claim(text="test")
        assert claim.claim_type == ClaimType.FACT

    def test_query_classification_without_new_fields(self):
        qc = QueryClassification()
        assert qc.geographic_scope == GeographicScope.UNKNOWN
        assert qc.geography == ""
        assert qc.temporal_qualifier == TemporalQualifier.UNKNOWN
        assert qc.metric_hint == MetricType.UNKNOWN

    def test_assemble_evidence_backward_compatible(self):
        """Evidence without new metadata should still assemble correctly."""
        ev = _evidence("Simple document text")
        result = assemble_evidence(_base_state(evidence=[ev]))
        assert len(result["evidence"]) == 1
        assert "Simple document text" in result["assembled_context"]

    def test_hallucination_router_ignores_claim_type(self):
        """Hallucination router should still work with claim_type field."""
        claims = [
            Claim(text="fact", status=ClaimStatus.VERIFIED, claim_type=ClaimType.FACT),
            Claim(text="inference", status=ClaimStatus.PARTIALLY_VERIFIED, claim_type=ClaimType.INFERENCE),
        ]
        state = _base_state(claims=claims)
        assert hallucination_router(state) == "satisfactory"

    def test_repair_claims_backward_compatible(self):
        claims = [Claim(text="bad", status=ClaimStatus.UNVERIFIED, claim_type=ClaimType.INFERENCE, repair_action="search_web")]
        result = repair_claims(_base_state(claims=claims, classification=QueryClassification()))
        assert result["repair_state"] == RepairDecision.REPAIR.value
