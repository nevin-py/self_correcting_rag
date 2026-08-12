"""
Architecture regression tests for the six observed failure modes + cross-turn state.

All tests here are DETERMINISTIC (no LLM / no network / no DB). They exercise the
pure normalization / ranking / conflict-classification / evidence-state / verification
modules directly.

Covers the 14 required scenarios:
 1. GSDP vs GVA confusion
 2. GVA vs output (GVA_SHARE vs OUTPUT_SHARE) confusion
 3. current vs constant prices
 4. actual vs advance estimate
 5. revised estimate vs advance estimate (an UPDATE, not a contradiction)
 6. Maharashtra vs India geographic mismatch
 7. irrelevant high-semantic-similarity source
 8. authoritative source vs low-quality source
 9. genuine contradiction
10. apparent contradiction caused by different years
11. multi-turn evidence persistence
12. newer evidence superseding older evidence
13. inference being presented as fact
14. unsupported causal claim
"""

import json

from app.agent.conflicts import classify_pair, detect_conflicts, ConflictType
from app.agent.evidence_state import (
    build_evidence_state,
    merge_evidence_state,
    evidence_state_from_json,
    serialize_for_storage,
    load_evidence_state_from_text,
    to_context_block,
)
from app.agent.normalization import (
    detect_metric_type,
    detect_price_basis,
    detect_temporal_qualifier,
    detect_place_mentions,
    extract_year_period,
    compose_search_query,
    geographic_match,
)
from app.agent.ranking import (
    rank_evidence,
    metric_match,
    select_latest_per_key,
    combined_score,
    evidence_fits_classification,
    filter_evidence_by_classification,
)
from app.agent.source_authority import authority_tier, classify_source_quality, authority_score
from app.agent.state import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    EvidenceState,
    GeographicScope,
    MetricType,
    PriceBasis,
    SourceQuality,
    SourceType,
    TemporalQualifier,
)
from app.agent.verification import audit_claims, extract_claim_entities


def _ev(text, **kw) -> Evidence:
    return Evidence(text=text, source_type=SourceType.WEB, **kw)


def _state(**over):
    s = {"evidence": [], "claims": [], "conflicts": [], "prior_evidence_state": None}
    s.update(over)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GSDP vs GVA confusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestGSDPvsGVA:
    def test_gsdp_detected(self):
        assert detect_metric_type("Karnataka GSDP grew by 8.3%") == MetricType.GSDP

    def test_gva_detected(self):
        assert detect_metric_type("India's GVA at constant prices") == MetricType.GVA

    def test_gdp_growth_rate_resolves_to_gdp_not_growth(self):
        # "GDP growth rate" must resolve to GDP, not the generic GROWTH_RATE.
        assert detect_metric_type("India GDP growth rate 2023") == MetricType.GDP

    def test_gsdp_gva_are_distinct(self):
        a = detect_metric_type("Karnataka GSDP for 2022-23")
        b = detect_metric_type("Karnataka GVA for 2022-23")
        assert a == MetricType.GSDP
        assert b == MetricType.GVA
        assert a != b


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GVA vs output confusion (GVA_SHARE != OUTPUT_SHARE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGVAvsOutput:
    def test_gva_share_detected(self):
        assert detect_metric_type("Services share of GVA was 54%") == MetricType.GVA_SHARE

    def test_output_share_detected(self):
        assert detect_metric_type("Manufacturing share of output was 17%") == MetricType.OUTPUT_SHARE

    def test_gva_share_not_confused_with_output_share(self):
        a = detect_metric_type("Share of GVA in services")
        b = detect_metric_type("Share of output in manufacturing")
        assert a == MetricType.GVA_SHARE
        assert b == MetricType.OUTPUT_SHARE
        assert a != b

    def test_ranker_treats_share_types_as_distinct(self):
        # Same geography/year but one is GVA share, other is output share -> not a match.
        q = _ev("", metric_type=MetricType.GVA_SHARE, geography="India", year_period="2022-23")
        e_gva = _ev("Services GVA share 54%", metric_type=MetricType.GVA_SHARE, geography="India", year_period="2022-23")
        e_out = _ev("Manufacturing output share 17%", metric_type=MetricType.OUTPUT_SHARE, geography="India", year_period="2022-23")
        assert metric_match(q.metric_type, e_gva.metric_type) == 1.0
        assert metric_match(q.metric_type, e_out.metric_type) < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. current vs constant prices
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceBasis:
    def test_current_prices(self):
        assert detect_price_basis("GDP at current prices was 300 lakh crore") == PriceBasis.CURRENT

    def test_nominal(self):
        assert detect_price_basis("nominal GDP rose") == PriceBasis.CURRENT

    def test_constant_prices(self):
        assert detect_price_basis("GVA at constant prices grew 7%") == PriceBasis.CONSTANT

    def test_real(self):
        assert detect_price_basis("real GDP increased") == PriceBasis.CONSTANT

    def test_price_basis_in_context(self):
        from app.agent.nodes import assemble_evidence
        ev = _ev("GDP at current prices was 300 lakh crore",
                 source_name="MoSPI", source_url="https://mospi.gov.in/x")
        ev = _enrich(ev)
        ev.combined_score = 0.9
        result = assemble_evidence(_state(evidence=[ev]))
        assert "price=current" in result["assembled_context"]

    def test_price_mismatch_flagged_in_verification(self):
        claim = Claim(text="GDP at constant prices was 300 lakh crore",
                      claim_type=ClaimType.FACT, status=ClaimStatus.VERIFIED,
                      evidence_ids=["e1"])
        ev = _ev("GDP at current prices was 300 lakh crore",
                 evidence_id="e1", metric_type=MetricType.GDP,
                 price_basis=PriceBasis.CURRENT, authority_score=0.9,
                 source_quality=SourceQuality.PRIMARY)
        errors = audit_claims([claim], [ev])
        assert any(e.issue == "PRICE_BASIS_MISMATCH" for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. actual vs advance estimate
# ═══════════════════════════════════════════════════════════════════════════════

class TestActualVsAdvance:
    def test_actual_detected(self):
        assert detect_temporal_qualifier("Actual GSDP for Karnataka confirmed") == TemporalQualifier.ACTUAL

    def test_advance_detected(self):
        assert detect_temporal_qualifier("Advance estimates suggest 6.5% growth") == TemporalQualifier.ADVANCE

    def test_actual_not_confused_with_advance(self):
        a = detect_temporal_qualifier("Actual GSDP")
        b = detect_temporal_qualifier("Advance estimate GSDP")
        assert a == TemporalQualifier.ACTUAL
        assert b == TemporalQualifier.ADVANCE
        assert a != b

    def test_year_extracted(self):
        assert extract_year_period("GSDP for 2024-25") == "2024-25"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. revised estimate vs advance estimate -> UPDATE, not contradiction
# ═══════════════════════════════════════════════════════════════════════════════

class TestRevisedVsAdvance:
    def test_classified_as_estimate_status_not_contradiction(self):
        a = _ev("2024-25 advance estimate of GSDP is 38 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2024-25",
                temporal_qualifier=TemporalQualifier.ADVANCE)
        b = _ev("2024-25 revised estimate of GSDP is 40 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2024-25",
                temporal_qualifier=TemporalQualifier.REVISED)
        ctype, reason = classify_pair(a, b)
        # An advance vs revised estimate is an UPDATE, not a contradiction.
        assert ctype in (ConflictType.DIFFERENT_ESTIMATE_STATUS, ConflictType.REVISED_VS_UNREVISED)
        assert ctype != ConflictType.GENUINE_CONTRADICTION

    def test_detect_conflicts_marks_it_as_non_contradiction(self):
        a = _ev("2024-25 advance estimate of GSDP is 38 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2024-25",
                temporal_qualifier=TemporalQualifier.ADVANCE, authority_score=0.9)
        b = _ev("2024-25 revised estimate of GSDP is 40 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2024-25",
                temporal_qualifier=TemporalQualifier.REVISED, authority_score=0.9)
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_type"] in (
            ConflictType.DIFFERENT_ESTIMATE_STATUS.value,
            ConflictType.REVISED_VS_UNREVISED.value,
        )
        assert conflicts[0]["is_contradiction"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Maharashtra vs India geographic mismatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeographicMismatch:
    def test_maharashtra_vs_india_mismatch(self):
        assert geographic_match("India", "Maharashtra") < 0.5

    def test_classified_as_different_scopes(self):
        # Same metric, different geography -> geographic scope conflict.
        a = _ev("India GSDP grew 8%", metric_type=MetricType.GSDP, geography="India",
                year_period="2023-24", temporal_qualifier=TemporalQualifier.ACTUAL)
        b = _ev("Maharashtra GSDP grew 8%", metric_type=MetricType.GSDP, geography="Maharashtra",
                year_period="2023-24", temporal_qualifier=TemporalQualifier.ACTUAL)
        ctype, reason = classify_pair(a, b)
        assert ctype == ConflictType.DIFFERENT_GEOGRAPHIC_SCOPES

    def test_place_discovery_is_generic(self):
        # Works for any place without a hardcoded list.
        assert detect_place_mentions("Maharashtra GSDP") == ["Maharashtra"]
        assert detect_place_mentions("Tamil Nadu GSDP") == ["Tamil Nadu"]
        assert detect_place_mentions("United States GDP") == ["United States"]

    def test_ranking_penalizes_wrong_geography(self):
        from app.agent.state import QueryClassification
        q = QueryClassification(geographic_scope=GeographicScope.NATIONAL, geography="India")
        ev_right = _ev("India GDP grew 8.2%", source_name="MoSPI",
                      source_url="https://mospi.gov.in/x", authority_score=0.95,
                      rerank_score=0.7, retrieval_score=0.7,
                      metric_type=MetricType.GDP, geography="India")
        # Highly semantically similar but about the wrong geography.
        ev_wrong = _ev("Maharashtra GSDP grew 8.2% (similar phrasing, different place)",
                      source_name="Survey", source_url="https://example.org/x",
                      authority_score=0.9, rerank_score=0.97, retrieval_score=0.97,
                      metric_type=MetricType.GSDP, geography="Maharashtra")
        ranked = rank_evidence([ev_right, ev_wrong], q)
        assert ranked[0].evidence_id == ev_right.evidence_id

    def test_hard_drop_geo_mismatch_when_geography_classified(self):
        from app.agent.state import QueryClassification
        q = QueryClassification(geography="India")
        ev_right = _ev("India GDP grew 8.2%", geography="India")
        ev_wrong = _ev("Unrelated coaching centres debate in another region", geography="Bihar")
        kept = filter_evidence_by_classification([ev_right, ev_wrong], q)
        assert ev_right in kept
        assert ev_wrong not in kept

    def test_no_geo_drop_when_geography_unclassified(self):
        from app.agent.state import QueryClassification
        q = QueryClassification(geography="")
        ev = _ev("Unrelated coaching centres debate")
        assert evidence_fits_classification(ev, q) is True
        assert filter_evidence_by_classification([ev], q) == [ev]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. irrelevant high-similarity source
# ═══════════════════════════════════════════════════════════════════════════════

class TestIrrelevantSimilarSource:
    def test_wrong_metric_penalized_despite_high_similarity(self):
        from app.agent.state import QueryClassification
        q = QueryClassification(metric_hint=MetricType.GSDP, geography="Karnataka")
        ev_target = _ev("Karnataka GSDP grew 8%", source_name="Economic Survey",
                        source_url="https://mospi.gov.in/x", authority_score=0.95,
                        rerank_score=0.7, retrieval_score=0.7,
                        metric_type=MetricType.GSDP, geography="Karnataka")
        # Semantically similar phrasing but about GVA, not GSDP.
        ev_distractor = _ev("Karnataka GVA grew 8% (similar wording)", source_name="Blog",
                            source_url="https://blog.example.com/x", authority_score=0.85,
                            rerank_score=0.98, retrieval_score=0.98,
                            metric_type=MetricType.GVA, geography="Karnataka")
        ranked = rank_evidence([ev_target, ev_distractor], q)
        assert ranked[0].evidence_id == ev_target.evidence_id


# ═══════════════════════════════════════════════════════════════════════════════
# 8. authoritative vs low-quality source
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceAuthority:
    def test_gov_is_primary(self):
        assert authority_tier("https://mof.gov.in/report", SourceType.WEB)[0] == "government"
        assert classify_source_quality("Survey", "https://mospi.gov.in/x", SourceType.WEB) == SourceQuality.PRIMARY

    def test_worldbank_primary(self):
        assert classify_source_quality("WB", "https://worldbank.org/indicator", SourceType.WEB) == SourceQuality.PRIMARY

    def test_wikipedia_tertiary(self):
        tier, score = authority_tier("https://en.wikipedia.org/wiki/Economy", SourceType.WEB)
        assert tier == "tertiary"
        assert score < 0.5
        assert classify_source_quality("Wiki", "https://en.wikipedia.org/x", SourceType.WEB) == SourceQuality.TERTIARY

    def test_no_hardcoded_per_domain_scores(self):
        # Authority is derived from DNS/TLD structure, not a per-domain lookup table.
        # A random .gov domain must still be treated as government.
        assert authority_tier("https://unknown-agency.gov.in/x", SourceType.WEB)[0] == "government"

    def test_primary_outranks_tertiary_in_ranking(self):
        ev_primary = _ev("India GDP grew 8.2%", source_name="RBI",
                         source_url="https://rbi.org.in/bulletin", authority_score=0.95,
                         rerank_score=0.7, retrieval_score=0.7, source_quality=SourceQuality.PRIMARY)
        ev_tertiary = _ev("India GDP data", source_name="Wikipedia",
                          source_url="https://en.wikipedia.org/x", authority_score=0.3,
                          rerank_score=0.7, retrieval_score=0.7, source_quality=SourceQuality.TERTIARY)
        assert combined_score(ev_primary) > combined_score(ev_tertiary)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. genuine contradiction
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenuineContradiction:
    def test_negation(self):
        ctype, _ = classify_pair(_ev("The product launched"), _ev("The product did not launch"))
        assert ctype == ConflictType.GENUINE_CONTRADICTION

    def test_antonym(self):
        ctype, _ = classify_pair(_ev("Profits increased"), _ev("Profits decreased"))
        assert ctype == ConflictType.GENUINE_CONTRADICTION

    def test_numeric(self):
        ctype, _ = classify_pair(_ev("Revenue was 100 million"), _ev("Revenue was 200 million"))
        assert ctype == ConflictType.GENUINE_CONTRADICTION


# ═══════════════════════════════════════════════════════════════════════════════
# 10. apparent contradiction caused by different years
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferentYears:
    def test_classified_as_different_years(self):
        a = _ev("Karnataka GSDP 2022-23 was 30 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2022-23",
                temporal_qualifier=TemporalQualifier.ACTUAL)
        b = _ev("Karnataka GSDP 2023-24 was 33 lakh crore",
                metric_type=MetricType.GSDP, geography="Karnataka", year_period="2023-24",
                temporal_qualifier=TemporalQualifier.ACTUAL)
        ctype, _ = classify_pair(a, b)
        assert ctype == ConflictType.DIFFERENT_YEARS
        assert ctype != ConflictType.GENUINE_CONTRADICTION

    def test_not_flagged_as_contradiction(self):
        a = _ev("GDP 2022-23 was 100", metric_type=MetricType.GDP, geography="India", year_period="2022-23")
        b = _ev("GDP 2023-24 was 110", metric_type=MetricType.GDP, geography="India", year_period="2023-24")
        conflicts = detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0]["is_contradiction"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 11. multi-turn evidence persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossTurnPersistence:
    def test_roundtrip_serialization(self):
        ev = _ev("Karnataka GSDP 2023-24 was 33 lakh crore", metric_type=MetricType.GSDP,
                 geography="Karnataka", year_period="2023-24", temporal_qualifier=TemporalQualifier.ACTUAL,
                 source_name="Economic Survey", authority_score=0.9)
        est = build_evidence_state([ev], [], [], turn=1)
        serialized = serialize_for_storage(est)
        loaded = load_evidence_state_from_text("node1 -> node2\n" + serialized)
        assert loaded is not None
        assert len(loaded.established) == 1
        assert loaded.established[0].geography == "Karnataka"

    def test_established_carried_into_next_turn_context(self):
        from app.agent.nodes import assemble_evidence
        prior_ev = _ev("Karnataka GSDP 2023-24 was 33 lakh crore", metric_type=MetricType.GSDP,
                       geography="Karnataka", year_period="2023-24",
                       temporal_qualifier=TemporalQualifier.ACTUAL, source_name="Economic Survey")
        prior = EvidenceState(established=[prior_ev], turn=1)
        result = assemble_evidence(_state(prior_evidence_state=prior))
        ctx = result["assembled_context"]
        assert "[CROSS-TURN EVIDENCE STATE]" in ctx
        assert "ESTABLISHED FACTS" in ctx
        assert "Karnataka GSDP 2023-24" in ctx

    def test_context_block_distinguishes_categories(self):
        prior = EvidenceState(
            established=[_ev("Fact A about India GDP", metric_type=MetricType.GDP, geography="India")],
            inferences=[_ev("Inference B", metric_type=MetricType.GDP, geography="India")],
            superseded=[_ev("Old figure", metric_type=MetricType.GDP, geography="India")],
            unresolved=["Unresolved claim C"],
        )
        block = to_context_block(prior)
        assert "ESTABLISHED FACTS" in block
        assert "PRIOR INFERENCES" in block
        assert "SUPERSEDED" in block
        assert "UNRESOLVED CLAIMS" in block


# ═══════════════════════════════════════════════════════════════════════════════
# 12. newer evidence superseding older evidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupersession:
    def test_select_latest_prefers_newer_year(self):
        old = _ev("2023-24 GSDP was 30", metric_type=MetricType.GSDP, geography="India",
                 year_period="2023-24", temporal_qualifier=TemporalQualifier.ACTUAL, authority_score=0.9)
        new = _ev("2024-25 GSDP was 33", metric_type=MetricType.GSDP, geography="India",
                 year_period="2024-25", temporal_qualifier=TemporalQualifier.ADVANCE, authority_score=0.9)
        latest = select_latest_per_key([old, new])
        assert latest[("gsdp", "india")].year_period == "2024-25"

    def test_merge_marks_older_as_superseded(self):
        # Same metric/geo/period but a revised figure replaces the advance estimate.
        old = _ev("2024-25 GSDP was 38 (advance)", metric_type=MetricType.GSDP, geography="India",
                 year_period="2024-25", temporal_qualifier=TemporalQualifier.ADVANCE, authority_score=0.9)
        new = _ev("2024-25 GSDP was 40 (revised)", metric_type=MetricType.GSDP, geography="India",
                 year_period="2024-25", temporal_qualifier=TemporalQualifier.REVISED, authority_score=0.9)
        prior = EvidenceState(established=[old], turn=1)
        current = EvidenceState(established=[new], turn=2)
        merged = merge_evidence_state(prior, current)
        assert any(e.evidence_id == old.evidence_id for e in merged.superseded)
        assert any(e.evidence_id == new.evidence_id for e in merged.established)

    def test_newer_does_not_overwrite_older_when_lower_authority(self):
        # A newer but low-quality source should NOT automatically win over an older
        # authoritative one for the SAME year/metric.
        auth = _ev("2024-25 GSDP was 33 (official)", metric_type=MetricType.GSDP, geography="India",
                  year_period="2024-25", temporal_qualifier=TemporalQualifier.ADVANCE, authority_score=0.95)
        lowq = _ev("2024-25 GSDP was 99 (blog)", metric_type=MetricType.GSDP, geography="India",
                  year_period="2024-25", temporal_qualifier=TemporalQualifier.ADVANCE, authority_score=0.3)
        latest = select_latest_per_key([auth, lowq])
        assert latest[("gsdp", "india")].evidence_id == auth.evidence_id


# ═══════════════════════════════════════════════════════════════════════════════
# 13. inference being presented as fact
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferenceVsFact:
    def test_inference_claim_preserved_as_inference(self):
        claim = Claim(text="Based on GSDP and population, per-capita income likely rose",
                      claim_type=ClaimType.INFERENCE, status=ClaimStatus.VERIFIED, evidence_ids=["e1"])
        ev = _ev("GSDP rose", evidence_id="e1", metric_type=MetricType.GSDP,
                 geography="Karnataka", authority_score=0.9, source_quality=SourceQuality.PRIMARY)
        est = build_evidence_state([ev], [claim], [], turn=1)
        assert len(est.inferences) == 1
        assert len(est.established) == 0

    def test_causal_factual_claim_flagged(self):
        # A FACT claim stating a causal conclusion stronger than a single source.
        claim = Claim(text="Higher GSDP causes higher employment",
                      claim_type=ClaimType.FACT, status=ClaimStatus.VERIFIED, evidence_ids=["e1"])
        ev = _ev("GSDP rose in the state", evidence_id="e1", metric_type=MetricType.GSDP,
                 geography="Karnataka", authority_score=0.9, source_quality=SourceQuality.PRIMARY)
        errors = audit_claims([claim], [ev])
        assert any(e.issue == "UNSUPPORTED_CAUSATION" for e in errors)

    def test_inference_label_surfaces_in_verification(self):
        claim = Claim(text="It can be inferred that growth will continue",
                      claim_type=ClaimType.INFERENCE, status=ClaimStatus.VERIFIED, evidence_ids=[])
        # No evidence cited but it is explicitly an inference -> not an unsupported-fact error.
        errors = audit_claims([claim], [])
        assert not any(e.issue == "UNSUPPORTED_CLAIM" for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. unsupported causal claim
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnsupportedCausation:
    def test_correlation_not_causation(self):
        claim = Claim(text="Because services grew, poverty fell",
                      claim_type=ClaimType.FACT, status=ClaimStatus.VERIFIED, evidence_ids=["e1"])
        ev = _ev("Services sector grew", evidence_id="e1", metric_type=MetricType.GVA_SHARE,
                 geography="India", authority_score=0.9, source_quality=SourceQuality.PRIMARY)
        errors = audit_claims([claim], [ev])
        assert any(e.issue == "UNSUPPORTED_CAUSATION" for e in errors)

    def test_metric_mismatch_in_claim(self):
        claim = Claim(text="GSDP grew 8%", claim_type=ClaimType.FACT, status=ClaimStatus.VERIFIED,
                      evidence_ids=["e1"])
        # Evidence is about GVA, not GSDP.
        ev = _ev("GVA grew 8%", evidence_id="e1", metric_type=MetricType.GVA,
                 geography="Karnataka", authority_score=0.9, source_quality=SourceQuality.PRIMARY)
        errors = audit_claims([claim], [ev])
        assert any(e.issue == "METRIC_MISMATCH" for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# Search-term composition (LLM-driven explicit queries)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchComposition:
    def test_compose_includes_metric_and_geo(self):
        q = compose_search_query(geography="USA", metric=MetricType.GDP, temporal=TemporalQualifier.ACTUAL)
        assert "USA" in q
        assert "GDP" in q
        assert "actual" in q

    def test_compose_handles_growth_rate(self):
        q = compose_search_query(geography="USA", metric=MetricType.GROWTH_RATE)
        assert "growth rate" in q


# ═══════════════════════════════════════════════════════════════════════════════
# 15. End-to-end graph smoke test (mocked LLMs, no network)
# ═══════════════════════════════════════════════════════════════════════════════

import uuid

import pytest
from unittest.mock import MagicMock, patch

from app.agent.graph import rag_app
from app.agent.nodes import _ClaimList
from app.agent.state import (
    PlanStep, PlannerOutput, QueryClassification,
)


@pytest.mark.asyncio
class TestEndToEndGraph:
    async def test_full_graph_with_prior_evidence_state(self):
        classify_resp = QueryClassification(
            needs_documents=True, needs_web=False, rewrite="Karnataka GSDP?",
            metric_hint=MetricType.GSDP, geography="Karnataka",
            geographic_scope=GeographicScope.STATE,
        )
        plan_resp = PlannerOutput(
            classification=classify_resp,
            steps=[PlanStep(action="retrieve_documents", queries=["Karnataka GSDP"],
                           expected_claims=[], rationale="")],
        )
        claim_resp = _ClaimList(claims=[
            Claim(text="Karnataka GSDP grew", claim_type=ClaimType.FACT,
                  status=ClaimStatus.VERIFIED, evidence_ids=[], reasoning="",
                  repair_action="none"),
        ])

        def _structured(schema):
            if schema is PlannerOutput:
                return MagicMock(invoke=MagicMock(return_value=plan_resp))
            if schema is QueryClassification:
                return MagicMock(invoke=MagicMock(return_value=classify_resp))
            if schema is _ClaimList:
                return MagicMock(invoke=MagicMock(return_value=claim_resp))
            return MagicMock(invoke=MagicMock(return_value=None))

        prior = EvidenceState(
            established=[_ev("Karnataka GSDP 2023-24 was 33 lakh crore",
                           metric_type=MetricType.GSDP, geography="Karnataka",
                           year_period="2023-24", temporal_qualifier=TemporalQualifier.ACTUAL,
                           source_name="Economic Survey", authority_score=0.9)],
            turn=1,
        )

        state = {
            "user_id": uuid.uuid4(), "chat_id": uuid.uuid4(), "query": "What about Karnataka GSDP?",
            "provider": "auto", "messages": [], "graph_steps": 0, "search_count": 0,
            "retrieval_count": 0, "regeneration_count": 0,
            "max_graph_steps": 12, "max_searches": 2, "max_retrievals": 3, "max_regenerations": 2,
            "evidence": [], "claims": [], "conflicts": [], "citation_usage": [],
            "assembled_context": "", "evidence_state": None, "prior_evidence_state": prior,
            "verification_errors": [], "classification": None, "plan": None, "answer": "",
            "final_status": "", "chunks": [], "search": [], "planner_state": "",
            "retrieval_queries": [], "wiki_queries": [], "tavily_queries": [],
            "searxng_queries": [], "repair_state": "", "provider_used": "",
            "need_repair": "", "hallucination_reason": [], "max_tries_planner": 0,
            "max_tries_hallucinator": 0, "steps_taken": 0, "searches_done": 0,
            "retrievals_done": 0, "regenerations_done": 0, "cross_chat_enabled": False,
        }

        with patch("app.agent.nodes.openrouter_planner_llm", MagicMock(with_structured_output=_structured)), \
             patch("app.agent.nodes.routing_llm", MagicMock(with_structured_output=_structured)), \
             patch("app.agent.nodes.openrouter_hallucination_llm", MagicMock(with_structured_output=_structured)), \
             patch("app.agent.nodes.openrouter_generator_llm", None), \
             patch("app.agent.nodes.chat_llm", MagicMock(invoke=MagicMock(return_value=MagicMock(content="Karnataka GSDP is 33 lakh crore.")))):
            final = await rag_app.ainvoke(state)

        assert final["answer"]
        assert isinstance(final.get("verification_errors", []), list)
        # Cross-turn established evidence must reach the generator context.
        assert "[CROSS-TURN EVIDENCE STATE]" in final["assembled_context"]
        assert "Karnataka GSDP 2023-24" in final["assembled_context"]


# Helper: enrich an Evidence using the node's enrichment (keeps tests DRY).
from app.agent.nodes import _enrich_evidence_metadata  # noqa: E402


def _enrich(ev: Evidence):
    return _enrich_evidence_metadata(ev)

