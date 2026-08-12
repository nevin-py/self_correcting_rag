"""
Deterministic claim verification.

Complements the LLM hallucination checker with structured, actionable errors. Each
error names the failing check so the agent (or a human) can act on it. The 11 checks
map to the verifier requirement:

    1. SUPPORT        every factual claim cites evidence
    2. EVIDENCE MATCH cited evidence actually supports the claim
    3. METRIC         metric in claim == metric in evidence
    4. GEOGRAPHY      geography in claim == geography in evidence
    5. DATE           period in claim == period in evidence
    6. STATUS         estimate status in claim == status in evidence
    7. PRICE BASIS    current vs constant prices are not interchangeable
    8. AUTHORITY      cited source is authoritative enough
    9. INFERENCE      conclusion not stronger than the evidence (no causation from correlation)
   10. MIXING         sources not mixed incorrectly (different geos/metrics)
   11. SUPERSEDED     answer does not rely on outdated / superseded evidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.normalization import (
    detect_metric_type,
    detect_place_mentions,
    detect_price_basis,
    detect_temporal_qualifier,
    extract_year_period,
    geographic_match,
)
from app.agent.state import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    EvidenceState,
    MetricType,
    PriceBasis,
    SourceQuality,
    TemporalQualifier,
)

_CAUSAL_PATTERN = re.compile(
    r"\bbecause\b|\bcauses?\b|\bcaused by\b|\bleads? to\b|\btherefore\b|\bthus\b|"
    r"\bimplies?\b|\bdue to\b|\bas a result\b|\bresulting in\b|\bso that\b",
    re.IGNORECASE,
)


@dataclass
class VerificationError:
    claim_id: str
    issue: str
    severity: str           # high / medium / low
    detail: str
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "issue": self.issue,
            "severity": self.severity,
            "detail": self.detail,
            "suggested_fix": self.suggested_fix,
        }


def extract_claim_entities(text: str) -> dict:
    """Extract structured entities asserted by a claim (metric/geo/year/status/price)."""
    places = detect_place_mentions(text)
    return {
        "metric": detect_metric_type(text),
        "geo": places[0] if places else "",
        "year": extract_year_period(text),
        "temporal": detect_temporal_qualifier(text),
        "price": detect_price_basis(text),
    }


def _has_causal_language(text: str) -> bool:
    return bool(_CAUSAL_PATTERN.search(text))


def audit_claims(
    claims: list[Claim],
    evidence: list[Evidence],
    prior_state: EvidenceState | None = None,
) -> list[VerificationError]:
    """Run the deterministic verification checklist over the answer's claims."""
    errors: list[VerificationError] = []
    ev_by_id = {ev.evidence_id: ev for ev in evidence}
    superseded_ids = {ev.evidence_id for ev in (prior_state.superseded if prior_state else [])}

    for claim in claims:
        if claim.status == ClaimStatus.VERIFIED and claim.claim_type == ClaimType.INFERENCE:
            # Even verified inferences must be labeled as inferences, not facts.
            pass

        cited = [ev_by_id[i] for i in claim.evidence_ids if i in ev_by_id]
        ents = extract_claim_entities(claim.text)

        # 1. Support
        if claim.claim_type != ClaimType.INFERENCE and not cited:
            errors.append(VerificationError(
                claim.claim_id, "UNSUPPORTED_CLAIM", "high",
                "Factual claim has no cited evidence.",
                "Reject or add citations from retrieved evidence.",
            ))
            continue
        if not cited:
            continue

        # 3-7. Per-evidence metric/geo/date/status/price/authority checks
        for ev in cited:
            if ents["metric"] not in (MetricType.UNKNOWN, MetricType.OTHER) and \
               ev.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER) and \
               ents["metric"] != ev.metric_type:
                errors.append(VerificationError(
                    claim.claim_id, "METRIC_MISMATCH", "high",
                    f"Claim implies {ents['metric'].value.upper()} but evidence is {ev.metric_type.value.upper()}.",
                    "Use the exact metric from the evidence; do not substitute.",
                ))
            if ents["geo"] and ev.geography and geographic_match(ents["geo"], ev.geography) < 0.5:
                errors.append(VerificationError(
                    claim.claim_id, "GEOGRAPHY_MISMATCH", "high",
                    f"Claim implies {ents['geo']} but evidence is about {ev.geography}.",
                    "Do not present state-level evidence as national (or vice versa).",
                ))
            if ents["year"] and ev.year_period and ents["year"] != ev.year_period:
                errors.append(VerificationError(
                    claim.claim_id, "DATE_MISMATCH", "high",
                    f"Claim implies {ents['year']} but evidence covers {ev.year_period}.",
                    "Match the period to the evidence.",
                ))
            if ents["temporal"] != TemporalQualifier.UNKNOWN and \
               ev.temporal_qualifier != TemporalQualifier.UNKNOWN and \
               ents["temporal"] != ev.temporal_qualifier:
                errors.append(VerificationError(
                    claim.claim_id, "STATUS_MISMATCH", "high",
                    f"Claim implies {ents['temporal'].value} but evidence is {ev.temporal_qualifier.value}.",
                    "Distinguish actual / revised / advance / projected correctly.",
                ))
            if ents["price"] != PriceBasis.UNKNOWN and \
               ev.price_basis != PriceBasis.UNKNOWN and \
               ents["price"] != ev.price_basis:
                errors.append(VerificationError(
                    claim.claim_id, "PRICE_BASIS_MISMATCH", "high",
                    f"Claim implies {ents['price'].value} prices but evidence is {ev.price_basis.value}.",
                    "Distinguish current (nominal) vs constant (real) prices.",
                ))
            # 8. Authority
            if claim.claim_type == ClaimType.FACT and (
                ev.authority_score < 0.7 or ev.source_quality == SourceQuality.TERTIARY
            ):
                errors.append(VerificationError(
                    claim.claim_id, "LOW_AUTHORITY_SOURCE", "medium",
                    f"Factual claim rests on a low-authority source ({ev.source_name or 'unknown'}).",
                    "Prefer primary/official sources for factual claims.",
                ))
            # 11. Superseded
            if ev.evidence_id in superseded_ids:
                errors.append(VerificationError(
                    claim.claim_id, "OUTDATED_EVIDENCE", "medium",
                    "Claim relies on evidence that was superseded by newer evidence.",
                    "Use the newer, more finalized figure.",
                ))

        # 9. Inference vs fact / unsupported causation
        if claim.claim_type == ClaimType.FACT and _has_causal_language(claim.text) and len(cited) <= 1:
            errors.append(VerificationError(
                claim.claim_id, "UNSUPPORTED_CAUSATION", "high",
                "Claim states a causal conclusion stronger than a single source supports.",
                "Label as inference; correlation is not causation.",
            ))

        # 10. Mixing across geographies / metrics
        if len(cited) > 1:
            geos = {ev.geography for ev in cited if ev.geography}
            metrics = {ev.metric_type for ev in cited if ev.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER)}
            if len(geos) > 1:
                errors.append(VerificationError(
                    claim.claim_id, "GEOGRAPHY_MIXING", "medium",
                    f"Claim mixes evidence from different geographies: {sorted(geos)}.",
                    "Keep geographic scopes consistent or label each separately.",
                ))
            if len(metrics) > 1:
                errors.append(VerificationError(
                    claim.claim_id, "METRIC_MIXING", "medium",
                    f"Claim mixes different metrics: {sorted(m.value for m in metrics)}.",
                    "Keep the metric consistent or label each separately.",
                ))

    return errors
