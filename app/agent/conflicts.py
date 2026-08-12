"""
Conflict classification — not every disagreement is a contradiction.

Given two evidence items we classify their relationship using the *structured*
fields (metric, geography, year, temporal qualifier, value) rather than naive
text similarity:

    GENUINE_CONTRADICTION      same metric/geo/period, opposite facts
    SOURCE_DISAGREEMENT        same facts, different sources disagree
    DIFFERENT_YEARS            different periods -> not a contradiction
    DIFFERENT_ESTIMATE_STATUS  advance vs revised vs actual -> an *update*
    DIFFERENT_METRICS          e.g. GSDP vs GVA
    DIFFERENT_GEOGRAPHIC_SCOPES national vs state
    REVISED_VS_UNREVISED       updated figure replaces older one
    INSUFFICIENT_OVERLAP       unrelated -> not a conflict at all

Important: 2024-25 advance estimate = X and 2024-25 revised estimate = Y is
classified as DIFFERENT_ESTIMATE_STATUS (an update), NOT a contradiction.
"""

from __future__ import annotations

import re

from app.agent.normalization import geographic_match as _geo_match
from app.agent.ranking import status_priority
from app.agent.state import (
    ConflictType,
    Evidence,
    MetricType,
    SourceType,
    TemporalQualifier,
)

_NEGATION_WORDS = {
    "not", "no", "never", "none", "cannot", "can't", "isn't", "isnt", "aren't",
    "arent", "wasn't", "wasnt", "weren't", "werent", "don't", "dont", "doesn't",
    "doesnt", "didn't", "didnt", "won't", "wont", "wouldn't", "wouldnt",
    "shouldn't", "shouldnt",
}

_ANTONYM_PAIRS = [
    {"increase", "decrease"}, {"increased", "decreased"}, {"increasing", "decreasing"},
    {"rise", "fall"}, {"rose", "fell"}, {"rising", "falling"}, {"rises", "falls"},
    {"higher", "lower"}, {"high", "low"}, {"more", "less"}, {"faster", "slower"},
    {"fast", "slow"}, {"before", "after"}, {"buy", "sell"}, {"bought", "sold"},
    {"buying", "selling"}, {"approve", "reject"}, {"approved", "rejected"},
    {"accept", "deny"}, {"accepted", "denied"}, {"true", "false"}, {"yes", "no"},
    {"up", "down"}, {"positive", "negative"}, {"gain", "loss"}, {"gained", "lost"},
    {"gains", "losses"}, {"profit", "loss"}, {"profits", "losses"},
    {"profitable", "unprofitable"}, {"bullish", "bearish"}, {"expand", "contract"},
    {"expanded", "contracted"}, {"expanding", "contracting"}, {"hire", "fire"},
    {"hired", "fired"}, {"hiring", "firing"}, {"launch", "cancel"},
    {"launched", "cancelled"}, {"launching", "cancelling"},
]


def _normalize_claim(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ngram_overlap(a: str, b: str, n: int = 2) -> float:
    tokens_a = _normalize_claim(a).split()
    tokens_b = _normalize_claim(b).split()
    if len(tokens_a) < n + 1 or len(tokens_b) < n + 1:
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))
    grams_a = {" ".join(tokens_a[i:i + n]) for i in range(len(tokens_a) - n + 1)}
    grams_b = {" ".join(tokens_b[i:i + n]) for i in range(len(tokens_b) - n + 1)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / max(len(grams_a), len(grams_b))


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", _normalize_claim(text)))


def _evidence_sort_key(ev: Evidence) -> tuple[float, float, int]:
    """Authority-first, then status finality, then year recency."""
    from app.agent.normalization import parse_year
    year = parse_year(ev.year_period) or 0
    return (ev.authority_score, status_priority(ev.temporal_qualifier), year)


def classify_pair(a: Evidence, b: Evidence) -> tuple[ConflictType, str]:
    """Classify the relationship between two evidence items."""
    overlap = _ngram_overlap(a.text, b.text)
    # Two items measuring the SAME metric are comparable even with low textual
    # overlap (e.g. "GDP 2022-23 was 100" vs "GDP 2023-24 was 110").
    comparable = (
        a.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER)
        and b.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER)
        and a.metric_type == b.metric_type
    )
    if overlap < 0.20 and not comparable:
        return ConflictType.INSUFFICIENT_OVERLAP, "low topical overlap"

    # Different metrics -> not a contradiction, just different measurements.
    if a.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER) and \
       b.metric_type not in (MetricType.UNKNOWN, MetricType.OTHER) and \
       a.metric_type != b.metric_type:
        return ConflictType.DIFFERENT_METRICS, f"different metrics ({a.metric_type.value} vs {b.metric_type.value})"

    # Different geographies -> different scopes, not a contradiction.
    if a.geography and b.geography and _geo_match(a.geography, b.geography) < 0.5:
        return ConflictType.DIFFERENT_GEOGRAPHIC_SCOPES, f"different geographies ({a.geography} vs {b.geography})"

    # Different years -> temporal update, not a contradiction.
    if a.year_period and b.year_period and a.year_period != b.year_period:
        return ConflictType.DIFFERENT_YEARS, f"different periods ({a.year_period} vs {b.year_period})"

    # Same year but different estimate status -> an update, not a contradiction.
    if (a.temporal_qualifier != TemporalQualifier.UNKNOWN and
            b.temporal_qualifier != TemporalQualifier.UNKNOWN and
            a.temporal_qualifier != b.temporal_qualifier):
        if TemporalQualifier.REVISED in (a.temporal_qualifier, b.temporal_qualifier):
            return ConflictType.REVISED_VS_UNREVISED, (
                f"revised vs unrevised ({a.temporal_qualifier.value} vs {b.temporal_qualifier.value})"
            )
        return ConflictType.DIFFERENT_ESTIMATE_STATUS, (
            f"different estimate status ({a.temporal_qualifier.value} vs {b.temporal_qualifier.value})"
        )

    # Same metric/geo/period: check for genuine opposition.
    a_norm = _normalize_claim(a.text)
    b_norm = _normalize_claim(b.text)
    a_neg = bool(_NEGATION_WORDS & set(a_norm.split()))
    b_neg = bool(_NEGATION_WORDS & set(b_norm.split()))
    if a_neg != b_neg:
        return ConflictType.GENUINE_CONTRADICTION, "polarity mismatch (negation)"

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    for pair in _ANTONYM_PAIRS:
        if len(a_tokens & pair) > 0 and len(b_tokens & pair) > 0:
            return ConflictType.GENUINE_CONTRADICTION, f"antonym pair detected: {sorted(pair)}"

    a_nums = _extract_numbers(a.text)
    b_nums = _extract_numbers(b.text)
    if a_nums and b_nums and a_nums != b_nums:
        return ConflictType.GENUINE_CONTRADICTION, "numeric mismatch on same quantity"

    # Same facts but sources disagree on value without clear opposition.
    return ConflictType.SOURCE_DISAGREEMENT, "sources report different values for same fact"


def detect_conflicts(evidence: list[Evidence]) -> list[dict]:
    """Pairwise conflict detection returning structured, classified records."""
    conflicts: list[dict] = []
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]
            ctype, reason = classify_pair(a, b)
            if ctype in (ConflictType.INSUFFICIENT_OVERLAP,):
                continue
            # Choose the winner (preferred source) for genuine disagreements.
            winner = a if _evidence_sort_key(a) >= _evidence_sort_key(b) else b
            loser = b if winner is a else a
            conflicts.append({
                "evidence_a": a.evidence_id,
                "evidence_b": b.evidence_id,
                "conflict_type": ctype.value,
                "reason": reason,
                "winner": winner.evidence_id,
                "loser": loser.evidence_id,
                "is_contradiction": ctype
                in (ConflictType.GENUINE_CONTRADICTION, ConflictType.SOURCE_DISAGREEMENT),
                "winner_reasoning": (
                    "prefer higher authority / more finalized status / newer period"
                ),
            })
    return conflicts


def is_genuine_contradiction(claim_text: str, evidence_text: str) -> tuple[bool, str]:
    """Backward-compatible contradiction check used by the verifier.

    Returns (is_contradant, reason). Treats genuine contradiction and source
    disagreement as contradictions; classified non-contradictions (different
    years / status / metrics / geographies) are NOT contradictions.
    """
    # Lightweight: build transient Evidence objects to reuse classify_pair.
    a = Evidence(text=claim_text, source_type=SourceType.UNKNOWN)
    b = Evidence(text=evidence_text, source_type=SourceType.UNKNOWN)
    ctype, reason = classify_pair(a, b)
    is_contra = ctype in (ConflictType.GENUINE_CONTRADICTION, ConflictType.SOURCE_DISAGREEMENT)
    return is_contra, reason
