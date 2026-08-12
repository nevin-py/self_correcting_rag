"""
Evidence ranking — semantic relevance + authority + recency + geographic/metric fit.

This is the SOURCE RANKING layer of the pipeline. It deliberately does NOT rely
solely on embedding/rerank similarity. A source that is semantically very similar
but answers a *different geography or metric* is penalized so it cannot outrank a
highly authoritative, directly relevant source.
"""

from __future__ import annotations

import math

from app.agent.normalization import (
    geographic_match as _geo_match,
    metric_search_term,
    parse_year,
)
from app.agent.state import (
    Evidence,
    GeographicScope,
    MetricType,
    QueryClassification,
    SourceQuality,
    TemporalQualifier,
)

# Status priority: finalized / revised figures are preferred over preliminary ones
# for the *same* year, but we never blindly prefer a newer low-quality source.
_STATUS_PRIORITY: dict[TemporalQualifier, float] = {
    TemporalQualifier.ACTUAL: 1.0,
    TemporalQualifier.REVISED: 0.98,
    TemporalQualifier.PRELIMINARY: 0.85,
    TemporalQualifier.ADVANCE: 0.82,
    TemporalQualifier.ESTIMATE: 0.80,
    TemporalQualifier.PROJECTED: 0.70,
    TemporalQualifier.UNKNOWN: 0.90,
}

_QUALITY_ADJUST: dict[SourceQuality, float] = {
    SourceQuality.PRIMARY: 0.08,
    SourceQuality.SECONDARY: 0.0,
    SourceQuality.TERTIARY: -0.12,
    SourceQuality.UNKNOWN: 0.0,
}


def metric_match(query_metric: MetricType, ev_metric: MetricType) -> float:
    """Fit between the query's target metric and an evidence item's metric.

    1.0 when matching; shares get a partial credit when both are share-types but
    distinct (GVA_SHARE vs OUTPUT_SHARE are related yet different); otherwise low.
    """
    if query_metric in (MetricType.UNKNOWN, MetricType.OTHER) or ev_metric in (MetricType.UNKNOWN, MetricType.OTHER):
        return 1.0
    if query_metric == ev_metric:
        return 1.0
    share_types = {MetricType.GVA_SHARE, MetricType.OUTPUT_SHARE}
    if query_metric in share_types and ev_metric in share_types:
        return 0.35  # related but distinct
    return 0.15


def temporal_recency(ev: Evidence, reference_year: int | None = None) -> float:
    """Recency score in [0, 1]; newer years score higher (half-life ~3 years)."""
    year = parse_year(ev.year_period)
    if year is None:
        return 0.5
    ref = reference_year or 2025
    age = max(0, ref - year)
    return math.exp(-age / 3.0)


def status_priority(tq: TemporalQualifier) -> float:
    return _STATUS_PRIORITY.get(tq, 0.90)


def combined_score(ev: Evidence, classification: QueryClassification | None = None) -> float:
    """Weighted, classification-aware ranking score in [0, 1]."""
    rerank = ev.rerank_score if ev.rerank_score is not None else ev.retrieval_score

    query_geo = classification.geography if classification else ""
    query_metric = classification.metric_hint if classification else MetricType.UNKNOWN

    geo = _geo_match(query_geo, ev.geography)
    metric = metric_match(query_metric, ev.metric_type)

    recency = temporal_recency(ev)

    base = (
        ev.retrieval_score * 0.15
        + rerank * 0.25
        + ev.authority_score * 0.25
        + recency * 0.15
        + geo * 0.10
        + metric * 0.10
    )

    base += _QUALITY_ADJUST.get(ev.source_quality, 0.0)

    # Hard penalty for mismatched geography/metric even when semantic similarity
    # is high -- this is what stops an irrelevant-but-similar source from ranking.
    if geo < 0.5:
        base *= 0.5
    if metric < 0.5:
        base *= 0.7

    return max(0.0, min(1.0, base))


def rank_evidence(
    evidence: list[Evidence],
    classification: QueryClassification | None = None,
) -> list[Evidence]:
    """Return evidence sorted by combined score (highest first)."""
    scored = []
    for ev in evidence:
        ev.combined_score = combined_score(ev, classification)
        scored.append(ev)
    scored.sort(key=lambda e: e.combined_score, reverse=True)
    return scored


def select_latest_per_key(evidence: list[Evidence]) -> dict[tuple[str, str], Evidence]:
    """Pick the best (latest + most authoritative) evidence per (metric, geography).

    Used by the verifier/synthesis to choose the 'latest' figure when several
    periods exist. Newer years and more finalized statuses win; authority breaks ties.
    """
    best: dict[tuple[str, str], Evidence] = {}
    for ev in evidence:
        key = (ev.metric_type.value, ev.geography.lower())
        if key == ("unknown", ""):
            continue
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = ev
            continue
        # Compare by year, then status priority, then authority.
        y_new, y_old = parse_year(ev.year_period), parse_year(incumbent.year_period)
        if y_new != y_old:
            if y_new is None or (y_old is not None and y_new < y_old):
                continue
            best[key] = ev
            continue
        # Same year: prefer more finalized status, then authority.
        if (status_priority(ev.temporal_qualifier), ev.authority_score) > (
            status_priority(incumbent.temporal_qualifier),
            incumbent.authority_score,
        ):
            best[key] = ev
    return best
