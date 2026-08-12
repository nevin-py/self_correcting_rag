"""
Offline eval checks for golden RAG cases.

These do not require live LLM/API calls — they score an answer + citations
against expected metric/geo/citation constraints from the golden set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.citation_validator import validate_answer_citations
from app.agent.normalization import detect_metric_type, detect_place_mentions
from app.agent.state import Evidence, MetricType, SourceType

GOLDEN_SET_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden_set.json"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseScore:
    case_id: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or GOLDEN_SET_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("cases", data if isinstance(data, list) else []))


def _evidence_from_case(case: dict) -> list[Evidence]:
    """Build Evidence objects from golden fixture snippets (for citation checks)."""
    out: list[Evidence] = []
    for i, snip in enumerate(case.get("evidence_fixtures") or []):
        eid = snip.get("evidence_id") or f"gold{i:04d}"
        out.append(
            Evidence(
                evidence_id=eid,
                text=snip.get("text", ""),
                source_type=SourceType.DOCUMENT if snip.get("source_type") == "document" else SourceType.WEB,
                source_name=snip.get("source_name", "golden"),
                source_url=snip.get("source_url"),
            )
        )
    return out


def score_case(case: dict, answer: str, *, evidence: list[Evidence] | None = None) -> CaseScore:
    """Score one golden case against a candidate answer."""
    checks: list[CheckResult] = []
    ev = evidence if evidence is not None else _evidence_from_case(case)
    expect = case.get("expect") or {}

    # Non-empty answer
    checks.append(
        CheckResult(
            "non_empty_answer",
            bool(answer and answer.strip()),
            "answer missing" if not (answer and answer.strip()) else "ok",
        )
    )

    # Required substrings / forbidden substrings
    for needle in expect.get("must_contain") or []:
        ok = needle.lower() in (answer or "").lower()
        checks.append(CheckResult(f"must_contain:{needle[:40]}", ok, "found" if ok else "missing"))

    for needle in expect.get("must_not_contain") or []:
        ok = needle.lower() not in (answer or "").lower()
        checks.append(CheckResult(f"must_not_contain:{needle[:40]}", ok, "absent" if ok else "present"))

    # Metric expectation
    metric = expect.get("metric")
    if metric:
        detected = detect_metric_type(answer or "")
        # Also accept explicit acronym presence
        acronym_ok = metric.lower() in (answer or "").lower()
        enum_ok = detected != MetricType.UNKNOWN and detected.value == metric.lower()
        ok = acronym_ok or enum_ok
        checks.append(
            CheckResult(
                f"metric:{metric}",
                ok,
                f"detected={detected.value}" if detected != MetricType.UNKNOWN else ("acronym" if acronym_ok else "missing"),
            )
        )
        # Must not conflate with forbidden metrics
        for bad in expect.get("forbid_metrics") or []:
            # Allow mentioning the textbook/old figure contextually if must_contain needs it;
            # only fail if bad metric is asserted as the *answer* metric without the good one nearby.
            if bad.lower() == metric.lower():
                continue
            # Soft: fail if answer talks about bad metric as the primary figure without good metric
            bad_hits = len(re.findall(rf"\b{re.escape(bad)}\b", answer or "", flags=re.I))
            good_hits = len(re.findall(rf"\b{re.escape(metric)}\b", answer or "", flags=re.I))
            ok_bad = not (bad_hits > good_hits and bad_hits > 0)
            checks.append(
                CheckResult(
                    f"forbid_metric_dominance:{bad}",
                    ok_bad,
                    f"bad={bad_hits} good={good_hits}",
                )
            )

    # Geography
    geo = expect.get("geography")
    if geo:
        places = detect_place_mentions(answer or "")
        ok = geo.lower() in (answer or "").lower() or any(geo.lower() in p.lower() for p in places)
        checks.append(CheckResult(f"geography:{geo}", ok, "found" if ok else "missing"))

    # Citation requirements
    if expect.get("require_citations", True) and ev:
        validation = validate_answer_citations(answer or "", ev, require_citations=True)
        max_uncited = int(expect.get("max_uncited_assertions", 0))
        uncited_ok = len(validation.uncited_sentences) <= max_uncited
        checks.append(
            CheckResult(
                "citation_uncited_budget",
                uncited_ok,
                f"uncited={len(validation.uncited_sentences)} max={max_uncited}",
            )
        )
        invalid_ok = len(validation.invalid_citation_ids) == 0
        checks.append(
            CheckResult(
                "citation_ids_resolve",
                invalid_ok,
                f"invalid={validation.invalid_citation_ids}",
            )
        )
        if expect.get("min_citations", 0):
            n = len(validation.cited_ids)
            ok = n >= int(expect["min_citations"])
            checks.append(CheckResult("min_citations", ok, f"cited={n}"))

    # Year / period hints
    for year in expect.get("years") or []:
        ok = str(year) in (answer or "")
        checks.append(CheckResult(f"year:{year}", ok, "found" if ok else "missing"))

    passed = all(c.passed for c in checks)
    return CaseScore(case_id=str(case.get("id", "unknown")), passed=passed, checks=checks)


def score_golden_answers(
    answers: dict[str, str],
    *,
    path: Path | None = None,
) -> list[CaseScore]:
    """Score a mapping of case_id → answer text against the golden set."""
    cases = {c["id"]: c for c in load_golden_set(path)}
    scores: list[CaseScore] = []
    for case_id, answer in answers.items():
        case = cases.get(case_id)
        if not case:
            scores.append(
                CaseScore(
                    case_id=case_id,
                    passed=False,
                    checks=[CheckResult("case_exists", False, "unknown case id")],
                )
            )
            continue
        scores.append(score_case(case, answer))
    return scores
