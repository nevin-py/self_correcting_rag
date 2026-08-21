#!/usr/bin/env python3
"""Offline golden-set scorer.

Runs each golden case's evidence fixtures through citation validation and
(optional) live pipeline scoring. No deleted heuristic modules involved.

Usage:
  # Static checks: good answers must pass, bad answers must fail
  python -m evals.run_eval

  # Score a JSON file of case_id → answer text
  python -m evals.run_eval --answers path/to/answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

from app.agent.citation_validator import validate_answer_citations
from app.agent.state import Evidence, SourceType


def load_golden_set() -> list[dict]:
    data = json.loads((ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    return data.get("cases", [])


def _fixture_evidence(case: dict) -> list[Evidence]:
    return [
        Evidence(
            evidence_id=f["evidence_id"],
            text=f["text"],
            source_type=SourceType.DOCUMENT if f.get("source_type") == "document" else SourceType.WEB,
            source_name=f.get("source_name", ""),
        )
        for f in case.get("evidence_fixtures", [])
    ]


def score_answer(case: dict, answer: str) -> tuple[bool, list[str]]:
    """An answer passes when every factual assertion cites valid fixture evidence."""
    result = validate_answer_citations(answer, _fixture_evidence(case))
    failures = [e.detail for e in result.errors]
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Score RAG answers against the golden set")
    parser.add_argument("--answers", type=Path, help="JSON map of case_id → answer text")
    args = parser.parse_args()

    cases = load_golden_set()
    passed = failed = 0

    if args.answers:
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
        for case in cases:
            answer = answers.get(case["id"])
            if answer is None:
                continue
            ok, failures = score_answer(case, answer)
            if ok:
                passed += 1
                print(f"PASS  {case['id']}")
            else:
                failed += 1
                print(f"FAIL  {case['id']}")
                for f in failures:
                    print(f"        - {f}")
    else:
        for case in cases:
            good = case.get("good_answer_example")
            bad = case.get("bad_answer_example")
            if good:
                ok, failures = score_answer(case, good)
                if ok:
                    passed += 1
                    print(f"PASS  {case['id']} (good example accepted)")
                else:
                    failed += 1
                    print(f"FAIL  {case['id']} (good example rejected)")
                    for f in failures:
                        print(f"        - {f}")
            if bad:
                ok, _ = score_answer(case, bad)
                if not ok:
                    passed += 1
                    print(f"PASS  {case['id']} (bad example correctly rejected)")
                else:
                    failed += 1
                    print(f"FAIL  {case['id']} (bad example wrongly accepted)")

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
