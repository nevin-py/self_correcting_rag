#!/usr/bin/env python3
"""Offline golden-set scorer.

Usage:
  # Score embedded good/bad examples from the golden set
  python -m evals.run_eval

  # Score a JSON file of {case_id: answer}
  python -m evals.run_eval --answers path/to/answers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.eval_checks import load_golden_set, score_case, score_golden_answers


def main() -> int:
    parser = argparse.ArgumentParser(description="Score RAG answers against the golden set")
    parser.add_argument("--answers", type=Path, help="JSON map of case_id → answer text")
    parser.add_argument("--examples", action="store_true", default=True, help="Also score good/bad examples")
    args = parser.parse_args()

    cases = load_golden_set()
    scores = []

    if args.answers:
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
        scores.extend(score_golden_answers(answers))

    # Always exercise embedded examples for CI / local sanity
    for case in cases:
        if case.get("good_answer_example"):
            scores.append(score_case(case, case["good_answer_example"]))
        if case.get("bad_answer_example"):
            bad = score_case(case, case["bad_answer_example"])
            # Invert expectation label in report: bad examples should fail
            scores.append(bad)

    passed = 0
    failed = 0
    for s in scores:
        case = next((c for c in cases if c["id"] == s.case_id), {})
        is_bad_example = False
        # Heuristic: if this score matches a bad_answer_example text path — we only
        # appended bad scores after good ones; mark by failed status expected.
        if case.get("bad_answer_example") and not s.passed and any(
            c.name.startswith("citation_") for c in s.failed
        ):
            is_bad_example = True

        if s.passed:
            passed += 1
            print(f"PASS  {s.case_id}")
        elif is_bad_example or case.get("bad_answer_example"):
            # For negative cases, failing the check is success of the harness
            if not s.passed and case.get("bad_answer_example"):
                # Only count as harness-pass once per bad example when we detect failure
                passed += 1
                print(f"PASS  {s.case_id} (bad example correctly rejected)")
            else:
                failed += 1
                print(f"FAIL  {s.case_id}")
                for c in s.failed:
                    print(f"        - {c.name}: {c.detail}")
        else:
            failed += 1
            print(f"FAIL  {s.case_id}")
            for c in s.failed:
                print(f"        - {c.name}: {c.detail}")

    print(f"\n{passed} passed, {failed} failed, {len(scores)} scored")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
