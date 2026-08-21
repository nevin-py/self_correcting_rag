"""
DeepEval evaluation for the self-correcting RAG pipeline.

Uses OpenRouter as the judge LLM (OPENROUTER_API_KEY from .env).

Usage:
    # All 18 cases
    deepeval test run evaluation/test_rag.py

    # Smoke test (5 cases)
    pytest evaluation/test_rag.py -v -x -k "0 or 1 or 2 or 3 or 4"

    # Single case
    pytest evaluation/test_rag.py -v -k "constant-prices"

    # List cases
    pytest evaluation/test_rag.py -v --co
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Env bootstrap ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

_or_key = os.getenv("OPENROUTER_API_KEY", "")
if _or_key:
    os.environ["OPENAI_API_KEY"] = _or_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# ── DeepEval imports ──────────────────────────────────────────────────────────
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def _load_cases() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("cases", data)


CASES = _load_cases()

# ── Judge model ───────────────────────────────────────────────────────────────
_METRIC_MODEL = os.getenv("DEEPEVAL_MODEL", "xiaomi/mimo-v2.5")
_THRESHOLD = float(os.getenv("DEEPEVAL_THRESHOLD", "0.3"))

# ── Global results collector ──────────────────────────────────────────────────
_eval_results: list[dict] = []


# ── RAG runner ────────────────────────────────────────────────────────────────

def _run_rag(query: str) -> dict:
    """Call the existing LangGraph pipeline synchronously."""
    from app.agent.graph import rag_app, _new_state

    # Use the user who has SQuAD passages ingested via setup_ingest.py
    USER_ID = uuid.UUID("aabbccdd-1122-3344-5566-778899001122")
    CHAT_ID = uuid.UUID("11223344-5566-7788-99aa-bbccddeeff00")

    state = _new_state(
        query=query,
        user_id=USER_ID,
        chat_id=CHAT_ID,
        provider="groq",
    )

    # Patch DB-touching helpers + disable Gemini (quota exhausted, retries burn 90s/model).
    import app.documents.clients as _clients
    _orig_google_alts = _clients._google_alt_llms
    _clients._google_alt_llms = []
    with patch("app.core.usage.enforce_tavily_budget", new_callable=AsyncMock), \
         patch("app.core.usage.record_usage", new_callable=AsyncMock), \
         patch("app.core.usage.enforce_query_rate", new_callable=AsyncMock), \
         patch("app.core.usage.enforce_ingest_budget", new_callable=AsyncMock), \
         patch("app.documents.clients.google_planner_llm", None), \
         patch("app.documents.clients.google_generator_llm", None), \
         patch("app.documents.clients.google_hallucination_llm", None):
        try:
            final = asyncio.run(rag_app.ainvoke(state))
        except Exception as exc:
            import logging
            logging.getLogger("evaluation").warning("Graph failed: %s", exc)
            return {"answer": "", "context": "", "evidence": []}
        finally:
            _clients._google_alt_llms = _orig_google_alts

    answer = final.get("answer", "") or ""
    evidence = final.get("evidence", [])
    context_parts = []
    for ev in evidence:
        text = getattr(ev, "text", "") or ""
        if text.strip():
            context_parts.append(text.strip())
    if not context_parts:
        assembled = final.get("assembled_context", "") or ""
        if assembled.strip():
            context_parts.append(assembled.strip())
    context = "\n\n".join(context_parts)
    return {"answer": answer, "context": context, "evidence": context_parts}


# ── Verbose reporting ─────────────────────────────────────────────────────────

def _print_header():
    print("\n" + "=" * 70)
    print("  SELF-CORRECTING RAG — EVALUATION BENCHMARK")
    print("=" * 70)
    print(f"  Cases       : {len(CASES)}")
    print(f"  Judge model : {_METRIC_MODEL}")
    print(f"  Threshold   : {_THRESHOLD}")
    print(f"  Results     : {RESULTS_PATH}")
    print(f"  Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70 + "\n")


def _print_case_result(idx: int, case: dict, answer: str, scores: dict, elapsed: float, passed: bool):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  [{idx + 1}/{len(CASES)}] {status}  {case['id']}  ({elapsed:.1f}s)")
    print(f"         Query    : {case['query'][:80]}")
    print(f"         Answer   : {answer[:120]}{'…' if len(answer) > 120 else ''}")
    for name, data in scores.items():
        score = data.get("score")
        reason = (data.get("reason") or "")[:100]
        met = "✓" if data.get("passed") else "✗"
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"         {met} {name:25s} {score_str}  {reason}")
    print()


def _save_results():
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": _METRIC_MODEL,
        "threshold": _THRESHOLD,
        "total_cases": len(_eval_results),
        "passed": sum(1 for r in _eval_results if r["passed"]),
        "failed": sum(1 for r in _eval_results if not r["passed"]),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevancy": _avg("answer_relevancy"),
        "avg_contextual_relevancy": _avg("contextual_relevancy"),
        "cases": _eval_results,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def _avg(metric_name: str) -> float:
    scores = [r["scores"][metric_name]["score"] for r in _eval_results if metric_name in r.get("scores", {}) and r["scores"][metric_name].get("score") is not None]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _print_summary(summary: dict):
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total passed  : {summary['passed']}/{summary['total_cases']}")
    print(f"  Total failed  : {summary['failed']}/{summary['total_cases']}")
    print()
    print(f"  Avg Faithfulness          : {summary['avg_faithfulness']:.4f}")
    print(f"  Avg Answer Relevancy      : {summary['avg_answer_relevancy']:.4f}")
    print(f"  Avg Contextual Relevancy  : {summary['avg_contextual_relevancy']:.4f}")
    print()
    if summary["failed"] > 0:
        print("  Failed cases:")
        for r in _eval_results:
            if not r["passed"]:
                score_vals = [v for v in r["scores"].values() if v.get("score") is not None]
                if score_vals:
                    worst = min(score_vals, key=lambda x: x["score"])
                    print(f"    ❌ {r['case_id']:35s}  worst: {worst['score']:.2f} ({worst.get('metric', 'n/a')})")
                else:
                    print(f"    ❌ {r['case_id']:35s}  (no scores)")
    print()
    print(f"  Results saved to: {RESULTS_PATH}")
    print("=" * 70 + "\n")


# ── Pytest setup/teardown ────────────────────────────────────────────────────

def pytest_configure(config):
    _print_header()


def pytest_sessionfinish(session, exitstatus):
    try:
        if _eval_results:
            summary = _save_results()
            _print_summary(summary)
    except Exception as exc:
        print(f"\n  ⚠️  Could not save results: {exc}")
        # Still try to print what we have
        if _eval_results:
            print(f"  Collected {len(_eval_results)} results")
            for r in _eval_results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"    {status} {r['case_id']}")


# ── Pytest parametrize ────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_rag(case: dict):
    """Evaluate one RAG test case with DeepEval metrics."""
    t0 = time.perf_counter()
    result = _run_rag(case["query"])
    answer = result["answer"]
    context = result["context"]
    rag_elapsed = time.perf_counter() - t0

    retrieval_context = result["evidence"] if result["evidence"] else [context] if context else []

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=answer,
        retrieval_context=retrieval_context,
        expected_output=case.get("expected_answer"),
    )

    metrics = [
        FaithfulnessMetric(model=_METRIC_MODEL, threshold=_THRESHOLD, include_reason=True),
        AnswerRelevancyMetric(model=_METRIC_MODEL, threshold=_THRESHOLD, include_reason=True),
        ContextualRelevancyMetric(model=_METRIC_MODEL, threshold=_THRESHOLD, include_reason=True),
    ]

    # Collect and display scores BEFORE asserting
    scores = {}
    all_passed = True
    for m in metrics:
        name = m.__class__.__name__.replace("Metric", "").replace(" ", "_")
        key = name.lower()
        passed = m.is_successful()
        if not passed:
            all_passed = False
        scores[key] = {
            "metric": name,
            "score": m.score if m.score is not None else 0.0,
            "threshold": m.threshold,
            "passed": passed,
            "reason": m.reason or "",
        }

    elapsed = time.perf_counter() - t0
    _print_case_result(CASES.index(case), case, answer, scores, elapsed, all_passed)
    _eval_results.append({
        "case_id": case["id"],
        "query": case["query"],
        "answer": answer[:500],
        "passed": all_passed,
        "rag_latency_s": round(rag_elapsed, 2),
        "total_latency_s": round(elapsed, 2),
        "scores": scores,
    })

    # Save incrementally after each case
    try:
        _save_results()
    except Exception:
        pass

    # Now assert (raises if any metric failed)
    assert_test(test_case, metrics)
