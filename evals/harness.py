#!/usr/bin/env python3
"""Live multi-model eval harness.

Runs each case through the FULL pipeline (real web search, real LLMs) once per
model configuration, then grades every answer with an LLM judge and reports a
comparison table: correctness score, citation coverage, caveat rate, latency,
and estimated tokens/cost (from the llm_call_traces recorded during the run).

Usage:
  # Compare models (each runs in a subprocess so env-config takes effect):
  python -m evals.harness --models qwen/qwen3-30b-a3b-instruct-2507,openai/gpt-oss-120b

  # Single model:
  python -m evals.harness --models openai/gpt-oss-120b

Results land in evals/results/<timestamp>_<model>.json and a combined
<timestamp>_comparison.md.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_PATH = ROOT / "evals" / "live_cases.json"


# ── Per-model run (executed in a subprocess per model) ───────────────────────


def run_one(model: str) -> list[dict]:
    """Run all cases with `model` as the OpenRouter primary. Returns case results."""
    os.environ["OPENROUTER_PLANNER_MODEL"] = model
    os.environ["OPENROUTER_GENERATOR_MODEL"] = model
    os.environ["OPENROUTER_HALLUCINATION_MODEL"] = model

    # Import AFTER env so clients build with the target model.
    import asyncio
    import logging

    logging.basicConfig(level=logging.WARNING)
    from app.agent import nodes
    from app.agent.graph import create_initial_state, rag_app

    async def fake_docs(queries, state):
        return []  # evals run web-only; no local KB

    nodes._retrieve_documents = fake_docs

    async def _run_case(case: dict) -> dict:
        t0 = time.perf_counter()
        state = create_initial_state(query=case["query"], provider="openrouter")
        final = await rag_app.ainvoke(state)
        elapsed = time.perf_counter() - t0
        claims = final.get("claims", []) or []
        answer = final.get("answer", "")
        return {
            "id": case["id"],
            "query": case["query"],
            "answer": answer[:2000],
            "final_status": final.get("final_status", ""),
            "latency_s": round(elapsed, 1),
            "claims_total": len(claims),
            "claims_verified": sum(1 for c in claims if getattr(c, "status", None) and c.status.value == "verified"),
            "claims_unverified": sum(1 for c in claims if getattr(c, "status", None) and c.status.value == "unverified"),
        }

    cases = json.loads(CASES_PATH.read_text())["cases"]
    results = []
    for case in cases:
        try:
            results.append(asyncio.run(_run_case(case)))
            print(f"  ran {case['id']}: {results[-1]['final_status']} ({results[-1]['latency_s']}s)", file=sys.stderr)
        except Exception as exc:
            results.append({"id": case["id"], "query": case["query"], "error": str(exc)[:300]})
            print(f"  ran {case['id']}: ERROR {str(exc)[:120]}", file=sys.stderr)
    return results


# ── Judge scoring ────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are grading an AI research assistant's answer.

Question: {query}

Reference answer (ground truth): {expected}

Assistant's answer: {answer}

Grade the assistant's answer from 0 to 5:
- 5: correct, complete, and directly answers the question
- 3-4: mostly correct, minor omissions or imprecision
- 1-2: partially wrong or misses the core of the question
- 0: wrong, evasive, or refuses without justification

Respond with ONLY a JSON object: {{"score": <0-5>, "reason": "<one sentence>"}}"""


def judge_answers(model: str, results: list[dict]) -> None:
    """Attach judge scores to each result using a fixed strong model."""
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    from app.core.config import settings

    class Grade(BaseModel):
        score: int
        reason: str = ""

    cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text())["cases"]}
    llm = ChatOpenAI(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
    )
    for r in results:
        if r.get("error") or not r.get("answer"):
            r["judge_score"] = 0
            r["judge_reason"] = r.get("error", "empty answer")[:200]
            continue
        case = cases[r["id"]]
        prompt = _JUDGE_PROMPT.format(
            query=case["query"],
            expected=case["expected_answer"],
            answer=r["answer"][:1500],
        )
        try:
            grade = llm.with_structured_output(Grade).invoke(prompt)
            r["judge_score"] = max(0, min(5, int(grade.score)))
            r["judge_reason"] = grade.reason[:200]
        except Exception as exc:
            r["judge_score"] = 0
            r["judge_reason"] = f"judge failed: {str(exc)[:150]}"


def judge_answers(model: str, results: list[dict]) -> None:
    """Attach judge scores to each result.

    Two grading modes:
    - reference: compare against the static expected answer (stable facts).
    - evidence_faithful: for current-events/temporal cases, an LLM judge with a
      knowledge cutoff cannot know the ground truth — grade whether the answer
      is internally consistent with the sources it cites instead.
    """
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    from app.core.config import settings

    class Grade(BaseModel):
        score: int
        reason: str = ""

    cases = {c["id"]: c for c in json.loads(CASES_PATH.read_text())["cases"]}
    llm = ChatOpenAI(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
    )
    for r in results:
        if r.get("error") or not r.get("answer"):
            r["judge_score"] = 0
            r["judge_reason"] = r.get("error", "empty answer")[:200]
            continue
        case = cases[r["id"]]
        if case.get("grading") == "evidence_faithful":
            prompt = (
                "You are grading an AI research assistant's answer to a CURRENT-EVENTS "
                "question. The answer was produced by live web search — your own training "
                "data may be outdated, so do NOT grade against what you remember. Grade "
                "only: (a) does the answer directly address the question, (b) does it cite "
                "specific sources, (c) is it internally consistent and specific (names, "
                "dates, scores)? Score 0-5.\n\n"
                f"Question: {case['query']}\n\nAssistant's answer: {r['answer'][:1500]}\n\n"
                'Respond with ONLY: {"score": <0-5>, "reason": "<one sentence>"}'
            )
        else:
            prompt = _JUDGE_PROMPT.format(
                query=case["query"],
                expected=case["expected_answer"],
                answer=r["answer"][:1500],
            )
        try:
            grade = llm.with_structured_output(Grade).invoke(prompt)
            r["judge_score"] = max(0, min(5, int(grade.score)))
            r["judge_reason"] = grade.reason[:200]
        except Exception as exc:
            r["judge_score"] = 0
            r["judge_reason"] = f"judge failed: {str(exc)[:150]}"


def collect_trace_summary(since_ts: float) -> dict:
    """Aggregate llm_call_traces written during this run (created_at >= since)."""
    import asyncio

    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.core.config import settings

    async def _agg():
        eng = create_async_engine(settings.DATABASE_URL, pool_recycle=10)
        from app.observability.models import LLMCallTrace

        cutoff = datetime.fromtimestamp(since_ts, tz=timezone.utc).replace(tzinfo=None)
        async with eng.connect() as conn:
            rows = (await conn.execute(
                select(
                    LLMCallTrace.model,
                    LLMCallTrace.status,
                    func.count().label("n"),
                    func.avg(LLMCallTrace.latency_ms).label("avg_lat"),
                    func.sum(LLMCallTrace.prompt_tokens_est).label("pt"),
                    func.sum(LLMCallTrace.completion_tokens_est).label("ct"),
                )
                .where(LLMCallTrace.created_at >= cutoff)
                .group_by(LLMCallTrace.model, LLMCallTrace.status)
            )).all()
        await eng.dispose()
        return [dict(r._mapping) for r in rows]

    return asyncio.run(_agg())


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Live multi-model eval harness")
    parser.add_argument("--models", required=True, help="comma-separated model ids")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b", help="fixed judge model")
    parser.add_argument("--run-one", dest="run_one_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Subprocess mode: run cases for one model, print JSON, exit.
    if args.run_one_model:
        results = run_one(args.run_one_model)
        print("@@RESULTS@@" + json.dumps(results))
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_runs: dict[str, dict] = {}

    for model in models:
        print(f"== Running cases with {model} ...", file=sys.stderr)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, __file__, "--models", model, "--run-one", model],
            capture_output=True, text=True, timeout=1800, cwd=str(ROOT),
        )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("@@RESULTS@@"):
                payload = json.loads(line[len("@@RESULTS@@"):])
                break
        if payload is None:
            print(f"  run failed: {proc.stderr[-400:]}", file=sys.stderr)
            continue
        print(f"  judging with {args.judge_model} ...", file=sys.stderr)
        judge_answers(args.judge_model, payload)
        traces = collect_trace_summary(started)
        all_runs[model] = {"results": payload, "traces": traces}
        (out_dir / f"{stamp}_{model.replace('/', '_')}.json").write_text(
            json.dumps(all_runs[model], indent=1)
        )

    # ── Comparison report ──
    lines = ["# Eval comparison", "", f"Judge: `{args.judge_model}` · cases: {len(json.loads(CASES_PATH.read_text())['cases'])}", ""]
    header = f"| {'model':<44} | avg score /5 | verified% | caveats% | errors% | avg latency | tokens (p+c) |"
    lines += [header, "|" + "---|" * 7]
    for model, run in all_runs.items():
        scored = [r["judge_score"] for r in run["results"] if "judge_score" in r]
        n = len(run["results"]) or 1
        caveats = sum(1 for r in run["results"] if r.get("final_status") == "answered_with_caveats")
        errors = sum(1 for r in run["results"] if r.get("error"))
        verified = sum(r.get("claims_verified", 0) for r in run["results"])
        total_claims = sum(r.get("claims_total", 0) for r in run["results"]) or 1
        latencies = [r["latency_s"] for r in run["results"] if "latency_s" in r]
        pt = sum(t.get("pt") or 0 for t in run["traces"])
        ct = sum(t.get("ct") or 0 for t in run["traces"])
        lines.append(
            f"| {model:<44} | {statistics.mean(scored):.2f} | "
            f"{100 * verified / total_claims:.0f}% | {100 * caveats / n:.0f}% | "
            f"{100 * errors / n:.0f}% | {statistics.mean(latencies):.1f}s | {pt + ct:,} |"
        )
    report = "\n".join(lines)
    (out_dir / f"{stamp}_comparison.md").write_text(report + "\n")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
