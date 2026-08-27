#!/usr/bin/env python3
"""RAGAS evaluation of the pipeline's answer quality.

Standard RAGAS metrics over the live cases:
  - faithfulness        : is every claim in the answer supported by the evidence?
  - answer_relevancy    : does the answer address the question?
  - context_precision   : is the retrieved evidence ranked usefully?
  - answer_correctness  : factual agreement with a reference answer

History: this script was removed after every free-tier judge failed silently —
Groq/Gemini free tiers throttled RAGAS's per-sample fan-out into TimeoutErrors
that degrade to NaN, and reasoning judges burn their completion budget before
emitting parseable JSON. It is viable again because the default judge is now a
PAID, non-reasoning model (xiaomi/mimo-v2.5 at ~$0.14/M prompt): a full run is
~300 sub-calls ≈ $0.04. Do NOT switch back to a free-tier or reasoning judge.

Usage:
  python -m evals.ragas_eval                    # full live run, MiMo judge
  python -m evals.ragas_eval --limit 3          # smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Shim: ragas 0.4.x imports a module langchain-community 1.x removed ──────
import types

_vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:  # noqa: D101 — never instantiated by this project
    pass


_vertexai_mod.ChatVertexAI = ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_mod)


class SerializedLLM:
    """Global-lock proxy around a langchain chat model.

    RAGAS metrics fire many concurrent sub-calls per sample (one per evidence
    chunk in context_precision, n=3 generations in relevancy). Even a cheap
    endpoint throttles that fan-out, retries exhaust as TimeoutError, and the
    affected metrics silently become NaN. Serializing both the sync path (used
    from RAGAS's thread pool) and the async path keeps every sub-call alive;
    attribute reads/writes delegate so the wrapper's temperature/n pokes work.
    """

    def __init__(self, inner, limit: int = 1):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tsem", threading.Semaphore(limit))
        object.__setattr__(self, "_asem", None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def generate_prompt(self, *a, **k):
        with self._tsem:
            return _unfence_result(self._inner.generate_prompt(*a, **k))

    async def agenerate_prompt(self, *a, **k):
        if self._asem is None:
            self._asem = asyncio.Semaphore(1)
        async with self._asem:
            return _unfence_result(await self._inner.agenerate_prompt(*a, **k))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _unfence_text(text: str) -> str:
    """Some models intermittently wrap JSON in markdown fences, which breaks
    RAGAS's structured-output parsing (-> silent retries -> job timeouts)."""
    if "```" not in text:
        return text
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _unfence_result(result):
    for gens in getattr(result, "generations", []) or []:
        for gen in gens:
            msg = getattr(gen, "message", None)
            if msg is not None and isinstance(msg.content, str):
                msg.content = _unfence_text(msg.content)
    return result


def build_judge(model: str):
    """Non-reasoning chat model via OpenRouter (or groq:<model>), serialized."""
    from app.core.config import settings

    if model.startswith("groq:"):
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model[len("groq:"):], api_key=settings.GROQ_KEY,
                       temperature=0, reasoning_effort="low")
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
        )
    return SerializedLLM(llm)


def _scoreable_text(answer: str) -> str:
    """Strip presentation artifacts before judging: the Caveats footer restates
    unverified claims as terse bullets (crashes answer_relevancy's reverse-question
    similarity: measured 0.0 -> 0.98 on one case) and [E#] markers are internal
    pointers RAGAS cannot resolve."""
    t = re.sub(r"\s*Caveats:\s*The following points.*$", "", answer, flags=re.DOTALL)
    return re.sub(r"\[E\d+\]", "", t).strip()


def run_pipeline_cases(
    limit: int,
    cases: list[dict] | None = None,
) -> list[dict]:
    """Run the full pipeline per case; return rows for RAGAS.

    ``cases``: optional list of case dicts (id/query/expected_answer). Defaults
    to ``evals/live_cases.json`` when None.
    """
    import logging

    logging.basicConfig(level=logging.WARNING)
    from app.agent import nodes
    from app.agent.graph import create_initial_state, rag_app

    async def fake_docs(queries, state):
        return []

    nodes._retrieve_documents = fake_docs

    # context_precision fires one judge call PER chunk; 12 large chunks ×
    # serialized judging exceeded the per-job timeout and NaN'd the metric on
    # most cases. Top-6 at ≤800 chars keeps precision@k meaningful and makes
    # the metric complete in time.
    TOP_CONTEXTS = 6
    CONTEXT_CHARS = 800

    if cases is None:
        cases = json.loads((ROOT / "evals" / "live_cases.json").read_text())["cases"]
    if limit:
        cases = cases[:limit]

    async def _run(case):
        t0 = time.perf_counter()
        state = create_initial_state(query=case["query"], provider="openrouter")
        final = await rag_app.ainvoke(state)
        evidence = final.get("evidence", []) or []
        row = {
            "user_input": case["query"],
            "response": _scoreable_text(final.get("answer", "")),
            "retrieved_contexts": [ev.text[:CONTEXT_CHARS] for ev in evidence[:TOP_CONTEXTS]],
            "reference": case.get("expected_answer", ""),
            "id": case["id"],
            "latency_s": round(time.perf_counter() - t0, 1),
        }
        print(f"  {case['id']}: {final.get('final_status', '')} ({row['latency_s']}s, ctx={len(evidence)})",
              file=sys.stderr)
        return row

    out = []
    for case in cases:
        try:
            out.append(asyncio.run(_run(case)))
        except Exception as exc:
            print(f"  {case['id']}: ERROR {str(exc)[:120]}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the pipeline")
    parser.add_argument("--judge-model", default="xiaomi/mimo-v2.5",
                        help="non-reasoning judge via OpenRouter (or groq:<model>). "
                             "Free-tier judges throttle to NaN; see module docstring")
    parser.add_argument("--limit", type=int, default=0, help="first N cases (0 = all)")
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    rows = run_pipeline_cases(args.limit)
    rows = [r for r in rows if r["response"]]
    if not rows:
        print("No successful runs to evaluate.")
        return 1

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    # Local MiniLM embeddings (same family Chroma ships) — no external API needed.
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    from langchain_core.embeddings import Embeddings

    class LocalMiniLM(Embeddings):
        def __init__(self):
            self._ef = ONNXMiniLM_L6_V2()

        def embed_documents(self, texts):
            return [v for v in self._ef(texts)]

        def embed_query(self, text):
            return self._ef([text])[0]

    judge = LangchainLLMWrapper(build_judge(args.judge_model))

    ds = Dataset.from_list([
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference"],
        }
        for r in rows
    ])

    from ragas.run_config import RunConfig

    # One evaluate() per metric, then merge: running all four in a single
    # evaluate() intermittently drops answer_correctness from the result frame
    # entirely (no error, no column). Isolated runs never fail. Same total
    # judge calls either way since SerializedLLM serializes them regardless.

    # max_workers MUST stay 1 for full 12-case runs: context_precision /
    # answer_correctness fan out per-chunk/per-generation sub-calls that trip
    # endpoint rate limits even through SerializedLLM (24 silent job failures
    # at workers=2 vs 0 at workers=1). Cost is runtime only (~65 min).
    run_config = RunConfig(timeout=480, max_retries=8, max_wait=120, max_workers=1)
    parts = []

    for metric in [faithfulness, answer_relevancy, context_precision, answer_correctness]:
        name = metric.name if hasattr(metric, "name") else type(metric).__name__
        print(f"Evaluating {len(rows)} answers · {name} (judge={args.judge_model}) ...",
              file=sys.stderr)
        part = evaluate(
            ds,
            metrics=[metric],
            llm=judge,
            embeddings=LocalMiniLM(),
            run_config=run_config,
        ).to_pandas()
        parts.append(part)

    base = parts[0]
    for part in parts[1:]:
        new_cols = [c for c in part.columns if c not in base.columns]
        base = base.join(part[new_cols])
    df = base
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.judge_model.replace("/", "_").replace(":", "-")
    df.insert(0, "id", [r["id"] for r in rows])
    csv_path = out_dir / f"{stamp}_ragas_{tag}.csv"
    df.to_csv(csv_path, index=False)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
    present = [m for m in metric_cols if m in df.columns]
    means = {m: round(float(df[m].mean()), 3) for m in present}
    nans = {m: int(df[m].isna().sum()) for m in present}
    missing = [m for m in metric_cols if m not in df.columns]

    print("\n=== RAGAS scores (current architecture) ===")
    for k, v in means.items():
        print(f"  {k:<20} {v}  ({nans[k]} NaN of {len(df)})")
    for m in missing:
        print(f"  {m:<20} MISSING (metric produced no column this run)")

    summary = {"judge": args.judge_model, "cases": len(df),
               "means": means, "nan_counts": nans,
               "missing_metrics": missing,
               "per_case": df[["id"] + present].to_dict("records")}
    (out_dir / f"{stamp}_ragas_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nPer-case CSV: {csv_path}")

    # Exit 2 only when the metric we trust (faithfulness) lost calls to NaN or
    # didn't run; other metrics degrade under chunk-fan-out timeouts and their
    # NaNs are reported, not fatal, so a flaky precision pass never discards an
    # otherwise good run.
    ok = "faithfulness" in present and nans.get("faithfulness", 1) == 0
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
