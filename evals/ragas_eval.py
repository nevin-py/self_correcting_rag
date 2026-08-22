#!/usr/bin/env python3
"""RAGAS evaluation of the pipeline's answer quality.

Standard RAGAS metrics over the live cases:
  - faithfulness        : is every claim in the answer supported by the evidence?
  - answer_relevancy    : does the answer address the question?
  - context_precision   : is the retrieved evidence ranked usefully?
  - answer_correctness  : factual agreement with a reference answer
                          (temporal cases will lag the judge model's cutoff — read
                          them together with the per-case notes)

Usage:
  python -m evals.ragas_eval --model qwen/qwen3-30b-a3b-instruct-2507 --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def run_pipeline_cases(model: str | None, limit: int) -> list[dict]:
    """Run the full pipeline per case; return rows for RAGAS."""
    if model:
        os.environ["OPENROUTER_PLANNER_MODEL"] = model
        os.environ["OPENROUTER_GENERATOR_MODEL"] = model
        os.environ["OPENROUTER_HALLUCINATION_MODEL"] = model

    import asyncio
    import logging

    logging.basicConfig(level=logging.WARNING)
    from app.agent import nodes
    from app.agent.graph import create_initial_state, rag_app

    async def fake_docs(queries, state):
        return []

    nodes._retrieve_documents = fake_docs

    cases = json.loads((ROOT / "evals" / "live_cases.json").read_text())["cases"][:limit]

    async def _run(case):
        t0 = time.perf_counter()
        state = create_initial_state(query=case["query"], provider="openrouter")
        final = await rag_app.ainvoke(state)
        elapsed = time.perf_counter() - t0
        evidence = final.get("evidence", []) or []
        return {
            "user_input": case["query"],
            "response": final.get("answer", ""),
            "retrieved_contexts": [ev.text for ev in evidence],
            "reference": case.get("expected_answer", ""),
            "id": case["id"],
            "grading": case.get("grading", "reference"),
            "final_status": final.get("final_status", ""),
            "latency_s": round(time.perf_counter() - t0, 1),
        }

    out = []
    for case in cases:
        try:
            row = asyncio.run(_run(case))
            print(f"  {case['id']}: {row['final_status']} ({row['latency_s']}s)", file=sys.stderr)
            out.append(row)
        except Exception as exc:
            print(f"  {case['id']}: ERROR {str(exc)[:120]}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the pipeline")
    parser.add_argument("--model", default=None, help="OpenRouter primary model override")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    rows = run_pipeline_cases(args.model, args.limit)
    rows = [r for r in rows if r["response"]]
    if not rows:
        print("No successful runs to evaluate.")
        return 1

    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        faithfulness,
        context_precision,
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

    from app.core.config import settings

    if args.judge_model.startswith("groq:"):
        # Groq-hosted models: fast, non-reasoning, strict-schema compliant —
        # the reliable choice for RAGAS's multi-step structured prompts.
        from langchain_groq import ChatGroq

        judge_llm = ChatGroq(
            model=args.judge_model[len("groq:"):],
            api_key=settings.GROQ_KEY,
            temperature=0,
        )
    else:
        judge_llm = ChatOpenAI(
            model=args.judge_model,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
        )

    judge = LangchainLLMWrapper(judge_llm)

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

    # The strong judge is slow; without generous timeouts its sub-calls fail
    # silently and metrics degrade to NaN / zero-credit rows.
    run_config = RunConfig(timeout=240, max_retries=8, max_wait=120)

    print(f"Evaluating {len(rows)} answers with RAGAS (judge={args.judge_model}) ...", file=sys.stderr)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
        llm=judge,
        embeddings=LocalMiniLM(),
        run_config=run_config,
    )

    df = result.to_pandas()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = (args.model or "default").replace("/", "_")
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "grading", [r["grading"] for r in rows])
    csv_path = out_dir / f"{stamp}_ragas_{tag}.csv"
    df.to_csv(csv_path, index=False)

    means = {
        m: round(float(df[m].mean()), 3)
        for m in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness")
        if m in df.columns
    }
    print("\n=== RAGAS scores (current architecture) ===")
    for k, v in means.items():
        print(f"  {k:<20} {v}")
    print(f"\nPer-case CSV: {csv_path}")

    summary = {"model": args.model or "default", "means": means,
               "per_case": df[["id", "faithfulness", "answer_relevancy",
                               "context_precision", "answer_correctness"]].to_dict("records")}
    (out_dir / f"{stamp}_ragas_{tag}.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
