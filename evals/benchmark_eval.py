#!/usr/bin/env python3
"""Public-benchmark evaluation of the self-correcting RAG pipeline.

Dataset: TruthfulQA generation (HuggingFace `truthfulqa/truthful_qa`), the
open-ended factual-QA benchmark that measures whether a system repeats common
misconceptions vs answers truthfully — a direct test of this project's thesis
("verify claims against evidence, never silently guess").

Unlike RAGAS, the judge is a FREE OpenRouter model so the benchmark can be run
at zero LLM cost. Rate limits are respected: judge calls are serialized via the
shared SerializedLLM proxy and `--workers 1`.

Usage:
  python -m evals.benchmark_eval --limit 30                      # free judge
  python -m evals.benchmark_eval --limit 30 --judge-model <model>
  python -m evals.benchmark_eval --limit 30 --free-model gemma-4-31b:free
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same vertexai shim ragas_0.4 needs.
_vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:  # noqa: D101
    pass


_vertexai_mod.ChatVertexAI = ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_mod)

# Free OpenRouter models that reliably follow structured output + factual
# grading. Order = preference for the free path (strongest instruction following
# and factual-grading first). Fallback walks the list until one is available.
FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m2.7:free",
]


@functools.lru_cache(maxsize=1)
def _load_truthfulqa(n: int) -> list[dict]:
    """Load the TruthfulQA generation/validation split, first ``n`` rows."""
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = []
    for i in range(min(n, len(ds))):
        rows.append({
            "id": f"tqa-{i}",
            "query": ds[i]["question"],
            # TruthfulQA has no single expected answer; the REFERENCE is the
            # model's "best answer" that we grade the pipeline against. Leave
            # empty — RAGAS faithfulness needs contexts, not references.
            "expected_answer": "",
        })
    return rows


def build_judge(model: str | None):
    """Structured-output judge on OpenRouter. If ``model`` is None, try the
    free models in order until one accepts a structured call."""
    from langchain_openai import ChatOpenAI

    from app.core.config import settings

    def _client(m: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=m,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={"HTTP-Referer": "https://self-correcting-rag.local"},
        )

    if model:
        return _client(model)

    # Free tier: first model that completes a structured call.
    from pydantic import BaseModel

    class Probe(BaseModel):
        ok: bool

    last = None
    for m in FREE_MODELS:
        try:
            c = _client(m)
            c.with_structured_output(Probe).invoke("Respond with ok=true.")
            print(f"  free judge picked: {m}", file=sys.stderr)
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise RuntimeError(f"no free OpenRouter model accepted a structured call: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TruthfulQA benchmark for the pipeline")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--judge-model", default=None,
                        help="override judge model (default: probe free models)")
    parser.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = parser.parse_args()

    cases = _load_truthfulqa(args.limit)
    print(f"Loaded {len(cases)} TruthfulQA questions.", file=sys.stderr)

    import evals.ragas_eval as rag

    rows = rag.run_pipeline_cases(args.limit, cases=cases)
    rows = [r for r in rows if r["response"]]
    print(f"Pipeline produced {len(rows)} answers.", file=sys.stderr)
    if not rows:
        return 1

    # Reuse RAGAS faithfulness on the free judge (only the metric our gate trusts).
    from langchain_core.embeddings import Embeddings
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class MiniLM(Embeddings):
        def __init__(self):
            self._ef = ONNXMiniLM_L6_V2()

        def embed_documents(self, texts):
            return [v for v in self._ef(texts)]

        def embed_query(self, t):
            return self._ef([t])[0]

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import faithfulness
    from ragas.run_config import RunConfig

    judge_llm = build_judge(args.judge_model)
    ds = Dataset.from_list([
        {"user_input": r["user_input"], "response": r["response"],
         "retrieved_contexts": r["retrieved_contexts"], "reference": r["reference"]}
        for r in rows
    ])
    res = evaluate(
        ds, metrics=[faithfulness], llm=LangchainLLMWrapper(judge_llm),
        embeddings=MiniLM(), run_config=RunConfig(timeout=480, max_retries=6,
                                                  max_wait=120, max_workers=1),
    ).to_pandas()
    f = float(res["faithfulness"].mean())
    nan = int(res["faithfulness"].isna().sum())
    print(f"\n=== TruthfulQA faithfulness (free model) ===\n  {f:.3f}  ({nan}/{len(rows)} NaN)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"{stamp}_truthfulqa_faithfulness.json").write_text(
        json.dumps({
            "dataset": "truthfulqa/truthful_qa",
            "judge": args.judge_model or "free-probe",
            "say": len(rows), "faithfulness_mean": f, "nan": nan,
            "per_case": res[["faithfulness"]].round(3).to_dict("records"),
        }, indent=1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())