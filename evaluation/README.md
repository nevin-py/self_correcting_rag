# RAG Evaluation (DeepEval)

Lightweight evaluation of the self-correcting RAG pipeline using [DeepEval](https://docs.confident-ai.com/).

## What It Does

Runs the **existing LangGraph pipeline** (`rag_app.ainvoke`) end-to-end for each test case, then scores the output with three LLM-as-judge metrics:

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the user's question? |
| **Contextual Relevancy** | Is the retrieved context relevant to the question? |

No separate RAG pipeline or database is created — evaluation calls your live graph directly.

## Install

```bash
pip install deepeval
```

## Configuration

The judge LLM uses **OpenRouter** (reads `OPENROUTER_API_KEY` from your `.env`).

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENROUTER_API_KEY` | *(from .env)* | Required — used for both the RAG pipeline and DeepEval judge |
| `DEEPEVAL_MODEL` | `xiaomi/mimo-v2.5` | Judge LLM model (OpenRouter model ID) |
| `DEEPEVAL_THRESHOLD` | `0.3` | Pass/fail threshold for all metrics |

## Run

```bash
# All 18 cases (expect ~35-50 min)
deepeval test run evaluation/test_rag.py

# Or via pytest
pytest evaluation/test_rag.py -v

# Smoke test (first 3 cases)
pytest evaluation/test_rag.py -v -x -k "mh-gsdp-services-share or india-gdp-national or insufficient-evidence"

# Single case
pytest evaluation/test_rag.py -v -k "constant-prices"

# List available test cases
pytest evaluation/test_rag.py -v --co
```

## Output

Each case prints verbose results during the run:

```
  [1/18] ✅ PASS  mh-gsdp-services-share  (45.2s)
         Query    : What share of Maharashtra GSDP comes from the services sector?
         Answer   : Maharashtra services contributed 64.27% of GSDP in 2024-25…
         ✓ faithfulness              0.85  All claims grounded in context
         ✓ answerrelevancy           0.92  Directly addresses the question
         ✓ contextualrelevancy       0.78  Most context is relevant

  [2/18] ❌ FAIL  constant-prices  (120.3s)
         Query    : Report Maharashtra GSDP growth at constant prices.
         Answer   : I don't have enough reliable information…
         ✗ faithfulness              0.00
         ✗ answerrelevancy           0.00
         ✗ contextualrelevancy       0.00
```

## Results File

After each run, results are saved to `evaluation/results.json`:

```json
{
  "timestamp": "2026-08-14T14:36:37Z",
  "judge_model": "xiaomi/mimo-v2.5",
  "threshold": 0.3,
  "total_cases": 18,
  "passed": 12,
  "failed": 6,
  "avg_faithfulness": 0.72,
  "avg_answer_relevancy": 0.68,
  "avg_contextual_relevancy": 0.55,
  "cases": [...]
}
```

View results after a run:

```bash
# Pretty-print summary
python -c "import json; d=json.load(open('evaluation/results.json')); print(f'Passed: {d[\"passed\"]}/{d[\"total_cases\"]}'); print(f'Faithfulness: {d[\"avg_faithfulness\"]:.2f}'); print(f'Relevancy: {d[\"avg_answer_relevancy\"]:.2f}'); print(f'Context: {d[\"avg_contextual_relevancy\"]:.2f}')"

# Full detail
cat evaluation/results.json | python -m json.tool
```

## Dataset

`evaluation/dataset.json` contains 18 test cases sourced from the existing golden set (`evals/golden_set.json`). Each case has:

- `id` — unique identifier
- `query` — the user question
- `context` — reference context
- `expected_answer` — ground truth answer

### Adding a New Test Case

Append to `dataset.json`:

```json
{
  "id": "my-new-case",
  "query": "What is the question?",
  "context": "Relevant context text.",
  "expected_answer": "The expected answer."
}
```

No code changes needed — the test file loads the dataset automatically.

## Cost Control

- Each test case makes ~4 LLM calls (1 graph run + 3 metric evaluations)
- 18 cases ≈ 72 LLM calls — use `-k` to run fewer for smoke tests
- The graph run uses your configured provider (Groq/OpenRouter/Google) with fallback
- DeepEval judge uses `xiaomi/mimo-v2.5` (free on OpenRouter)
