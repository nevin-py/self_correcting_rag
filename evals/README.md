# Golden eval set

Offline checks for citation grounding, metric naming (GSDP vs GDP, etc.), and geography.

## Files

- `golden_set.json` — 20 cases with queries, synthetic evidence fixtures, expects, and good/bad answer examples
- `run_eval.py` — CLI scorer (no live LLM required for examples)

## Run

```bash
# From repo root
.venv/bin/python -m evals.run_eval

# Or via pytest
.venv/bin/pytest tests/test_citation_and_golden.py -q
```

## Scoring a live agent dump

Write `{ "case_id": "answer text", ... }` then:

```bash
.venv/bin/python -m evals.run_eval --answers /tmp/answers.json
```

Checks include: required/forbidden strings, metric/geography presence, min citations, uncited assertion budget, and citation id resolution against fixtures.

## RAGAS metrics (`ragas_eval.py`)

Industry-standard faithfulness / answer_relevancy / context_precision /
answer_correctness over the live cases.

**Judge requirements (learned the hard way — do not regress):**
- Non-reasoning model only. Reasoning judges burn their completion budget on
  thinking tokens and RAGAS's structured sub-calls fail silently to NaN.
- A *paid* endpoint with real rate limits. Free tiers (Groq, Google AI) throttle
  under RAGAS's fan-out; retries surface as TimeoutError → NaN. Default is
  `xiaomi/mimo-v2.5` (~$0.14/M prompt; a full run ≈ $0.04).
- Judge calls are serialized + JSON-fence-stripped by `SerializedLLM` in the
  script — MiMo intermittently wraps JSON in markdown fences, which otherwise
  triggers parse-retry loops.

Temporal cases need current `expected_answer` values: `context_precision` and
`answer_correctness` grade against them, so a stale reference scores a *correct*
answer at 0 (see `world-cup-latest` history).

```bash
.venv/bin/python -m evals.ragas_eval                 # full run (~20 min)
.venv/bin/python -m evals.ragas_eval --limit 3       # smoke (~15 min)
# exit code 2 = any NaN — judge layer degrading again
```

## Repeats & case stability (`--repeat`, `--limit`)

Single-run score deltas under ±0.4 on this 12-case set are noise (live web +
judge variance). `--repeat N` runs each arm N times, averages per case, and
reports an "Unstable cases" section for any case whose cross-repeat score σ ≥ 1.0
— treat those as quarantined until their expected answers or grading mode are
fixed. `--limit N` runs just the first N cases for fast iteration.

```bash
.venv/bin/python -m evals.harness --models qwen/qwen3-30b-a3b-instruct-2507 \
    --ab-repair --repeat 3          # averaged lift + unstable-case report
```
