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
