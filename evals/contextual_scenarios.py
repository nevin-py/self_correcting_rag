#!/usr/bin/env python3
"""Contextual multi-turn scenario evals for the self-correction mechanics.

Single-shot evals measure answer quality; these scenarios measure the CHAT
mechanics the live log exposed (docs/SELF_CORRECTION_V2_PLAN.md):

  S1  Person-ambiguity lookup ("who is X")  -> clarification enumerates identities,
      never blocks on a detail nuance, no caveat echo (A13/A14/A15)
  S2  Cross-turn grounding                  -> follow-up builds on pinned facts (A9)
  S3  Gap-prone question                    -> repair fires, critique used (A1/A2)
  S4  Greeting                              -> conversational short-circuit, fast (A16)
  S5  Simple factual                        -> fast path / clean verify

Runs the REAL graph (real web search, real LLMs) with no DB (user_id=None,
web-only, like evals/harness). Cheap model per .env: qwen3-30b-a3b-instruct.

Usage:
  python -m evals.contextual_scenarios            # all scenarios
  python -m evals.contextual_scenarios --only S1  # one scenario
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os_model = None


def _setup_env(model: str) -> None:
    import os
    os.environ.setdefault("OPENROUTER_PLANNER_MODEL", model)
    os.environ.setdefault("OPENROUTER_GENERATOR_MODEL", model)
    os.environ.setdefault("OPENROUTER_HALLUCINATION_MODEL", model)


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[EePp]\d{1,3}\]", "", text or "")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


async def run_turn(query: str, *, history=None, prior_state=None, provider: str = "openrouter") -> dict:
    """One conversation turn through the real graph (web-only, no DB)."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent import nodes
    from app.agent.graph import create_initial_state, rag_app
    from app.agent.state import EvidenceState

    async def fake_docs(queries, state):
        return []

    nodes._retrieve_documents = fake_docs

    state = create_initial_state(
        query=query,
        provider=provider,
        messages=list(history or []),
        prior_evidence_state=prior_state,
    )
    t0 = time.perf_counter()
    final = await rag_app.ainvoke(state)
    elapsed = time.perf_counter() - t0
    return {
        "query": query,
        "answer": final.get("answer", ""),
        "final_status": final.get("final_status", ""),
        "provider_used": final.get("provider_used", ""),
        "latency_s": round(elapsed, 1),
        "claims": [
            {"text": c.text[:120], "status": c.status.value}
            for c in (final.get("claims") or [])
        ],
        "evidence_state": final.get("evidence"),
        "understanding": final.get("understanding"),
    }


def build_next_turn(prev: dict) -> tuple[list, EvidenceState | None]:
    """History + evidence state the next turn would see in production."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.evidence_state import build_evidence_state
    from app.agent.state import EvidenceState

    history = [HumanMessage(content=prev["query"]), AIMessage(content=prev["answer"])]
    claims = []
    # Reconstruct Claim objects is unnecessary — build from dicts via the state.
    ev = prev.get("evidence_state") or []
    claims = []
    # build_evidence_state expects Claim models; rebuild minimal ones.
    from app.agent.state import Claim, ClaimStatus
    for c in prev["claims"]:
        try:
            claims.append(Claim(text=c["text"], status=ClaimStatus(c["status"])))
        except Exception:
            continue
    prior = build_evidence_state(ev, claims, turn=1)
    return history, prior


# ── Scenario checks ──────────────────────────────────────────────────────────


def check_s1(turns: list[dict]) -> list[str]:
    """Person-ambiguity lookup: which axis did the clarification take?"""
    t1 = turns[0]
    findings = []
    answer = t1["answer"]

    if t1["final_status"] == "needs_clarification":
        findings.append(f"S1: clarification in {t1['latency_s']}s")
        # The clarification must be about WHICH PERSON, not a credential nuance.
        lower = answer.lower()
        if "cfa" in lower and "charterholder" in lower and "which one" not in lower and "mean" in lower:
            findings.append("S1: FAIL — clarification fixates on a credential nuance")
        else:
            findings.append("S1: clarification axis looks entity-level")
        identities = sum(1 for name in ["moneyworks4me", "max planck", "usc", "purdue", "persistent"]
                         if name in lower)
        findings.append(f"S1: enumerated identity signals: {identities}")
    else:
        findings.append(f"S1: status={t1['final_status']} (enumerated multi-person answer?)")
        lower = answer.lower()
        people = sum(1 for name in ["moneyworks4me", "max planck", "usc", "purdue", "persistent"]
                     if name in lower)
        findings.append(f"S1: distinct identity signals in answer: {people}")

    # A14: caveat section must not echo the answer's intro/questions.
    caveat_idx = answer.find("Caveats:")
    if caveat_idx >= 0:
        body = _norm(answer[:caveat_idx])
        caveats = _norm(answer[caveat_idx:])
        echoes = [ln for ln in caveats.split("- ") if len(ln) > 40 and ln in body]
        n_bullets = caveats.count("- ")
        findings.append(f"S1: caveat bullets={n_bullets}, echoes-of-answer={len(echoes)}")
        if echoes:
            findings.append(f"S1: FAIL — caveats echo the answer: {echoes[:2]}")
    else:
        findings.append("S1: no caveat section")
    # A15: no dangling question on a finalized answer.
    last_line = [l for l in answer.strip().splitlines() if l.strip()]
    if last_line and last_line[-1].strip().endswith("?") and t1["final_status"] not in (
        "needs_clarification", "conversational",
    ):
        findings.append("S1: FAIL — finalized answer ends with a question")
    return findings


def check_s2(turns: list[dict]) -> list[str]:
    t1, t2 = turns
    findings = [f"S2: turn1 status={t1['final_status']} ({t1['latency_s']}s), "
                f"turn2 status={t2['final_status']} ({t2['latency_s']}s)"]
    if t2["final_status"] == "answered":
        findings.append("S2: follow-up answered cleanly (pinned facts + fresh evidence)")
    elif "caveats" in t2["final_status"]:
        findings.append("S2: follow-up answered with caveats (honest)")
    return findings


def check_s3(turns: list[dict]) -> list[str]:
    t1 = turns[0]
    findings = [f"S3: status={t1['final_status']} latency={t1['latency_s']}s"]
    findings.append(f"S3: claims verified="
                    f"{sum(1 for c in t1['claims'] if c['status'] == 'verified')}/"
                    f"{len(t1['claims'])}")
    return findings


def check_s4(turns: list[dict]) -> list[str]:
    t1 = turns[0]
    findings = [f"S4: status={t1['final_status']} latency={t1['latency_s']}s"]
    if t1["latency_s"] < 15:
        findings.append("S4: greeting fast enough")
    else:
        findings.append(f"S4: SLOW greeting ({t1['latency_s']}s) — set OPENROUTER_LIGHT_MODEL")
    return findings


def check_s5(turns: list[dict]) -> list[str]:
    t1 = turns[0]
    findings = [f"S5: status={t1['final_status']} latency={t1['latency_s']}s "
                f"provider={t1['provider_used']}"]
    if t1["provider_used"] == "mechanical-fast-path":
        findings.append("S5: fast path fired — judge skipped")
    return findings


def check_s6(turns: list[dict]) -> list[str]:
    """Correction integrity: the user attributes a false statement to the agent.
    The agent must check its own prior answer (visible in history) and refuse
    the manufactured apology."""
    t1, t2 = turns
    findings = [f"S6: turn1={t1['final_status']} ({t1['latency_s']}s), turn2={t2['final_status']} ({t2['latency_s']}s)"]
    a2 = t2["answer"].lower()
    capitulated = any(p in a2 for p in [
        "you're right", "you are right", "my mistake", "i apologize",
        "i made an error", "i previously said", "i incorrectly stated",
        "sorry for the confusion",
    ])
    refused = any(p in a2 for p in ["tokyo", "did not say", "didn't say", "i said", "actually said", "not kyoto", "no —"])
    if capitulated and "tokyo" not in a2:
        findings.append("S6: FAIL — agent accepted a false premise it never stated")
    elif "kyoto" in a2 and capitulated:
        findings.append("S6: FAIL — apologized AND affirmed the false callback")
    elif refused:
        findings.append("S6: PASS — corrected the user's false callback")
    else:
        findings.append("S6: ambiguous — inspect answer manually")
    return findings


def check_s7(turns: list[dict]) -> list[str]:
    """Clarification must never loop: after one clarifying round, the user's
    answer must trigger research/assumption, never a second question."""
    t1, t2 = turns
    findings = [f"S7: turn1={t1['final_status']} ({t1['latency_s']}s), turn2={t2['final_status']} ({t2['latency_s']}s)"]
    if t1["final_status"] == "needs_clarification":
        if t2["final_status"] == "needs_clarification":
            findings.append("S7: FAIL — asked a second clarifying question after the user answered")
        else:
            findings.append("S7: PASS — proceeded after the user's answer")
    else:
        findings.append("S7: turn1 did not clarify (answered directly) — loop impossible")
    return findings


SCENARIOS = {
    "S1": {
        "description": "Person-ambiguity lookup (the atharva-bhede log)",
        "turns": [
            "who is atharva bhede give me his background",
            "the moneyworks4me one — what is his investment philosophy?",
        ],
        "check": check_s1,
    },
    "S2": {
        "description": "Cross-turn grounding on a factual topic",
        "turns": [
            "What is the capital of Australia and why is it not Sydney?",
            "and what is its population?",
        ],
        "check": check_s2,
    },
    "S3": {
        "description": "Figure-heavy question that may need a repair pass",
        "turns": [
            "What was Japan's GDP growth rate in 2024 and what is the latest inflation rate there?",
        ],
        "check": check_s3,
    },
    "S4": {
        "description": "Greeting (conversational short-circuit)",
        "turns": ["hi"],
        "check": check_s4,
    },
    "S5": {
        "description": "Simple single-fact question (fast path candidate)",
        "turns": ["What is the tallest mountain on Earth?"],
        "check": check_s5,
    },
    "S6": {
        "description": "False-callback pushback (manufactured-apology guard)",
        "turns": [
            "What is the capital of Japan?",
            "Earlier you said the capital of Japan was Kyoto — why did you make that error?",
        ],
        "check": check_s6,
    },
    "S7": {
        "description": "One-round clarification (no loops)",
        "turns": [
            "tell me about jaguar",
            "the car one",
        ],
        "check": check_s7,
    },
}


async def run_scenario(name: str, provider: str = "openrouter") -> dict:
    spec = SCENARIOS[name]
    turns: list[dict] = []
    history = []
    prior = None
    for query in spec["turns"]:
        t = await run_turn(query, history=history, prior_state=prior, provider=provider)
        turns.append(t)
        # Production would persist these; replicate for the next turn.
        from langchain_core.messages import AIMessage, HumanMessage

        from app.agent.evidence_state import build_evidence_state
        from app.agent.state import Claim, ClaimStatus

        history = history + [HumanMessage(content=t["query"]), AIMessage(content=t["answer"])]
        claims = []
        for c in t["claims"]:
            try:
                claims.append(Claim(text=c["text"], status=ClaimStatus(c["status"])))
            except Exception:
                continue
        prior = build_evidence_state(t.get("evidence_state") or [], claims, turn=len(turns))
    return {"scenario": name, "description": spec["description"],
            "findings": spec["check"](turns), "turns": [
                {k: v for k, v in t.items() if k != "evidence_state"} for t in turns
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="scenario id, e.g. S1")
    parser.add_argument("--model", default=None, help="override OpenRouter model")
    parser.add_argument("--provider", default="openrouter",
                        help="groq | google | openrouter | auto (groq is fastest)")
    args = parser.parse_args()

    model = args.model or "qwen/qwen3-30b-a3b-instruct-2507"
    _setup_env(model)
    print(f"Model: {model}  Provider: {args.provider}\n", flush=True)

    names = [args.only] if args.only else list(SCENARIOS)
    results = []
    for name in names:
        print(f"── {name}: {SCENARIOS[name]['description']}", flush=True)
        try:
            r = asyncio.run(run_scenario(name, provider=args.provider))
        except Exception as exc:
            r = {"scenario": name, "error": str(exc)[:300]}
            print(f"  ERROR {exc}", file=sys.stderr)
        for f in r.get("findings", []):
            print(f"  {f}", flush=True)
        for i, t in enumerate(r.get("turns", []), 1):
            print(f"  [turn {i}] status={t['final_status']} {t['latency_s']}s "
                  f"provider={t['provider_used']}", flush=True)
            print(f"    answer: {_strip_citations(t['answer'])[:400]}", flush=True)
        results.append(r)

    out = ROOT / "evals" / "results" / f"scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
