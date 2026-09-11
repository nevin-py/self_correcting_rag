#!/usr/bin/env python3
"""Multi-turn golden eval: uploaded-document preference vs live web search.

Uses the REAL stack end-to-end: Postgres+pgvector, Nomic embeddings, hybrid
retrieval, the full LangGraph pipeline, and live web search.

Isolation trick: the ingested "Zentro Dynamics" report is fictitious — its
figures exist nowhere on the web. Any answer carrying them must have come from
the document (C1-C4). C5 runs the same question with NO document ingested and
must NOT produce the private figures (hallucination/leakage control).

DB rows created for the eval (2 users, 2 chats, chunk rows) are deleted at the
end. Nothing is committed to git by this script.

Usage:
  python -m evals.multiturn_doc_eval --provider groq
  python -m evals.multiturn_doc_eval --provider openrouter --model nvidia/nemotron-3-super-120b-a12b:free
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

DOC_FILENAME = "Zentro Dynamics Q3 FY2025 Investor Report.md"

DOC_TEXT = """# Zentro Dynamics — Q3 FY2025 Investor Report

## Financial Highlights
Zentro Dynamics closed Q3 FY2025 with revenue of Rs 412 crore, up 23% quarter-over-quarter,
driven by enterprise subscriptions of Orbit Analytics 2.0. Gross margin held at 61%.

## Expansion
During the quarter the company expanded operations to 27 new cities, bringing the total
city coverage to 89 across India and Southeast Asia.

## Customer Metrics
Monthly logo churn improved to 2.1%, the lowest in company history. Net revenue retention
stood at 118%. The company reported an NPS of 71.

## Organization
Headcount reached 1,840 employees at quarter end. The company was founded by Meera Krishnan
and is headquartered in Pune, India.

## Product
Orbit Analytics 2.0 launched in September 2025, adding real-time forecasting to the suite.

## Guidance
Management guides to FY2026 revenue of Rs 1,900 crore.
"""


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[EePp]\d{1,3}\]", "", text or "")


async def setup_db() -> dict:
    """Create synthetic user+chat rows and ingest the document. Returns ids."""
    from sqlalchemy import text as sql_text

    from app.core.database import AsyncLocalSession
    from app.documents.service import full_pipeline

    eval_user = uuid.uuid4()
    eval_chat = uuid.uuid4()
    control_chat = uuid.uuid4()
    async with AsyncLocalSession() as db:
        await db.execute(
            sql_text("INSERT INTO users (user_id, email, hashed_password) "
                     "VALUES (:uid, :email, :pw)"),
            {"uid": eval_user, "email": f"eval-{eval_user}@eval.invalid", "pw": "x"},
        )
        await db.execute(
            sql_text("INSERT INTO chats (chat_id, user_id, title) VALUES (:cid, :uid, :t)"),
            {"cid": eval_chat, "uid": eval_user, "t": "doc-eval"},
        )
        await db.execute(
            sql_text("INSERT INTO chats (chat_id, user_id, title) VALUES (:cid, :uid, :t)"),
            {"cid": control_chat, "uid": eval_user, "t": "doc-eval-control"},
        )
        await db.commit()

    result = await full_pipeline(DOC_TEXT.encode(), DOC_FILENAME, eval_user, eval_chat)
    return {
        "user_id": eval_user,
        "chat_id": eval_chat,
        "control_chat_id": control_chat,
        "chunks": result["no_of_chunks"],
    }


async def cleanup_db(ids: dict) -> None:
    from sqlalchemy import delete as sql_delete, text as sql_text

    from app.core.database import AsyncLocalSession
    from app.documents.models import DocumentChunk
    from app.agent.models import Chats

    uid = ids["user_id"]
    async with AsyncLocalSession() as db:
        await db.execute(sql_delete(DocumentChunk).where(DocumentChunk.user_id == uid))
        await db.execute(sql_delete(Chats).where(Chats.user_id == uid))
        await db.execute(sql_text("DELETE FROM users WHERE user_id = :uid"), {"uid": uid})
        await db.commit()


async def run_turn(query: str, *, user_id, chat_id, document_inventory, history=None,
                   prior_state=None, provider: str) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.graph import create_initial_state, rag_app
    from app.agent.state import Claim, ClaimStatus

    state = create_initial_state(
        query=query,
        user_id=user_id,
        chat_id=chat_id,
        provider=provider,
        messages=list(history or []),
        prior_evidence_state=prior_state,
        document_inventory=document_inventory,
    )
    t0 = time.perf_counter()
    final = await rag_app.ainvoke(state)
    elapsed = time.perf_counter() - t0

    citations = []
    for ev in (final.get("evidence") or [])[:12]:
        if ev.metadata.get("cite_key"):
            citations.append({
                "key": ev.metadata["cite_key"],
                "type": ev.source_type.value,
                "name": ev.source_name[:60],
            })
    return {
        "query": query,
        "answer": final.get("answer", ""),
        "final_status": final.get("final_status", ""),
        "latency_s": round(elapsed, 1),
        "claims": [{"text": c.text[:150], "status": c.status.value}
                   for c in (final.get("claims") or [])],
        "citations": citations,
        "evidence": final.get("evidence") or [],
        "understanding": final.get("understanding"),
    }


def next_turn_state(prev: dict):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.agent.evidence_state import build_evidence_state
    from app.agent.state import Claim, ClaimStatus

    history = [HumanMessage(content=prev["query"]), AIMessage(content=prev["answer"])]
    claims = []
    for c in prev["claims"]:
        try:
            claims.append(Claim(text=c["text"], status=ClaimStatus(c["status"])))
        except Exception:
            continue
    return history, build_evidence_state(prev["evidence"], claims, turn=1)


def doc_cited(turn: dict) -> bool:
    return any(c["type"] == "document" for c in turn["citations"])


def score_case(case: dict, turns: list[dict]) -> list[str]:
    exp = case["expect"]
    findings = []
    for i, turn in enumerate(turns, 1):
        answer = _strip_citations(turn["answer"])
        lower = answer.lower()
        findings.append(f"    turn{i}: {turn['final_status']} {turn['latency_s']}s")
        if turn["final_status"] not in exp["status"]:
            findings.append(f"    FAIL turn{i}: status={turn['final_status']} not in {exp['status']}")
        prefix = "" if i == 1 else f"turn{i}_"
        contains_key = f"{prefix}must_contain_any" if f"{prefix}must_contain_any" in exp else "must_contain_any"
        if contains_key in exp:
            hits = [v for v in exp[contains_key] if v.lower() in lower]
            findings.append(f"    {'PASS' if hits else 'FAIL'} turn{i}: contains {hits or exp[contains_key]}")
        for bad in exp.get("must_not_contain_any", []):
            if bad.lower() in lower:
                findings.append(f"    FAIL turn{i}: leaked forbidden value {bad!r}")
        if exp.get("doc_source_in_citations") or (f"turn{i}_must_cite_document" in exp):
            findings.append(f"    {'PASS' if doc_cited(turn) else 'FAIL'} turn{i}: document-type evidence cited")
    # Caveat hygiene check on every turn.
    for i, turn in enumerate(turns, 1):
        answer = turn["answer"]
        idx = answer.find("Caveats:")
        if idx >= 0:
            bullets = answer[idx:].count("- ")
            if bullets > 5:
                findings.append(f"    FAIL turn{i}: {bullets} caveat bullets (cap is 5)")
    return findings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--keep-rows", action="store_true")
    args = parser.parse_args()
    if args.model:
        import os
        os.environ["OPENROUTER_PLANNER_MODEL"] = args.model
        os.environ["OPENROUTER_GENERATOR_MODEL"] = args.model
        os.environ["OPENROUTER_HALLUCINATION_MODEL"] = args.model

    spec = json.loads((ROOT / "evals" / "multiturn_golden.json").read_text())
    print(f"Ingesting document (real Nomic embeddings + pgvector)...", flush=True)
    ids = await setup_db()
    print(f"  ingested {ids['chunks']} chunks for user={ids['user_id']}\n", flush=True)

    inventory = [DOC_FILENAME]
    results = []
    try:
        for case in spec["cases"]:
            control = case["id"] == "C5-no-doc-control"
            chat_id = ids["control_chat_id"] if control else ids["chat_id"]
            inv = inventory if not control else []
            print(f"── {case['id']}{' (NO DOC control)' if control else ''}", flush=True)
            turns = []
            history, prior = [], None
            for q in case["turns"]:
                t = await run_turn(q, user_id=ids["user_id"], chat_id=chat_id,
                                   document_inventory=inv, history=history,
                                   prior_state=prior, provider=args.provider)
                turns.append(t)
                history, prior = next_turn_state(t)
            findings = score_case(case, turns)
            for f in findings:
                print(f"  {f}", flush=True)
            print(f"  answer: {_strip_citations(turns[-1]['answer'])[:300]}", flush=True)
            results.append({
                "case": case["id"],
                "findings": findings,
                "turns": [{k: v for k, v in t.items() if k != "evidence"} for t in turns],
            })
    finally:
        if not args.keep_rows:
            await cleanup_db(ids)
            print("\n(eval rows cleaned up)")

    out = ROOT / "evals" / "results" / f"multiturn_doc_{uuid.uuid4().hex[:8]}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    n_fail = sum(1 for r in results for f in r["findings"] if "FAIL" in f)
    print(f"\nSaved: {out}")
    print(f"RESULT: {len(results)} cases, {n_fail} failed checks")


if __name__ == "__main__":
    asyncio.run(main())
