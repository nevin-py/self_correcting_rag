# Self-Correction V2 — Diagnosis & Redesign Plan

> Why the agent "gives good information but contextually misunderstands and sometimes
> doesn't correct", grounded in a full code read (`app/agent/nodes.py`, `state.py`,
> `citation_validator.py`, `support.py`, `evidence_state.py`, `chat_service.py`,
> `graph.py`) and in the current literature.

---

## Part 1 — Diagnosis: 12 concrete defects

### A. The correction loop is structurally broken

**A1. Repair regenerates blind — no critique reaches the generator.**
`verify_answer` sets `repair_queries` and the graph loops `gather → generate → verify`.
But `generate_answer` never sees the previous answer, the judge's verdict, or the
failed-claim reasons — it re-runs the identical prompt with slightly different evidence.
This violates the core finding of Reflexion (arXiv:2303.11366): self-improvement comes
from persisting *verbal feedback* and feeding it into the next attempt. Without it, the
second pass repeats the same mistakes; the loop mostly wastes a cycle. **This is the
single biggest reason "it doesn't correct."**

**A2. Repair evidence is reranked against the wrong query.**
On the repair pass, `gather_evidence` reranks the full pool against
`state.get("query")` — the *original rewritten query* — not the judge's repair queries.
Gap-targeting evidence retrieved via `repair_queries` therefore scores low and usually
fails to make the top-12 cut into `assembled_context`. The repair pass often gathers
evidence the generator never sees. **Repair is a no-op far more often than the logs
suggest.**

**A3. The judge verifies in a vacuum — the user's question is missing.**
`_VERIFY_PROMPT` contains evidence + answer but **not the question being answered**.
The judge can check "is each sentence supported?" but cannot detect the failure mode you
are reporting: a fluent, well-cited answer to *the wrong question* (missed referent in a
follow-up, wrong period, wrong entity) verifies as fully "supported".

**A4. Generator/verifier context asymmetry produces false verdicts.**
The generator sees 1200 chars per evidence block (`EVIDENCE_SNIPPET_CHARS`); the judge
sees only 400 chars and a 6000-char total cap (`_verify_context`). The judge
"unverifies" claims whose support it simply wasn't shown → spurious caveats → the
answer reads as unsure even when it was right.

**A5. Claims are merged by exact text.**
In `verify_answer`, judge claims and mechanical claims are merged via
`merged[c.text] = c`. The judge paraphrases; the result is duplicate claims (one
"verified", one "unverified" for the same sentence) feeding noisy caveats.

**A6. The support gate measures cosine, not entailment.**
`support.py` scores a one-sentence claim against a 1500-char evidence chunk with MiniLM
cosine at 0.55. Long chunks dilute similarity; paraphrases with different surface forms
fail. This strips legitimate citations and manufactures caveats —
"capacit(g)y is good but it contextually misunderstands" is partly this gate silently
demoting correct claims.

**A7. No evidence-sufficiency gate before generation.**
`gather_evidence` takes top-12 by rerank score unconditionally. If retrieval returned
junk (or the rewrite drifted), the generator confidently answers from garbage and the
mistake is only discovered after a full generate+verify cycle. CRAG (arXiv:2401.15884)
puts a retrieval evaluator *before* generation for exactly this reason.

**A8. Repair triggers are too narrow.**
Only `unverified` claims with `repair_queries` and zero contradictions trigger repair.
`uncertain` (ambiguous/partial evidence) never triggers a disambiguating search;
"answered the wrong question" (A3) was undetectable; `MAX_REPAIR_PASSES=1` gives one
shot that (per A1/A2) usually doesn't work.

### B. Context & chat mechanics

**A9. Established facts compete in rerank and lose.**
Prior-turn verified evidence (`evidence_state.established`) is dumped into the candidate
pool and reranked against the *new* query. On a follow-up ("why?") the grounding for the
previous answer can drop out of top-12 → the agent contradicts or re-hedges what it had
already established. **This is the cross-turn misunderstanding.**

**A10. History is a hard 12-message trim with no compaction.**
Long chats lose the thread entirely; there is no rolling summary, and the generator gets
no structured memory of last turn's verdict (only `classify_and_plan` sees
`prior_evidence_state`).

**A11. Dead legacy code confuses maintenance.**
`chat_service._load_prior_evidence_summary` parses an old `metric|geography|period|value`
format that nothing writes anymore. `nodes.py` defines `_structured_invoke` twice.

**A12. No position-bias mitigation.** Evidence blocks are rendered strictly score-desc;
relevant-but-middle evidence is where "lost in the middle" hurts most.

### C. Live-log case study (the "atharba bhede" chat) — 4 more defects

Real interaction: "hi" (27s) → "who is atharba bhede give me his background" → a
`needs_clarification` turn in **250s** asking about CFA charterholder vs Level-3
candidate → user corrects spelling → `answered_with_caveats` turn in **299s** whose
caveat list echoes the answer's own intro and closing question.

**A13. Clarification targets the wrong axis.** The user's ambiguity was *which
Atharva Bhide*; the agent blocked them on a credential nuance they never asked about.
Root cause: `_conflict_clarification()` fires on any contradicted claim, and the judge
writes `clarification_question` fixated on the contradicted detail (charter status),
not the user's intent. The judge never sees the question (A3), so it cannot know
identity is what's at stake.

**A14. The caveat list echoes the answer.** Turn 3's caveats include "Based on the
evidence, there appear to be multiple individuals…" and "Could you clarify which one
you're interested in?" — the answer's own intro and closing question. Mechanism:
`_is_factual_assertion()` accepts any sentence containing `\b[A-Z][a-z]+` ("Based",
"Could" match), such sentences are UNCITED_ASSERTION → UNVERIFIED →
`_append_caveats()` with no meta-sentence filter, no cap, and no dedupe against the
answer body. Judge paraphrase duplicates (A5) inflate it further (19 verified / 14
unverified for ~8 real claims).

**A15. Hybrid clarify-or-answer state.** Turn 3's answer ends with a clarifying
question but `final_status = answered_with_caveats` — the judge left
`clarification_question` empty, so the prose question leaked through with no state
change. Turn 2's clarification also emitted 43 claims (20 verified / 23 unverified)
into evidence state for a turn that asked the user something.

**A16. Latency is unbudgeted.** 27s for a greeting (single LLM call on a free-tier
fallback model), 250–300s for research turns: repair pass doubles the whole pipeline,
the judge fallback chain can stack 3×30s structured timeouts, and full-page enrichment
runs unconditionally. There is no fast path: even an answer that passes every
deterministic check still pays for the LLM judge.

---

## Part 2 — What the literature says (triangulated)

| Paper | Core idea | What it fixes here |
|---|---|---|
| **Reflexion** (arXiv:2303.11366) | Persist verbal self-critique in episodic memory; feed it to the next trial | A1 — critique must reach regeneration |
| **CRAG** (arXiv:2401.15884) | Retrieval evaluator grades retrieved docs correct/incorrect/ambiguous *before* generation; triggers corrective search / query refinement | A7, A8 — sufficiency gate before generating |
| **Self-RAG** (arXiv:2310.11511) | Reflection tokens: is passage relevant, is response supported, is it useful | A3, A5 — judge must score relevance-to-question and support separately |
| **Decomposing LLM Self-Correction** (arXiv:2601.00828) | Intrinsic self-correction is unreliable; decompose into detect → localize → correct with *external* feedback | A1, A5 — the deterministic gates + critique hand-off are the correction mechanism, not the model's conscience |
| **Search-o1** (arXiv:2501.05366) | Reason-in-Documents module prunes/refines retrieved docs before injecting them | A4, A12 — evidence refinement for the consumer of the context |
| **RAC** (arXiv:2601.11722) | Clarification questions must be grounded in the corpus | keeps your clarification node honest |
| **S2G / structured stopping** (arXiv:2608.13237) | Explicit sufficiency + gap judgment decides when to stop retrieving | A7 — "is this enough?" as an explicit structured verdict |

Convergent design across all of them: **evaluate retrieval quality before generating,
ground the critique externally (deterministic gates + evidence), feed the critique back
into a revision pass, and make "sufficient? / supported? / relevant?" explicit
structured judgments rather than vibes.**

---

## Part 3 — The plan (5 phases, each independently shippable + A/B testable)

### Phase 1 — Make the repair loop actually correct (highest ROI, ~1–2 days)

1. **Critique-carrying regeneration (Reflexion).** Extend `RAGState` with
   `critique: str | None`. When `verify_answer` schedules repair, it serializes the
   verdict (failed claims + reasons + repair intent) into `critique`. `generate_answer`
   checks `state["repair_count"] > 0` and switches to a **REVISE** system prompt:

   > You previously drafted an answer; a verifier found these problems: {critique}.
   > New evidence is marked ★. Rewrite the answer: fix or remove the flagged claims,
   > integrate the new evidence, keep everything that was verified.

   The previous answer goes in as a draft; tokens stream as an `answer_reset` (already
   supported by the streaming layer).

2. **Rerank repair evidence against the right queries.** In `gather_evidence`, when
   `repair_mode`, rerank new evidence against the *repair queries* (max score across
   original + repair queries) and **reserve 4 of 12 context slots** for repair-pass
   evidence so it cannot be crowded out by the old pool.

3. **Put the question in the judge's prompt.** Add `question` (original + rewritten) to
   `_VERIFY_PROMPT` and add to the verdict schema:
   `addresses_question: bool`, `question_gaps: list[str]`. Wrong-question answers now
   route to repair (a better-targeted rewrite) instead of "supported".

4. **Symmetric context.** Give the judge the same 1200-char snippets as the generator
   and raise its cap to ~12k chars. A verdict made on truncated evidence is not a
   verdict.

5. **Fuzzy claim merge.** Merge judge + mechanical claims on normalized text
   (lowercase, strip citations/punct) with fallback on evidence_ids overlap + status,
   so one sentence produces one claim.

6. **Caveat hygiene (fixes A14).** In `citation_validator` + `_append_caveats`:
   exclude questions, first-person sentences, and meta openers ("Based on the
   evidence", "Here's what I found") from claim extraction entirely; caveat bullets
   must be atomic, factual, and not already qualified inline; cap at 5 bullets; drop
   any bullet whose normalized text already appears in the verified answer body.

7. **Exclusive clarify-or-answer (fixes A15).** Post-process the final answer: if it
   ends with a question to the user, either set `final_status = needs_clarification`
   or strip the question and commit to the best-guess answer with the assumption
   stated ("I've summarized all five; tell me which one if you want depth"). Never
   both. Skip claim/evidence-state persistence on clarification turns so 43 phantom
   claims never enter memory.

### Phase 2 — Sufficiency gate before generation (CRAG pattern, ~1 day)

8. **Sufficiency check before generation (CRAG).** After the first `gather_evidence`, run
   deterministic floor first (top rerank score ≥ threshold, ≥ 2 non-duplicate
   evidence); if it fails, loop back to search with decomposed queries — *before*
   wasting a generate+verify cycle. Optional tiny LLM evaluator
   (`correct | ambiguous | incorrect` per CRAG) only when the deterministic check is
   borderline. Bounded by existing `MAX_SEARCHES`.
9. **Widen repair triggers:** `uncertain` claims whose reasoning mentions conflicting
   evidence → one disambiguating repair query; `question_gaps` → rewrite-based repair.
   Bump `MAX_REPAIR_PASSES` to 2 with distinct modes: pass 1 = targeted search,
   pass 2 = revise-only (no new search, critique-driven rewrite).

10. **Early entity triage (fixes A13).** For person/org lookups, run a cheap structured
   triage right after the first `gather_evidence` (before generate+verify): one small
   LLM call — "how many distinct real-world referents does the queried name match in
   this evidence?" If >1 plausible, ask the user immediately with the enumerated
   identities from the evidence (RAC-style grounded clarification:
   "Atharva Bhide — finance at MoneyWorks4Me / physics PhD at MPQ / AI engineer at
   USC / …"). This converts a 250s clarification into a ~30s one and asks the
   question the user can actually answer. Detail-level conflicts (charterholder vs
   candidate) are never clarification-worthy: report both inline with attribution.

11. **Grounded clarification rule:** a clarification question may only be asked when
   the *final answer materially differs* between interpretations, and it must list
   the interpretations found in evidence. Credential/source nuances are resolved by
   preferring the more authoritative source and stating the discrepancy.

### Phase 3 — Support gate: cosine → entailment (~1 day)

12. **Support-gate windows.** Score the claim against the **best-matching sliding window** of the cited chunk
   (e.g., 2–3 sentence windows, max similarity) instead of whole-chunk cosine — fixes
   long-chunk dilution without a new model. Keep 0.55 on windows.
13. **Entailment model.** Optional: local ONNX NLI model (DeBERTa-MNLI) for cited-claim entailment, falling
   back to the window-cosine. Safe-directional failure preserved (demote → caveat,
   never silently keep). Re-run `evals/harness --ab-repair` with the gate on/off.

### Phase 4 — Chat & cross-turn context management (~1–2 days)

14. **Pin established facts.** Prior-turn verified evidence gets reserved context slots
    rendered as a separate labeled block ("Established in this conversation
    (previously verified)") in the generate prompt — no longer competing in rerank.
    Also inject `_prior_evidence_block` into the *generate* prompt, not just classify.
15. **Conversation compaction.** Rolling summary for messages beyond the recent
    verbatim window (persist in chat metadata, update per turn; ~80 tokens). Planner,
    generator, and judge all receive `summary + last-N verbatim`.
16. **Verdict memory.** Store a compact verdict digest (`{verified: n, unresolved: [..],
    conflicts: [..]}`) in the assistant message provenance; inject the previous turn's
    digest into the generate prompt so the agent never re-asserts something it
    caveated last turn.
17. Cleanup: delete `_load_prior_evidence_summary` (dead), dedupe the double
    `_structured_invoke` definition.

### Phase 5 — Latency & confident fast-path (fixes A16, ~1 day)

18. **Fast-path skip the judge.** If mechanical citation validation + support gate
    pass (all factual claims cited and supported, zero contradictions, question
    addressed per Phase 1's `addresses_question`), skip the LLM judge entirely —
    `final_status = "answered"`. This is the design's own "cheap checks first" rule
    taken to its conclusion; it removes ~30–90s from every good answer.
19. **Per-node wall-clock budgets** derived from `QUERY_TIMEOUT_SECONDS` (e.g.
    planner 20% / gather 25% / generate 30% / judge 25%); on breach, skip the node's
    non-essential work (enrichment, raw-JSON retries) rather than stacking 30s
    structured timeouts across a 3-model fallback chain.
20. **Fast model for conversational node** — the greeting took 27.4s because it rode
    the generator chain. Route `conversational_response` to the smallest/fastest
    configured model (new `OPENROUTER_LIGHT_MODEL`), falling back to a canned reply
    on failure — a greeting never needs a research-grade model.
21. Cap full-page enrichment by wall-clock, not just count (`EVIDENCE_FETCH_TOP_N=2`
    stays, each fetch gets a 8s budget).

### Phase 6 — Prove it (~0.5 day + runs)

22. Extend `evals/harness`:
    - **repair-lift v2**: report whether the repair pass *changed the answer* and
      whether flagged claims flipped to verified (detects A1/A2 no-ops).
    - **wrong-question suite**: follow-up cases where the naive path answers the wrong
      referent/period (tests A3).
    - **follow-up grounding suite**: multi-turn cases testing pinned established facts
      (tests A9).
18. A/B each phase independently (`--ab-repair` pattern exists); keep the phase only on
    judge-score + verified-claim-share lift with caveats-precision not regressing.
    Add latency P50/P95 per node to the report (Phase 5 acceptance: P50 research turn
    < 60s, conversational < 5s), plus a **clarification-quality suite** replaying the
    atharba-bhede log: the clarification must enumerate persons, not credential nuance.

### Config additions (`.env.example`)

```
REPAIR_RESERVED_SLOTS=4        # context slots guaranteed to repair evidence
JUDGE_SNIPPET_CHARS=1200       # match generator view
JUDGE_CONTEXT_CHARS=12000
SUFFICIENCY_TOP_SCORE=0.35     # deterministic retrieval-quality floor
REPAIR_MAX_PASSES=2            # 1 search pass + 1 revise-only pass
```

**Expected impact:** Phase 1 converts the existing repair machinery from a ~no-op into
a real correction loop (the harness already showed +0.42 judge score when repair
*gathers*; critique-carrying regeneration is where the remaining lift is). Phase 2
prevents the confident-wrong-answer class outright. Phases 3–4 remove the false-caveat
noise that reads as "misunderstanding".

---

## References

- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through
  Self-Reflection*, arXiv:2310.11511
- Yan et al., *Corrective Retrieval Augmented Generation*, arXiv:2401.15884
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*,
  arXiv:2303.11366
- *Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth
  Hypothesis*, arXiv:2601.00828
- Li et al., *Search-o1: Agentic Search-Enhanced Large Reasoning Models*,
  arXiv:2501.05366
- *RAC: Retrieval-Augmented Clarification for Faithful Conversational Search*,
  arXiv:2601.11722
- *When Should Multi-Round RAG Stop? Structured Stopping Judgments*, arXiv:2608.13237
