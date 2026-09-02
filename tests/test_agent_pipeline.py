"""Tests for the lean self-correcting RAG pipeline.

Covers the observable contract of the new architecture:
  - classify_and_plan routing (research / conversational / clarification)
  - gather_evidence (parallel sources, dedupe, cite keys, prior-evidence carry)
  - generate_answer (cited output, empty-context honesty)
  - verify_answer (judge verdicts, repair loop, caveats)
  - cross-turn evidence state round-trip
  - placeholder API-key guard

LLMs are fakes — no network access in tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.agent import nodes
from app.agent.evidence_state import (
    build_evidence_state,
    load_evidence_state_from_text,
    merge_evidence_state,
    serialize_for_storage,
)
from langchain_core.messages import HumanMessage
from app.agent.graph import create_initial_state, rag_app
from app.agent.nodes import (
    ask_clarification,
    classify_and_plan,
    conversational_response,
    gather_evidence,
    generate_answer,
    route_after_classify,
    route_after_verify,
    verify_answer,
)
from app.agent.state import (
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceState,
    QueryMode,
    QueryUnderstanding,
    SourceType,
    Verdict,
)
from app.documents.clients import valid_key


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeLLM:
    """Minimal LangChain-like chat model returning canned content."""

    def __init__(self, text: str = "", structured: object = None, fail: bool = False):
        self.text = text
        self.structured = structured
        self.fail = fail
        self.calls: list[list] = []

    def invoke(self, messages, *a, **k):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(content=self.text)

    def with_structured_output(self, schema, method=None):
        outer = self

        class _Bound:
            def invoke(self, messages, *a, **k):
                outer.calls.append(messages)
                if outer.fail:
                    raise RuntimeError("boom")
                return outer.structured

        return _Bound()


class FakeLLMs:
    """Stand-in for ProviderLLMs."""

    def __init__(self, planner=None, generator=None, verifier=None):
        self.planner = planner or FakeLLM()
        self.generator = generator or FakeLLM()
        self.verifier = verifier or FakeLLM()
        self.planner_fallbacks: tuple = ()
        self.generator_fallbacks: tuple = ()
        self.verifier_fallbacks: tuple = ()
        self.label = "fake"


def _patch_llms(monkeypatch, llms: FakeLLMs):
    monkeypatch.setattr(nodes, "_resolve_llms", lambda state: llms)


def _patch_rerank(monkeypatch):
    """Identity rerank — avoids loading the real FlashRank model in tests."""
    monkeypatch.setattr(nodes, "rerank", lambda q, items, top_k=None: [
        SimpleNamespace(text=t, score=0.9, source=s) for s, t in items
    ])


def _state(**over) -> dict:
    base = create_initial_state(
        query=over.pop("query", "What is the population of Japan?"),
        user_id="u1",
        chat_id="c1",
    )
    base.update(over)
    return dict(base)


_JAPAN_EV_TEXT = "Japan's population is about 124 million."


def _japan_ev() -> Evidence:
    ev = Evidence(evidence_id="e1", text=_JAPAN_EV_TEXT,
                  source_type=SourceType.WEB, source_name="wb")
    ev.metadata["cite_key"] = "E1"
    return ev


# ── classify_and_plan ────────────────────────────────────────────────────────


class TestClassifyAndPlan:
    def test_research_mode_plans_queries(self, monkeypatch):
        u = QueryUnderstanding(
            mode=QueryMode.RESEARCH,
            rewritten_query="population of Japan 2025",
            needs_documents=False,
            needs_web=True,
            search_queries=["Japan population 2025"],
            temporal_focus="latest",
            geography="Japan",
        )
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state())
        assert out["understanding"].mode == QueryMode.RESEARCH
        assert out["query"] == "population of Japan 2025"
        assert route_after_classify(out) == "gather_evidence"

    def test_conversational_routes_to_chat(self, monkeypatch):
        u = QueryUnderstanding(mode=QueryMode.CONVERSATIONAL)
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state(query="hey there"))
        assert route_after_classify(out) == "conversational_response"

    def test_clarification_routes_when_question_present(self, monkeypatch):
        u = QueryUnderstanding(
            mode=QueryMode.CLARIFICATION,
            clarification_question="Do you mean the city of Paris or Paris, Texas?",
        )
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state(query="paris"))
        assert route_after_classify(out) == "ask_clarification"
        result = ask_clarification(out)
        assert result["final_status"] == "needs_clarification"
        assert "Paris" in result["answer"]

    def test_llm_failure_defaults_to_research(self, monkeypatch):
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(fail=True)))
        out = classify_and_plan(_state())
        u = out["understanding"]
        assert u.mode == QueryMode.RESEARCH
        assert u.needs_documents and u.needs_web
        assert u.search_queries

    def test_temporal_context_injected_in_prompt(self, monkeypatch):
        u = QueryUnderstanding(mode=QueryMode.RESEARCH, search_queries=["x"])
        planner = FakeLLM(structured=u)
        _patch_llms(monkeypatch, FakeLLMs(planner=planner))
        classify_and_plan(_state(request_context={"timezone": "Asia/Kolkata"}))
        prompt_text = str(planner.calls[0][0].content)
        # The current year must appear so the LLM is temporally grounded
        assert str(datetime.now().year) in prompt_text
    def test_clarification_without_question_coerces_to_research(self, monkeypatch):
        """Planner said 'clarification' but wrote no question — must degrade to
        research with web search, never starve the pipeline."""
        u = QueryUnderstanding(mode=QueryMode.CLARIFICATION, clarification_question="")
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state(query="what happened to cjp in todays news"))
        got = out["understanding"]
        assert got.mode == QueryMode.RESEARCH
        assert got.needs_web is True
        assert any(q.strip() for q in got.search_queries)
        assert route_after_classify(out) == "gather_evidence"

    def test_research_without_sources_forces_web(self, monkeypatch):
        u = QueryUnderstanding(mode=QueryMode.RESEARCH, needs_documents=False, needs_web=False)
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state())
        assert out["understanding"].needs_web is True

    def test_document_reference_coerces_clarification_to_research(self, monkeypatch):
        """"talk about this paper" after ingesting ZK-PFL Paper.pdf must NEVER
        produce a clarification question — the planner knows the doc list and
        the mechanical guard routes to research anchored on the document."""
        u = QueryUnderstanding(
            mode=QueryMode.CLARIFICATION,
            clarification_question="Which paper do you mean?",
            search_queries=[""],
        )
        _patch_llms(monkeypatch, FakeLLMs(planner=FakeLLM(structured=u)))
        out = classify_and_plan(_state(
            query="talk about this paper",
            document_inventory=["ZK-PFL Paper.pdf"],
            messages=[HumanMessage(content="i just gave you the zk pfl paper")],
        ))
        got = out["understanding"]
        assert got.mode == QueryMode.RESEARCH
        assert got.needs_documents is True
        assert route_after_classify(out) == "gather_evidence"
        # Retrieval must be anchored on the referenced document's own tokens.
        assert any("zk" in q.lower() or "pfl" in q.lower() for q in got.search_queries)
# ── conversational_response ──────────────────────────────────────────────────


class TestConversational:
    def test_replies_via_llm(self, monkeypatch):
        _patch_llms(monkeypatch, FakeLLMs(generator=FakeLLM(text="Hello! What can I research for you?")))
        out = conversational_response(_state(query="hi"))
        assert out["final_status"] == "conversational"
        assert "research" in out["answer"].lower()

    def test_llm_failure_still_answers(self, monkeypatch):
        _patch_llms(monkeypatch, FakeLLMs(generator=FakeLLM(fail=True)))
        out = conversational_response(_state(query="hi"))
        assert out["answer"]


# ── gather_evidence ──────────────────────────────────────────────────────────


class TestGatherEvidence:
    def test_cite_keys_assigned_and_context_built(self, monkeypatch):
        async def fake_web(queries, state):
            return [
                Evidence(text="Japan's population is about 124 million.", source_type=SourceType.WEB,
                         source_name="worldbank", source_url="https://worldbank.org/jp"),
                Evidence(text="UN data: Japan population is about 124 million.", source_type=SourceType.WEB,
                         source_name="un", source_url="https://un.org/jp"),
            ]

        monkeypatch.setattr(nodes, "_search_web", fake_web)
        _patch_rerank(monkeypatch)
        st = _state(understanding=QueryUnderstanding(
            mode=QueryMode.RESEARCH, needs_documents=False, needs_web=True,
            search_queries=["Japan population"],
        ))
        out = asyncio.run(gather_evidence(st))
        assert len(out["evidence"]) == 2
        assert list(out["cite_map"].keys()) == ["E1", "E2"]
        assert "[E1]" in out["assembled_context"]

    def test_prior_evidence_carried_into_context(self, monkeypatch):
        async def fake_web(queries, state):
            return []

        monkeypatch.setattr(nodes, "_search_web", fake_web)
        _patch_rerank(monkeypatch)
        prior_ev = Evidence(evidence_id="p1", text="Prior verified fact: Tokyo is the capital.",
                            source_type=SourceType.WEB, source_name="old")
        st = _state(
            understanding=QueryUnderstanding(mode=QueryMode.RESEARCH, needs_documents=False, needs_web=True),
            prior_evidence_state=EvidenceState(turn=1, established=[prior_ev]),
        )
        out = asyncio.run(gather_evidence(st))
        assert any(ev.evidence_id == "p1" for ev in out["evidence"])
        assert "Tokyo is the capital" in out["assembled_context"]

    def test_no_sources_needed_yields_empty_context(self):
        st = _state(understanding=QueryUnderstanding(
            mode=QueryMode.RESEARCH, needs_documents=False, needs_web=False,
        ))
        out = asyncio.run(gather_evidence(st))
        assert out["assembled_context"] == ""


# ── generate_answer ──────────────────────────────────────────────────────────


class TestGenerateAnswer:
    def test_cited_answer_passes_mechanical_check(self, monkeypatch):
        gen = FakeLLM(text=_JAPAN_EV_TEXT + " [E1].")
        _patch_llms(monkeypatch, FakeLLMs(generator=gen))
        out = generate_answer(_state(
            assembled_context=f"[E1] web: wb\n{_JAPAN_EV_TEXT}",
            evidence=[_japan_ev()], cite_map={"E1": "e1"},
        ))
        assert "[E1]" in out["answer"]
        assert not out["verification_errors"]

    def test_empty_context_is_honest(self, monkeypatch):
        out = generate_answer(_state(assembled_context="", evidence=[]))
        assert "enough reliable information" in out["answer"]

    def test_forged_citations_flagged(self, monkeypatch):
        gen = FakeLLM(text="Japan has 999 billion people [E7].")
        _patch_llms(monkeypatch, FakeLLMs(generator=gen))
        out = generate_answer(_state(
            assembled_context=f"[E1] web: wb\n{_JAPAN_EV_TEXT}",
            evidence=[_japan_ev()], cite_map={"E1": "e1"},
        ))
        assert any(e["issue"] == "INVALID_CITATION" for e in out["verification_errors"])


# ── verify_answer ────────────────────────────────────────────────────────────


class TestVerifyAnswer:
    def _setup(self, monkeypatch, verdict: Verdict, answer: str):
        _patch_llms(monkeypatch, FakeLLMs(verifier=FakeLLM(structured=verdict)))
        return _state(
            answer=answer,
            evidence=[_japan_ev()],
            cite_map={"E1": "e1"},
            assembled_context=f"[E1] web: wb\n{_JAPAN_EV_TEXT}",
        )

    def test_supported_answer_finalized(self, monkeypatch):
        v = Verdict(overall="supported", claims=[
            Claim(text=_JAPAN_EV_TEXT + ".", status=ClaimStatus.VERIFIED, evidence_ids=["E1"]),
        ])
        st = self._setup(monkeypatch, v, _JAPAN_EV_TEXT + " [E1].")
        out = verify_answer(st)
        assert out["final_status"] == "answered"
        assert route_after_verify(out) == "__end__"

    def test_fixable_gap_schedules_repair_search(self, monkeypatch):
        v = Verdict(overall="partial",
                    claims=[Claim(text="GDP grew 3%", status=ClaimStatus.UNVERIFIED)],
                    repair_queries=["Japan GDP growth 2025"])
        st = self._setup(monkeypatch, v, "Japan's GDP grew 3% last year.")
        out = verify_answer(st)
        assert out["repair_queries"] == ["Japan GDP growth 2025"]
        assert out["repair_count"] == 1
        assert route_after_verify(out) == "gather_evidence"

    def test_repair_budget_respected(self, monkeypatch):
        v = Verdict(overall="partial",
                    claims=[Claim(text="GDP grew 3%", status=ClaimStatus.UNVERIFIED)],
                    repair_queries=["Japan GDP growth 2025"])
        st = self._setup(monkeypatch, v, "Japan's GDP grew 3% last year.")
        st["repair_count"] = 1  # budget exhausted
        out = verify_answer(st)
        assert out["repair_queries"] == []
        assert out["final_status"] == "answered_with_caveats"
        assert "Caveats:" in out["answer"]

    def test_contradiction_never_loops(self, monkeypatch):
        v = Verdict(overall="unsupported",
                    claims=[Claim(text="Population is 999 billion", status=ClaimStatus.CONTRADICTED)],
                    repair_queries=["more searches"])
        st = self._setup(monkeypatch, v, "Japan's population is 999 billion.")
        out = verify_answer(st)
        assert out["repair_queries"] == []
        assert out["final_status"] == "answered_with_caveats"

    def test_stale_repair_queries_do_not_loop(self, monkeypatch):
        """Regression: after a repair pass, repair_queries lingered in graph
        state and routed verify_answer back to gathering forever."""
        v = Verdict(overall="supported", claims=[
            Claim(text=_JAPAN_EV_TEXT + " [E1].", status=ClaimStatus.VERIFIED, evidence_ids=["E1"]),
        ])
        st = self._setup(monkeypatch, v, _JAPAN_EV_TEXT + " [E1].")
        st["repair_count"] = 1          # budget already used by pass #1
        st["repair_queries"] = ["Japan GDP growth 2025"]  # stale from pass #1
        out = verify_answer(st)
        assert out["repair_queries"] == []   # must be explicitly cleared
        assert route_after_verify(out) == "__end__"

    def test_ambiguity_contradiction_asks_clarification(self, monkeypatch):
        """Contradictions caused by an ambiguous term -> ask, don't guess."""
        v = Verdict(
            overall="unsupported",
            claims=[Claim(text="CJP announced X", status=ClaimStatus.CONTRADICTED)],
            repair_queries=[],
            clarification_question="By CJP do you mean the Chief Justice of Pakistan or the Centre for Justice Policy?",
        )
        st = self._setup(monkeypatch, v, "The CJP announced X today.")
        out = verify_answer(st)
        assert out["final_status"] == "needs_clarification"
        assert "Chief Justice of Pakistan" in out["answer"]
        assert "conflicting findings" in out["answer"].lower()
        assert route_after_verify(out) == "__end__"


# ── evidence_state round-trip ────────────────────────────────────────────────


class TestEvidenceStateRoundTrip:
    def test_build_keeps_only_verified_support(self):
        ev_ok = Evidence(evidence_id="ok", text="verified fact", source_type=SourceType.WEB)
        ev_bad = Evidence(evidence_id="bad", text="unsupported", source_type=SourceType.WEB)
        claims = [
            Claim(text="verified fact", status=ClaimStatus.VERIFIED, evidence_ids=["ok"]),
            Claim(text="unsupported", status=ClaimStatus.UNVERIFIED),
        ]
        es = build_evidence_state([ev_ok, ev_bad], claims, turn=2)
        assert [ev.evidence_id for ev in es.established] == ["ok"]
        assert es.unresolved == ["unsupported"]

    def test_serialize_load_round_trip(self):
        ev = Evidence(evidence_id="ok", text="a verified fact", source_type=SourceType.WEB,
                      source_name="src", source_url="https://x.example")
        es = EvidenceState(turn=3, established=[ev], unresolved=["open question"])
        blob = serialize_for_storage(es)
        loaded = load_evidence_state_from_text("classify_and_plan → generate_answer\n" + blob)
        assert loaded is not None
        assert loaded.turn == 3
        assert loaded.established[0].text == "a verified fact"
        assert loaded.unresolved == ["open question"]

    def test_merge_dedupes_and_resets_unresolved(self):
        old = Evidence(evidence_id="old", text="an older verified fact", source_type=SourceType.WEB)
        new = Evidence(evidence_id="new", text="a fresh verified fact", source_type=SourceType.WEB)
        prior = EvidenceState(turn=1, established=[old], unresolved=["stale"])
        current = EvidenceState(turn=2, established=[new], unresolved=["fresh"])
        merged = merge_evidence_state(prior, current)
        assert {ev.evidence_id for ev in merged.established} == {"old", "new"}
        assert merged.unresolved == ["fresh"]


# ── key hygiene ──────────────────────────────────────────────────────────────


class TestKeyGuard:
    @pytest.mark.parametrize("key,expected", [
        ("fill this later", False),
        ("TODO", False),
        ("short", False),
        ("gsk_" + "a" * 40, True),
        (None, False),
    ])
    def test_valid_key(self, key, expected):
        assert valid_key(key) is expected


# ── full-graph smoke (fakes only) ────────────────────────────────────────────


class TestGraphSmoke:
    def test_conversational_short_circuit(self, monkeypatch):
        u = QueryUnderstanding(mode=QueryMode.CONVERSATIONAL)
        planner = FakeLLM(structured=u)
        gen = FakeLLM(text="Hi! Ask me to research something.")
        _patch_llms(monkeypatch, FakeLLMs(planner=planner, generator=gen))
        final = asyncio.run(rag_app.ainvoke(create_initial_state(query="hello there")))
        assert final["final_status"] == "conversational"
        assert final["answer"]

    def test_research_path_end_to_end(self, monkeypatch):
        u = QueryUnderstanding(
            mode=QueryMode.RESEARCH,
            rewritten_query="Japan population",
            needs_documents=False,
            needs_web=True,
            search_queries=["Japan population 2025"],
        )
        ev = Evidence(evidence_id="e1", text=_JAPAN_EV_TEXT,
                      source_type=SourceType.WEB, source_name="wb",
                      source_date=datetime(2024, 6, 1, tzinfo=timezone.utc))
        ev.metadata["cite_key"] = "E1"

        async def fake_web(queries, state):
            return [ev]

        monkeypatch.setattr(nodes, "_search_web", fake_web)
        _patch_rerank(monkeypatch)
        verdict = Verdict(overall="supported", claims=[
            Claim(text=_JAPAN_EV_TEXT + " [E1].", status=ClaimStatus.VERIFIED, evidence_ids=["E1"]),
        ])
        _patch_llms(monkeypatch, FakeLLMs(
            planner=FakeLLM(structured=u),
            generator=FakeLLM(text=_JAPAN_EV_TEXT + " [E1]."),
            verifier=FakeLLM(structured=verdict),
        ))
        final = asyncio.run(rag_app.ainvoke(create_initial_state(query="What is the population of Japan?")))
        assert final["final_status"] == "answered"
        assert "[E1]" in final["answer"]
