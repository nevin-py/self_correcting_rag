"""Tests for the claim↔evidence support gate (app/agent/support.py + validator).

The real ONNX encoder is never loaded here: every test injects a deterministic
fake embedding so the offline suite stays network-free and fast.
"""

import pytest

from app.agent import support as support_gate
from app.agent.citation_validator import (
    strip_weak_markers,
    validate_answer_citations,
)
from app.agent.state import ClaimStatus, Evidence, SourceType


def _fake_embed_factory():
    """Orthogonal keyword vectors: same-topic texts → cosine 1.0, disjoint → 0.0."""

    def embed(texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0, 0.0]
            tl = t.lower()
            if "canberra" in tl or "capital" in tl:
                v[0] = 1.0
            if "ban" in tl or "brawl" in tl or "altercation" in tl:
                v[1] = 1.0
            if "spain" in tl or "argentina" in tl or "world cup final" in tl:
                v[2] = 1.0
            if "quantum" in tl:
                v[3] = 1.0
            out.append(v)
        return out

    return embed


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    """Replace the lazy real encoder everywhere; gate stays configurable per-test."""
    monkeypatch.setattr(support_gate, "_local_embed_fn", lambda: _fake_embed_factory())


@pytest.fixture()
def gate_on(monkeypatch):
    monkeypatch.setattr(support_gate, "gate_enabled", lambda: True)
    monkeypatch.setattr(support_gate, "min_sim", lambda: 0.55)


def _evidence(text: str, cite_key: str = "E1") -> Evidence:
    return Evidence(
        text=text,
        source_type=SourceType.WEB,
        source_name="test",
        source_url="https://example.com/a",
        metadata={"cite_key": cite_key},
    )


def test_max_support_ranks_best_evidence():
    fn = _fake_embed_factory()
    s = "The capital of Australia is Canberra."
    assert support_gate.max_support(s, ["Canberra is the capital of Australia."], embed=fn) == pytest.approx(1.0)
    assert support_gate.max_support(s, ["Players faced a ban after the brawl."], embed=fn) == pytest.approx(0.0)
    assert support_gate.max_support(s, [], embed=fn) == 0.0
    assert support_gate.max_support("", ["x"], embed=fn) == 1.0


def test_unsupported_claim_demoted_and_markers_recorded(gate_on):
    ev = _evidence("Four players received a ban after the altercation.")
    answer = "Spain won the World Cup final against Argentina. [E1]"
    result = validate_answer_citations(answer, [ev], entailment_gate=True)

    assert len(result.claims) == 1
    assert "does not semantically support" in result.claims[0].reasoning
    assert result.unsupported_citations and result.unsupported_citations[0][1] == ["E1"]
    assert any(e.issue == "WEAK_SUPPORT" for e in result.errors)


def test_supported_claim_stays_verified(gate_on):
    ev = _evidence("Canberra is the capital city of Australia, purpose-built for government.")
    answer = "The capital of Australia is Canberra. [E1]"
    result = validate_answer_citations(answer, [ev], entailment_gate=True)

    assert len(result.claims) == 1
    assert result.claims[0].status == ClaimStatus.VERIFIED
    assert not result.unsupported_citations
    assert not any(e.issue == "WEAK_SUPPORT" for e in result.errors)


def test_gate_disabled_preserves_legacy_behavior():
    ev = _evidence("Completely unrelated ban text about players.")
    answer = "The capital of Australia is Canberra. [E1]"
    # no `gate_on` fixture → autouse fixture has gate_enabled() -> False
    result = validate_answer_citations(answer, [ev], entailment_gate=True)

    assert result.claims[0].status == ClaimStatus.VERIFIED
    assert not result.unsupported_citations


def test_strip_weak_markers_removes_only_flagged_sentence(gate_on):
    ev_ban = _evidence("Four players received a ban after the altercation.")
    ev_cap = _evidence("Canberra is the capital city of Australia.", cite_key="E2")
    answer = (
        "Spain won the World Cup final against Argentina. [E1]\n\n"
        "The capital of Australia is Canberra. [E2]"
    )
    result = validate_answer_citations(answer, [ev_ban, ev_cap], entailment_gate=True)
    cleaned = strip_weak_markers(answer, result)

    assert "[E2]" in cleaned                      # supported citation untouched
    assert "[E1]" not in cleaned                  # unsupported marker stripped
    assert "World Cup final" in cleaned           # sentence text itself kept


def test_temporal_sort_key_nudges_authoritative_domains():
    from app.agent.nodes import _evidence_sort_key

    wiki = _evidence("event page", cite_key="E1")
    wiki.source_url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_final"
    blog = _evidence("tangential news", cite_key="E2")
    blog.source_url = "https://random-sports-blog.net/world-cup-reaction"
    wiki.rerank_score = blog.rerank_score = 0.5

    # Nudge only applies under temporal queries; ties otherwise stay stable.
    assert _evidence_sort_key(wiki, temporal=True) > _evidence_sort_key(blog, temporal=True)
    assert _evidence_sort_key(wiki) == _evidence_sort_key(blog)


def test_strip_weak_markers_noop_without_flags():
    answer = "Plain answer with no issues. [E1]"
    assert strip_weak_markers(answer, validate_answer_citations(answer, [])) == answer
