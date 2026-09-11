"""Embedding-based claim↔evidence support gate.

Citation-id resolution (does [E3] exist?) says nothing about whether E3's text
actually supports the sentence citing it. This module scores that cheaply with
a local MiniLM encoder — no API, deterministic, lazy-loaded on first use.

Failure mode is deliberately safe-directional: a below-threshold claim loses
its citation markers and is demoted to a caveat, never silently kept.
"""

from __future__ import annotations

import functools
import logging
import math
import re

from app.core.config import settings

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 1500  # evidence texts are truncated to this before encoding
_WINDOW_SENTENCES = 2  # sliding-window size, in sentences

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _windows(text: str) -> list[str]:
    """Sliding windows of ~_WINDOW_SENTENCES sentences (stride 1)."""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]
    if len(sentences) <= _WINDOW_SENTENCES:
        return [text] if text.strip() else []
    return [
        " ".join(sentences[i : i + _WINDOW_SENTENCES])
        for i in range(len(sentences) - _WINDOW_SENTENCES + 1)
    ]


@functools.lru_cache(maxsize=1)
def _local_embed_fn():
    """Lazy singleton: ONNX MiniLM via fastembed (no torch, no chromadb).

    Returns None when fastembed/model isn't available — callers fall back to
    keyword-overlap cosine (weaker, but the gate stays safe-directional).
    """
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

        def embed(texts: list[str]) -> list[list[float]]:
            return [list(map(float, vec)) for vec in model.embed(texts)]

        return embed
    except Exception:
        logger.warning("fastembed unavailable — support gate falls back to keyword cosine")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def max_support(
    sentence: str,
    evidence_texts: list[str],
    *,
    embed=None,
) -> float:
    """Best claim-vs-window similarity across the cited evidence texts.

    Each evidence chunk is cut into ~2-sentence sliding windows; the score is
    the max cosine between the claim and any window (whole chunks dilute a
    single supporting sentence). Returns 1.0 for an empty sentence (nothing to
    falsify) and 0.0 when no evidence texts are given.
    """
    if not sentence or not sentence.strip():
        return 1.0
    if not evidence_texts:
        return 0.0
    embed = embed or _local_embed_fn()
    windows = [w for t in evidence_texts for w in _windows(t[:_SNIPPET_CHARS])]
    if not windows:
        return 0.0
    if embed is None:
        return _best_window_cosine(sentence, evidence_texts)
    vecs = embed([sentence.strip()] + windows)
    claim_vec = vecs[0]
    return max(_cosine(claim_vec, ev) for ev in vecs[1:])


def _best_window_cosine(sentence: str, evidence_texts: list[str]) -> float:
    """Keyword-cosine fallback: best claim-vs-window overlap. Weak but safe-directional."""
    from collections import Counter

    def bag(t: str) -> Counter:
        return Counter(w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2)

    sb = bag(sentence)
    if not sb:
        return 1.0 if sentence.strip() else 0.0
    best = 0.0
    for t in evidence_texts:
        for w in _windows(t[:_SNIPPET_CHARS]):
            tb = bag(w)
            if not tb:
                continue
            common = set(sb) & set(tb)
            dot = sum(sb[x] * tb[x] for x in common)
            na = math.sqrt(sum(v * v for v in sb.values()))
            nb = math.sqrt(sum(v * v for v in tb.values()))
            best = max(best, dot / (na * nb) if na and nb else 0.0)
    return best


def gate_enabled() -> bool:
    return bool(settings.CITATION_SUPPORT_GATE)


def min_sim() -> float:
    return float(settings.CITATION_SUPPORT_MIN_SIM)
