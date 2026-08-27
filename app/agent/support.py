"""Embedding-based claim↔evidence support gate.

Citation-id resolution (does [E3] exist?) says nothing about whether E3's text
actually supports the sentence citing it. This module scores that cheaply with
a local MiniLM encoder — no API, deterministic, lazy-loaded on first use.

Failure mode is deliberately safe-directional: a below-threshold claim loses
its citation markers and is demoted to a caveat, never silently kept.
"""

from __future__ import annotations

import functools
import math

from app.core.config import settings

_SNIPPET_CHARS = 1500  # evidence texts are truncated to this before encoding


@functools.lru_cache(maxsize=1)
def _local_embed_fn():
    """Lazy singleton: ONNX MiniLM (same family Chroma ships locally)."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    ef = ONNXMiniLM_L6_V2()

    def embed(texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in ef(list(texts))]

    return embed


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
    """Best cosine similarity between ``sentence`` and any of ``evidence_texts``.

    Returns 1.0 for an empty sentence (nothing to falsify) and 0.0 when no
    evidence texts are given.
    """
    if not sentence or not sentence.strip():
        return 1.0
    if not evidence_texts:
        return 0.0
    embed = embed or _local_embed_fn()
    vecs = embed([sentence.strip()] + [t[:_SNIPPET_CHARS] for t in evidence_texts])
    claim_vec = vecs[0]
    return max(_cosine(claim_vec, ev) for ev in vecs[1:])


def gate_enabled() -> bool:
    return bool(settings.CITATION_SUPPORT_GATE)


def min_sim() -> float:
    return float(settings.CITATION_SUPPORT_MIN_SIM)
