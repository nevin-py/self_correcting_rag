"""Reranking with pluggable backends and a hard memory ceiling.

Backends (settings.RERANK_BACKEND):
- ``openrouter`` (default): POST /api/v1/rerank with the free Nemotron rerank
  model. Zero resident RAM, zero cost, no ONNX runtime — but adds ~1-4s and
  depends on a remote API (rate limits on free tiers).
- ``flashrank``: local ONNX cross-encoder. ~40MB resident and a ~1.3MB/passage
  activation spike that OOM-killed a 512MB container at 60 realistic-length
  passages — so the local path is (a) hard-capped at RERANK_MAX_PASSAGES,
  (b) run in a worker thread so it never blocks the event loop, and
  (c) guarded by a process-wide lock so concurrent spikes cannot stack.

Every path funnels into the same keyword/BM25 fallback on failure, so a rerank
outage degrades gracefully to pre-rerank ordering instead of failing the turn.
"""

import asyncio
import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Lazy-loaded reranker singleton ───────────────────────────────────────────

_reranker = None
_init_attempted = False
# Serializes FlashRank inference: rerank() is CPU-bound ONNX work; one at a
# time keeps peak memory bounded and the event loop responsive.
_flashrank_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _flashrank_lock
    if _flashrank_lock is None:
        _flashrank_lock = asyncio.Lock()
    return _flashrank_lock


def _get_reranker():
    """Lazy-load the FlashRank reranker on first use (fallback path only)."""
    global _reranker, _init_attempted
    if _reranker is not None:
        return _reranker
    if _init_attempted:
        return None
    _init_attempted = True

    try:
        from flashrank import Ranker, Config  # noqa: F401  (Ranker used below)
        # Default model: ms-marco-TinyBERT-L-2-v2 (~3MB), ultra-fast CPU reranking
        _reranker = Ranker(cache_dir=settings.FLASHRANK_CACHE_DIR)
        logger.info("FlashRank reranker loaded: ms-marco-TinyBERT-L-2-v2")
    except Exception:
        logger.exception("Failed to load FlashRank reranker — falling back to keyword scoring")
        _reranker = None
    return _reranker


@dataclass
class RankedItem:
    """A document with its relevance score."""
    text: str
    score: float
    source: str  # "chunk" or "search"


async def rerank(query: str, items: list[tuple[str, str]], top_k: int = 15) -> list[RankedItem]:
    """Rerank items by relevance to the query.

    Args:
        query: The user's question
        items: List of (source, text) tuples where source is "chunk" or "search",
               in pre-rerank (retrieval-merge) order — the passage cap keeps the
               first RERANK_MAX_PASSAGES, so callers should order best-first
               when they can.
        top_k: Number of top items to return

    Returns:
        List of RankedItem sorted by score (highest first). Items beyond top_k
        (or beyond the passage cap) are simply absent — callers treat missing
        scores as "keep the retrieval score".
    """
    if not items:
        return []

    backend = (settings.RERANK_BACKEND or "openrouter").strip().lower()
    if backend == "flashrank":
        return await _rerank_flashrank(query, items, top_k)

    try:
        return await _rerank_openrouter(query, items, top_k)
    except Exception as e:
        logger.warning("OpenRouter rerank failed (%s) — falling back to FlashRank", e)
        return await _rerank_flashrank(query, items, top_k)


async def _rerank_openrouter(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Cross-encoder rerank via the OpenRouter rerank API (Cohere-compatible).

    Uses the server OPENROUTER_API_KEY — reranking is system infrastructure,
    not per-user model choice. Raises on any failure so the caller's fallback
    chain (FlashRank → keyword) takes over.
    """
    from app.core.http_client import get_http_client

    api_key = (settings.OPENROUTER_API_KEY or "").strip()
    if not api_key or len(api_key) < 20:
        raise RuntimeError("no server OPENROUTER_API_KEY configured")

    capped = items[: settings.RERANK_MAX_PASSAGES]
    if len(items) > len(capped):
        logger.info("rerank: capped %d passages → %d (RERANK_MAX_PASSAGES)", len(items), len(capped))

    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": [text for _, text in capped],
        "top_n": min(top_k, len(capped)),
    }
    client = get_http_client()
    resp = await client.post(
        "https://openrouter.ai/api/v1/rerank",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=settings.RERANK_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    ranked: list[RankedItem] = []
    for r in results:
        idx = int(r.get("index", -1))
        if 0 <= idx < len(capped):
            source, text = capped[idx]
            ranked.append(RankedItem(text=text, score=float(r.get("relevance_score", 0.0)), source=source))
    logger.debug("OpenRouter reranked %d passages → %d results (%s)", len(capped), len(ranked), settings.RERANK_MODEL)
    return ranked


async def _rerank_flashrank(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Local ONNX cross-encoder path — capped, locked, thread-offloaded."""
    capped = items[: min(settings.RERANK_MAX_PASSAGES, 40)]
    if len(items) > len(capped):
        logger.info("rerank: FlashRank capped %d passages → %d (RAM guard)", len(items), len(capped))
    try:
        async with _get_lock():
            return await asyncio.to_thread(_flashrank_rerank_sync, query, capped, top_k)
    except Exception as e:
        logger.warning("FlashRank reranking failed: %s — falling back to keyword scoring", e)
        return _keyword_rerank(query, capped, top_k)


def _flashrank_rerank_sync(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Blocking FlashRank inference — always called via asyncio.to_thread."""
    ranker = _get_reranker()
    if ranker is None:
        return _keyword_rerank(query, items, top_k)

    from flashrank import RerankRequest

    # FlashRank expects passages with "id" and "text" fields
    passages = [{"id": i, "text": text} for i, (_, text) in enumerate(items)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    # Map results back to RankedItem
    ranked = []
    for result in results[:top_k]:
        idx = result["id"]
        source, text = items[idx]
        # FlashRank returns scores in [0, 1] range
        ranked.append(RankedItem(text=text, score=result["score"], source=source))

    logger.debug("FlashRank reranked %d items → top %d", len(items), len(ranked))
    return ranked


def _keyword_rerank(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Fallback: keyword overlap scoring (no external dependencies)."""
    # Try BM25 first (better than simple keyword overlap)
    try:
        return _bm25_rerank(query, items, top_k)
    except Exception as e:
        logger.warning("BM25 rerank failed: %s — falling back to simple keyword scoring", e)
        return _simple_keyword_rerank(query, items, top_k)


def _bm25_rerank(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Rerank using BM25 algorithm (Okapi BM25)."""
    from rank_bm25 import BM25Okapi
    import re

    if not items:
        return []

    # Tokenize all documents
    def tokenize(text: str) -> list[str]:
        words = re.findall(r'[a-z0-9]+', text.lower())
        return [w for w in words if len(w) > 2]

    # Build BM25 index
    tokenized_docs = [tokenize(text) for _, text in items]
    bm25 = BM25Okapi(tokenized_docs)

    # Score query
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Normalize scores to [0, 1]
    max_score = max(scores) if scores and max(scores) > 0 else 1.0

    ranked = []
    for i, (source, text) in enumerate(items):
        normalized_score = scores[i] / max_score if max_score > 0 else 0.0
        ranked.append(RankedItem(text=text, score=normalized_score, source=source))

    # Sort by score descending
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]


def _simple_keyword_rerank(query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Simple keyword overlap scoring (fallback when BM25 unavailable)."""
    import re

    def tokenize(text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "and", "or", "if", "the", "this", "that", "it", "its",
        }
        words = re.findall(r'[a-z0-9]+', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    query_tokens = tokenize(query)
    if not query_tokens:
        return [RankedItem(text=text, score=0.0, source=source) for source, text in items[:top_k]]

    scored = []
    for source, text in items:
        text_tokens = tokenize(text)
        if not text_tokens:
            scored.append(RankedItem(text=text, score=0.0, source=source))
            continue

        overlap = query_tokens & text_tokens
        # Weighted: query coverage matters more than precision
        query_coverage = len(overlap) / len(query_tokens)
        text_precision = len(overlap) / len(text_tokens)
        score = 0.7 * query_coverage + 0.3 * text_precision
        scored.append(RankedItem(text=text, score=score, source=source))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]
