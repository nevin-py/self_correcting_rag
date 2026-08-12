"""
Cross-encoder reranker using FlashRank.

Replaces keyword-based relevance scoring with a proper cross-attention model
that compares query-document pairs word-for-word.

FlashRank uses quantized ONNX models that run on CPU with no GPU required.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Lazy-loaded reranker singleton ───────────────────────────────────────────

_reranker = None
_init_attempted = False


def _get_reranker():
    """Lazy-load the FlashRank reranker on first use."""
    global _reranker, _init_attempted
    if _reranker is not None:
        return _reranker
    if _init_attempted:
        return None
    _init_attempted = True

    try:
        from flashrank import Ranker, RerankRequest
        # Default model: ms-marco-TinyBERT-L-2-v2 (~3MB), ultra-fast CPU reranking
        _reranker = Ranker()
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


def rerank(query: str, items: list[tuple[str, str]], top_k: int = 15) -> list[RankedItem]:
    """Rerank items by relevance to query using cross-encoder.

    Args:
        query: The user's question
        items: List of (source, text) tuples where source is "chunk" or "search"
        top_k: Number of top items to return

    Returns:
        List of RankedItem sorted by relevance (highest first)
    """
    if not items:
        return []

    ranker = _get_reranker()

    # Fallback to keyword scoring if reranker unavailable
    if ranker is None:
        return _keyword_rerank(query, items, top_k)

    try:
        return _flashrank_rerank(ranker, query, items, top_k)
    except Exception as e:
        logger.warning("FlashRank reranking failed: %s — falling back to keyword scoring", e)
        return _keyword_rerank(query, items, top_k)


def _flashrank_rerank(ranker, query: str, items: list[tuple[str, str]], top_k: int) -> list[RankedItem]:
    """Rerank using FlashRank cross-encoder."""
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
        logger.warning("BM25 reranking failed: %s — falling back to simple keyword scoring", e)
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

    # Normalize scores to [0, 1] range
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
            "and", "but", "or", "if", "the", "this", "that", "it", "its",
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
