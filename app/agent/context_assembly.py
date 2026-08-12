"""
Context assembly with token budgeting, cross-encoder reranking, and deduplication.

Pipeline:
1. Clean boilerplate (regex, <1ms)
2. Deduplicate across sources (threshold 0.5)
3. Fast pre-filter: keep top 25 via lightweight BM25
4. FlashRank cross-encoder reranking on top 25
5. Rank-then-fill greedy assembly within token budget
"""

import time

import logging
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log

from app.agent.reranker import rerank, RankedItem

logger = logging.getLogger(__name__)

# ── Pre-filter constants ─────────────────────────────────────────────────────

FLASHRANK_MAX_CANDIDATES = 25  # Keep small for speed (<20ms)

# ── Token counting ───────────────────────────────────────────────────────────

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken. Falls back to word-based estimate."""
    if _encoder:
        return len(_encoder.encode(text))
    return max(1, int(len(text.split()) * 1.3))


# ── Deduplication ────────────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """Fast similarity using SequenceMatcher (no external deps)."""
    return SequenceMatcher(None, a[:500], b[:500]).ratio()


def _deduplicate(items: list[tuple[str, str]], threshold: float = 0.5) -> list[tuple[str, str]]:
    """Remove near-duplicate texts. Keeps the first occurrence.
    
    Args:
        items: List of (source, text) tuples
        threshold: Similarity threshold for deduplication (0-1)
    
    Returns:
        Deduplicated list of (source, text) tuples
    """
    if len(items) <= 1:
        return items

    unique: list[tuple[str, str]] = []
    for source, text in items:
        is_dup = False
        for _, existing in unique:
            if _text_similarity(text, existing) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append((source, text))

    removed = len(items) - len(unique)
    if removed > 0:
        logger.debug("Deduplication: removed %d near-duplicates from %d items", removed, len(items))
    return unique


# ── Boilerplate cleaning (regex, <1ms) ───────────────────────────────────────

_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
_HTML_ENTITY = re.compile(r'&[a-z]+;|&#\d+;')
_NAV_NOISE = re.compile(
    r'(?:menu|navigation|skip to|click here|read more|subscribe|sign up|log in|home|about|contact|privacy|terms|cookie)\b',
    re.IGNORECASE
)
_REPEATED_PUNCT = re.compile(r'[.!?]{3,}|[-=]{3,}|[*]{3,}')
_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{3,}')
_BLANK_LINES = re.compile(r'^\s+$', re.MULTILINE)


def clean_boilerplate(text: str) -> str:
    """Remove URLs, HTML entities, navigation noise, and normalize whitespace.
    Runs in <1ms on typical search snippets.
    """
    text = _URL_PATTERN.sub('', text)
    text = _HTML_ENTITY.sub(' ', text)
    text = _NAV_NOISE.sub('', text)
    text = _REPEATED_PUNCT.sub('', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    text = _BLANK_LINES.sub('', text)
    return text.strip()


# ── BM25 pre-filter (fast, no external deps) ─────────────────────────────────

_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'and', 'but', 'or', 'if', 'not', 'this', 'that', 'it', 'its',
})


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens, remove stopwords."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute BM25 score for a single document."""
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt in tf:
            f = tf[qt]
            # BM25 TF normalization
            tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
            score += tf_norm  # IDF omitted for speed (all docs from same query)
    return score


def bm25_prefilter(query: str, items: list[tuple[str, str]], top_k: int = FLASHRANK_MAX_CANDIDATES) -> list[tuple[str, str]]:
    """Fast BM25 pre-filter to reduce candidates before cross-encoder.
    
    Args:
        query: The user's question
        items: List of (source, text) tuples
        top_k: Number of top items to keep (default 40)
    
    Returns:
        Top-k items sorted by BM25 score (highest first)
    """
    if len(items) <= top_k:
        return items

    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return items[:top_k]

    # Tokenize all documents
    doc_token_lists = [_bm25_tokenize(text) for _, text in items]
    avg_dl = sum(len(dt) for dt in doc_token_lists) / max(len(doc_token_lists), 1)

    # Score each document
    scored = []
    for i, (_, text) in enumerate(items):
        score = _bm25_score(query_tokens, doc_token_lists[i], avg_dl)
        scored.append((score, i))

    # Sort by score descending, return top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [items[idx] for _, idx in scored[:top_k]]


# ── Context assembly ─────────────────────────────────────────────────────────

@dataclass
class Budget:
    """Token budget for a specific context section."""
    total_tokens: int
    label: str


# Pre-defined budgets (in tokens, not chars)
PLANNER_BUDGET = Budget(total_tokens=1500, label="planner")
GENERATOR_BUDGET = Budget(total_tokens=5000, label="generator")
HALLUCINATION_BUDGET = Budget(total_tokens=2000, label="hallucination_checker")


def assemble_context(
    query: str,
    chunks: list[str],
    search_results: list[str],
    budget: Budget,
    include_header: bool = True,
    rerank_top_k: int = 15,
) -> tuple[str, int]:
    """Assemble context within a token budget using cross-encoder reranking."""
    total_start = time.perf_counter()

    # 1. Clean boilerplate and tag source
    all_items: list[tuple[str, str]] = []
    for item in chunks:
        cleaned = clean_boilerplate(item)
        if cleaned and len(cleaned) > 20:
            all_items.append(("chunk", cleaned))
    for item in search_results:
        cleaned = clean_boilerplate(item)
        if cleaned and len(cleaned) > 20:
            all_items.append(("search", cleaned))

    if not all_items:
        return "", 0

    # 2. Deduplicate
    deduped = _deduplicate(all_items, threshold=0.5)

    # 3. BM25 pre-filter
    prefiltred = bm25_prefilter(query, deduped, top_k=FLASHRANK_MAX_CANDIDATES)

    # 4. FlashRank cross-encoder reranking
    rerank_start = time.perf_counter()
    ranked: list[RankedItem] = rerank(query, prefiltred, top_k=rerank_top_k)
    rerank_elapsed = time.perf_counter() - rerank_start

    logger.info("[context] %d raw → %d deduped → %d BM25 → %d FlashRank (%.1fs)",
               len(all_items), len(deduped), len(prefiltred), len(ranked), rerank_elapsed)

    # 5. Rank-then-fill
    selected_chunks: list[str] = []
    selected_search: list[str] = []
    tokens_used = 0

    overhead = 200 if include_header else 0
    remaining_budget = budget.total_tokens - overhead

    for item in ranked:
        item_tokens = count_tokens(item.text)

        if tokens_used + item_tokens > remaining_budget:
            chars_per_token = len(item.text) / max(item_tokens, 1)
            remaining_chars = int((remaining_budget - tokens_used) * chars_per_token)
            if remaining_chars > 100:
                truncated = item.text[:remaining_chars] + f"\n... [truncated to fit {budget.label} budget]"
                if item.source == "chunk":
                    selected_chunks.append(truncated)
                else:
                    selected_search.append(truncated)
                tokens_used += count_tokens(truncated)
            continue

        tokens_used += item_tokens
        if item.source == "chunk":
            selected_chunks.append(item.text)
        else:
            selected_search.append(item.text)

    # 6. Assemble final text
    parts = []
    if include_header:
        if selected_chunks:
            parts.append("Document Evidence:\n" + "\n---\n".join(selected_chunks))
        if selected_search:
            parts.append("Web Search Results:\n" + "\n---\n".join(selected_search))
    else:
        parts.extend(selected_chunks)
        parts.extend(selected_search)

    assembled = "\n\n".join(parts)
    actual_tokens = count_tokens(assembled)
    total_elapsed = time.perf_counter() - total_start

    logger.info("[context] %s: %d tokens selected (budget %d) — %.1fs",
               budget.label, actual_tokens, budget.total_tokens, total_elapsed)

    return assembled, actual_tokens


def assemble_hallucination_context(
    query: str,
    answer: str,
    chunks: list[str],
    search_results: list[str],
) -> tuple[str, int]:
    """Assemble context for the hallucination checker.

    The checker needs: query + answer + key supporting facts.
    It does NOT need the full evidence — just enough to verify claims.
    """
    budget = HALLUCINATION_BUDGET

    # Reserve space for query + answer + hallucination prompt
    answer_tokens = count_tokens(answer)
    overhead = answer_tokens + 300  # prompt + query
    remaining = budget.total_tokens - overhead

    if remaining < 200:
        return f"Query: {query}\n\nAnswer to verify:\n{answer}", budget.total_tokens

    # Assemble evidence within remaining budget
    evidence_budget = Budget(total_tokens=remaining, label="hc_evidence")
    evidence_text, _ = assemble_context(
        query=query,
        chunks=chunks,
        search_results=search_results,
        budget=evidence_budget,
        include_header=False,
        rerank_top_k=15,  # Hallucination checker gets more items to verify claims
    )

    full_text = f"Query: {query}\n\nEvidence:\n{evidence_text}\n\nAnswer to verify:\n{answer}"
    return full_text, count_tokens(full_text)
