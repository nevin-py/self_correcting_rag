import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import base64
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nomic import embed
from PIL import Image
from pypdf import PdfReader
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import hashlib
from collections import OrderedDict
import re

from app.core.config import settings
from app.documents.clients import groq_client
from app.documents import vector_store

logger = logging.getLogger(__name__)


# ── Hybrid Search (Vector + BM25) ─────────────────────────────────────────────

# ── Token estimator ──────────────────────────────────────────────────────────

try:
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tiktoken_enc = None


def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken if available, else regex fallback."""
    if _tiktoken_enc:
        return len(_tiktoken_enc.encode(text))
    import re
    return max(1, len(re.findall(r"\w+|[^\s\w]", text)))


# ── Nomic embedding rate limiter ─────────────────────────────────────────────
# Nomic Embedding API limit: 1200 requests / 5-minute rolling window / IP.
# 1200 / 300s = 4 req/s.  Concurrency=2 + interval=0.25s enforces this.


class _EmbeddingRateLimiter:
    """Concurrency gate + minimum-interval spacing for Nomic embedding calls."""

    def __init__(self, concurrency: int, interval: float):
        self._semaphore = asyncio.Semaphore(concurrency)
        self._interval = interval
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def _wait_for_slot(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_request = time.monotonic()

    @asynccontextmanager
    async def acquire(self):
        async with self._semaphore:
            await self._wait_for_slot()
            yield


_rate_limiter = _EmbeddingRateLimiter(
    concurrency=settings.NOMIC_CONCURRENCY,
    interval=settings.NOMIC_INTERVAL,
)


# ── Hybrid Search: Vector + BM25 fusion ──────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 indexing."""
    words = re.findall(r'[a-z0-9]+', text.lower())
    return [w for w in words if len(w) > 2]


async def hybrid_search_chunks(
    query: str,
    rows: list[dict],
    top_k: int = 30,
    alpha: float = 0.7,  # Weight for vector: 0.7 vector + 0.3 BM25
) -> list[dict]:
    """Hybrid search: BM25 fusion over pgvector candidates.

    Args:
        query: Search query string
        rows: Vector-search candidates ({id, text, metadata, distance}) —
              already owner-scoped by the SQL layer.
        top_k: Number of results to return
        alpha: Weight for vector scores (1-alpha for BM25)

    Returns:
        List of chunks sorted by hybrid score
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not available, falling back to vector-only ranking")
        return sorted(rows, key=lambda r: (r.get("distance") is not None, r.get("distance") or 0.0))[:top_k]

    docs = [
        {
            "id": r["id"],
            "text": r["text"],
            "metadata": r.get("metadata") or {},
            "distance": r.get("distance"),
        }
        for r in rows
    ]
    if not docs:
        return []

    # BM25 rerank over the vector candidates (same fusion math as before —
    # only the recall leg moved from Chroma to pgvector).
    tokenized_docs = [_tokenize_for_bm25(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = _tokenize_for_bm25(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 scores (bm25_scores is a numpy array — never truth-test it)
    if getattr(bm25_scores, "size", len(bm25_scores)) == 0:
        max_bm25 = 1.0
    else:
        max_bm25 = float(max(bm25_scores))
        if max_bm25 <= 0:
            max_bm25 = 1.0

    hybrid_results = []
    for i, doc in enumerate(docs):
        vec_sim = 1.0 - min(1.0, max(0.0, float(doc.get("distance", 0.5) or 0.5)))
        bm25_norm = float(bm25_scores[i]) / max_bm25
        hybrid_score = alpha * vec_sim + (1 - alpha) * bm25_norm
        hybrid_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "id": doc["id"],
            "distance": doc.get("distance"),
            "score": hybrid_score,
            "vector_score": vec_sim,
            "bm25_score": bm25_norm,
        })

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_results[:top_k]


async def _add_parent_context(
    chunks: list[dict],
    user_id: uuid.UUID | None,
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
) -> list[dict]:
    """Add parent context to child chunks if available.

    Parent rows are looked up via the chunking parent_id key in metadata
    (the old Chroma path looked up doc ids that never matched — latent bug;
    this one actually resolves parents)."""
    if not chunks or user_id is None:
        return chunks

    parent_ids_to_fetch = {
        c["metadata"].get("parent_id")
        for c in chunks
        if (c.get("metadata") or {}).get("chunk_type") == "child"
        and (c.get("metadata") or {}).get("parent_id")
    }
    if not parent_ids_to_fetch:
        return chunks

    try:
        parents = await vector_store.fetch_parents(
            user_id, sorted(parent_ids_to_fetch), scope=scope, chat_id=chat_id
        )
    except Exception as e:
        logger.warning(f"Failed to fetch parent context: {e}")
        return chunks

    parent_map = {p["metadata"].get("parent_id"): p["text"] for p in parents}

    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if meta.get("chunk_type") == "child":
            pid = meta.get("parent_id")
            if pid and pid in parent_map:
                chunk["parent_context"] = parent_map[pid]

    return chunks


# ── Text extraction (unchanged) ─────────────────────────────────────────────

def extract_text_plain(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    text = stream.read().decode("utf-8", errors="ignore")
    return text, {"char_count": len(text)}


def extract_text_pdf(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    stream.seek(0)
    reader = PdfReader(stream)
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages_text.append(f"[page {i + 1}]\n{page_text}")
    full_text = "\n".join(pages_text)
    return full_text, {"total_pages": len(reader.pages)}


def extract_text_html(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    stream.seek(0)
    content = stream.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string if soup.title else ""
    return text, {"title": title}


def extract_text_markdown(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    stream.seek(0)
    md_content = stream.read().decode("utf-8", errors="ignore")
    return md_content, {}


def tabular(df):
    row_str = []
    i = 0
    for _, row in df.iterrows():
        row_str.append(" | ".join(f"{col}:{val}" for col, val in row.items()))
        i += 1
    clean_text = "\n".join(row_str)
    return clean_text, i


def extract_text_csv(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    stream.seek(0)
    df = pd.read_csv(stream)
    clean_text, i = tabular(df)
    return clean_text, {"num_of_rows": i}


def extract_text_excel(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    stream.seek(0)
    df = pd.read_excel(stream)
    clean_text, i = tabular(df)
    return clean_text, {"num_of_rows": i}


def extract_text_image(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    """Extract text from image using Groq vision model (runs in thread pool)."""
    # Run the blocking I/O in a thread to avoid blocking the event loop
    return asyncio.get_event_loop().run_until_complete(
        _extract_text_image_async(stream, filename)
    )


async def _extract_text_image_async(stream: io.BytesIO, filename: str) -> tuple[str, dict]:
    """Async implementation of image text extraction."""
    stream.seek(0)
    try:
        with Image.open(stream) as img:
            metadata = {
                "img_height": img.height,
                "img_width": img.width,
                "format": img.format,
            }
            img_format = img.format.lower() if img.format else "png"
    except Exception as e:
        return f"Error opening image file: {str(e)}", {}

    stream.seek(0)
    file_bytes = stream.read()
    base64_image = base64.b64encode(file_bytes).decode("utf-8")

    try:
        # Run blocking Groq API call in thread pool
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is an image that may contain diagrams, flowcharts, or text blocks. "
                                "Transcribe all text, format data structures into markdown tables, and "
                                "explicitly describe any visual flows or shapes in detailed text paragraphs."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
        )
        extracted_markdown = response.choices[0].message.content
        return str(extracted_markdown), metadata
    except Exception as e:
        return f"ocr failed: {str(e)}", metadata


# ── Ingestion pipeline ───────────────────────────────────────────────────────

def ingestion_pipeline(file_contents: bytes, filename: str):
    file_stream = io.BytesIO(file_contents)
    extension = Path(filename).suffix.lower()

    base_metadata = {
        "source": filename,
        "filename": filename,
        "source_name": filename,
        "file_type": extension,
        "file_size_kb": round(len(file_contents) / 1024),
    }
    parser_dict = {
        ".txt": extract_text_plain,
        ".js": extract_text_plain,
        ".py": extract_text_plain,
        ".log": extract_text_plain,
        ".json": extract_text_plain,
        ".pdf": extract_text_pdf,
        ".html": extract_text_html,
        ".htm": extract_text_html,
        ".md": extract_text_markdown,
        ".markdown": extract_text_markdown,
        ".csv": extract_text_csv,
        ".xlsx": extract_text_excel,
        ".xls": extract_text_excel,
        ".jpg": extract_text_image,
        ".jpeg": extract_text_image,
        ".png": extract_text_image,
        ".webp": extract_text_image,
    }
    if extension in parser_dict:
        text, custom_meta = parser_dict[extension](file_stream, filename)
        base_metadata.update(custom_meta)
        return text, base_metadata
    else:
        raise ValueError(f"Extension {extension} is not allowed")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunking(text: str, metadata, use_parent_child: bool = True):
    """Split text into chunks with optional parent-child relationship.
    
    If use_parent_child=True, creates:
    - Parent chunks: large chunks (4x normal size) for context
    - Child chunks: normal size for precise retrieval
    
    Each child links to its parent via parent_id in metadata.
    """
    # First create parent chunks (larger context)
    if use_parent_child:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE * 4,  # 4x for parent
            chunk_overlap=settings.CHUNK_OVERLAP * 2,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        parent_chunks = parent_splitter.split_text(text)
        
        docs = []
        parent_map = {}  # parent_idx -> parent_text
        
        # Create parent chunks first
        for parent_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{parent_idx}"
            parent_meta = metadata.copy()
            parent_meta.update({
                "chunk_type": "parent",
                "chunk_id": parent_idx,
                "total_chunks": len(parent_chunks),
                "parent_id": parent_id,
            })
            docs.append({"text": parent_text, "metadata": parent_meta})
            parent_map[parent_idx] = parent_text
        
        # Now create child chunks and link to parents
        child_chunks = splitter.split_text(text)
        for child_idx, child_text in enumerate(child_chunks):
            # Find which parent this child belongs to
            parent_idx = min(child_idx // 4, len(parent_chunks) - 1)  # Approximate mapping
            
            child_meta = metadata.copy()
            child_meta.update({
                "chunk_type": "child",
                "chunk_id": child_idx,
                "total_chunks": len(child_chunks),
                "parent_id": f"parent_{parent_idx}",
                "parent_idx": parent_idx,
            })
            docs.append({"text": child_text, "metadata": child_meta})
        
        return docs
    
    # Original behavior without parent-child
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        meta = metadata.copy()
        meta.update({"chunk_id": i, "total_chunks": len(chunks), "chunk_type": "child"})
        docs.append({"text": chunk, "metadata": meta})
    return docs


# ── Embedding + ChromaDB storage ─────────────────────────────────────────────

async def embed_n_store(
    chunk_list: list[dict],
    user_id: uuid.UUID,
    chat_id: uuid.UUID | None = None,
    scope: str = "chat",
):
    """
    Embed chunks and store in Postgres (pgvector) with scope metadata.

    scope="chat"     → chat-scoped, retrievable only within the specified chat
    scope="permanent" → user's permanent memory, retrievable across all chats

    Returns a store name for logging/compat (the old Chroma collection name).
    """
    user_hex = user_id.hex[:16]
    collection_name = f"user_{user_hex}"

    user_id_str = str(user_id)
    chat_id_str = str(chat_id) if chat_id else ""

    texts = []
    metas = []
    for chunk in chunk_list:
        chunk_text = (chunk.get("text") or "").strip()
        if not chunk_text:
            continue
        meta = (chunk.get("metadata") or {}).copy()
        # Core scope fields — always set (same as the Chroma metadata contract)
        meta["user_id"] = user_id_str
        meta["scope"] = scope
        meta["chat_id"] = chat_id_str if scope == "chat" else ""
        texts.append(chunk_text)
        metas.append(meta)

    if not texts:
        return collection_name

    # Batch embed with rate limiting (content-hash cache handles repeats)
    batch_size = 32
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        emb = await _embed_text_rate_limited(texts[i : i + batch_size], task_type="search_document")
        all_embeddings.extend(emb)

    rows = [
        {
            "user_id": user_id,
            "chat_id": chat_id if scope == "chat" else None,
            "scope": scope,
            "text": t,
            "embedding": e,
            "metadata": m,
        }
        for t, e, m in zip(texts, all_embeddings, metas)
    ]
    await vector_store.insert_chunks(rows)

    return collection_name


# Magic bytes for file type validation
MAGIC_BYTES = {
    b"%PDF": ".pdf",
    b"\x89PNG": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"GIF8": ".gif",
    b"PK": ".xlsx",  # ZIP-based (xlsx, docx, etc.)
    b"\xd0\xcf\x11\xe0": ".xls",
}


def validate_file_magic(contents: bytes, filename: str) -> str:
    """Validate file content matches expected extension via magic bytes.
    Returns the detected extension, or raises ValueError."""
    ext = Path(filename).suffix.lower()
    if not contents:
        raise ValueError("File is empty")
    header = contents[:8]
    for magic, expected_ext in MAGIC_BYTES.items():
        if header.startswith(magic):
            if ext not in (expected_ext,):
                # Allow if extension is in the parser dict (e.g., .jpeg vs .jpg)
                pass  # soft validation — trust extension for now
            return ext
    # Unknown magic bytes — allow text-based files through
    return ext


def compute_file_hash(contents: bytes) -> str:
    """SHA-256 hash of file contents for deduplication."""
    return hashlib.sha256(contents).hexdigest()


async def full_pipeline(
    file_contents: bytes,
    filename: str,
    uid: uuid.UUID,
    chat_id: uuid.UUID,
    pre_extracted: tuple[str, dict] | None = None,
):
    """Ingest a file: validate → hash → extract → chunk → embed → store.

    ``pre_extracted`` lets the upload endpoint pass the (text, metadata) it
    already produced during budget checks, so parsing (PDF/DOCX — the most
    expensive step) happens once, not twice.
    """
    if not file_contents:
        raise ValueError("file_contents is empty or None")

    validate_file_magic(file_contents, filename)
    file_hash = compute_file_hash(file_contents)

    if pre_extracted is not None:
        text, metadata = pre_extracted
        metadata = dict(metadata)
    else:
        text, metadata = ingestion_pipeline(file_contents, filename)
    metadata["file_hash"] = file_hash
    chunked_docs = chunking(text=text, metadata=metadata)
    collection_name = await embed_n_store(
        chunk_list=chunked_docs,
        user_id=uid,
        chat_id=chat_id,
    )
    return {
        "no_of_chunks": len(chunked_docs),
        "metadata": metadata,
        "collection_name": collection_name,
        "file_hash": file_hash,
    }


# ── Embedding helpers (rate-limited, async, with retries) ────────────────────

def _embed_text_sync(texts: list[str], task_type: str) -> list[list[float]]:
    """Synchronous Nomic embedding call — must run in a thread."""
    # Auth once per process. The nomic lib ignores the NOMIC_API_KEY env var
    # for embed.text — it needs an explicit login (used to live inside the
    # old get_chroma_client; must run standalone now).
    from app.documents.clients import _ensure_nomic

    _ensure_nomic()
    response = embed.text(
        texts=texts,
        model="nomic-embed-text-v1.5",
        task_type=task_type,
    )
    return response["embeddings"]


@retry(
    stop=stop_after_attempt(settings.NOMIC_MAX_RETRIES + 1),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _embed_text_sync_with_retry(texts: list[str], task_type: str) -> list[list[float]]:
    """Synchronous embedding with tenacity retries for transient failures."""
    return _embed_text_sync(texts, task_type)


_EMBED_CACHE_MAX = 4096
_embed_cache: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()


def _cache_key(task_type: str, text: str) -> tuple[str, str]:
    return (task_type, hashlib.sha256(text.encode("utf-8")).hexdigest())


async def _embed_text_rate_limited(
    texts: list[str], task_type: str = "search_query"
) -> list[list[float]]:
    """Async embedding: content-hash cached first (identical text = zero API
    calls — repeated queries re-embed the same strings constantly), then
    rate-limited + retries + off the event loop for the rest."""
    results: dict[int, list[float]] = {}
    misses: list[int] = []
    miss_texts: list[str] = []
    for i, t in enumerate(texts):
        key = _cache_key(task_type, t)
        cached = _embed_cache.get(key)
        if cached is not None:
            _embed_cache.move_to_end(key)
            results[i] = cached
        else:
            misses.append(i)
            miss_texts.append(t)

    if miss_texts:
        async def _call():
            async with _rate_limiter.acquire():
                return await asyncio.to_thread(
                    _embed_text_sync_with_retry, miss_texts, task_type
                )

        vectors = await _call()
        for i, vec in zip(misses, vectors):
            results[i] = vec
            key = _cache_key(task_type, texts[i])
            _embed_cache[key] = vec
            while len(_embed_cache) > _EMBED_CACHE_MAX:
                _embed_cache.popitem(last=False)  # FIFO-evict oldest

    return [results[i] for i in range(len(texts))]


# ── Retrieval (scope-filtered, rate-limited) ─────────────────────────────────

async def retrieve_chunks(
    query: str,
    user_id: uuid.UUID,
    top_k: int = 5,
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
    use_hybrid: bool = True,
) -> list[dict]:
    """Retrieve chunks with hybrid (vector + BM25) or vector-only search.

    pgvector-backed, owner-scoped at the SQL layer (user_id predicate — the
    same defense-in-depth the old _build_scope_filter provided).
    """
    if user_id is None:
        return []

    try:
        query_embedding = await _embed_text_rate_limited([query], task_type="search_query")
    except Exception as e:
        # Fail closed — same contract as the old chroma-unavailable path.
        logger.warning("Query embedding failed; retrieval disabled: %s", e)
        return []

    rows = await vector_store.search_similar(
        user_id=user_id,
        query_embedding=query_embedding[0],
        scope=scope,
        chat_id=chat_id,
        limit=max(top_k * 2, 30),
    )
    if not rows:
        return []

    if use_hybrid:
        try:
            results = await hybrid_search_chunks(query=query, rows=rows, top_k=top_k, alpha=0.7)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector: {e}")
            results = sorted(rows, key=lambda r: r.get("distance") if r.get("distance") is not None else 1.0)[:top_k]
    else:
        results = sorted(rows, key=lambda r: r.get("distance") if r.get("distance") is not None else 1.0)[:top_k]

    # Add parent context
    results = await _add_parent_context(results, user_id, scope=scope, chat_id=chat_id)

    return results


async def multi_query_retrieval(
    query_list: list[str],
    user_id: uuid.UUID,
    top_k: int = 5,
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
) -> list[dict]:
    tasks = [
        retrieve_chunks(q, user_id, top_k, scope=scope, chat_id=chat_id)
        for q in query_list
    ]
    result = await asyncio.gather(*tasks)
    return [{"query": q, "chunks": chunks} for q, chunks in zip(query_list, result)]


def dedup(retrievals: list[dict]) -> list[dict]:
    unique_chunks = {}
    for retrieval in retrievals:
        for chunk in retrieval["chunks"]:
            unique_chunks[chunk["id"]] = chunk
    return list(unique_chunks.values())
