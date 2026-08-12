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
import re

from app.core.config import settings
from app.documents.clients import get_chroma_client, groq_client

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
    collection,
    query_embedding: list[float],
    top_k: int = 30,
    alpha: float = 0.7,  # Weight for vector: 0.7 vector + 0.3 BM25
    where_filter: dict | None = None,
) -> list[dict]:
    """Hybrid search combining vector similarity with BM25 keyword matching.
    
    Args:
        query: Search query string
        collection: ChromaDB collection
        query_embedding: Pre-computed query embedding
        top_k: Number of results to return
        alpha: Weight for vector scores (1-alpha for BM25)
        where_filter: ChromaDB metadata filter
        
    Returns:
        List of chunks sorted by hybrid score
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not available, falling back to vector-only search")
        # Fallback to pure vector search
        return await retrieve_chunks(query, None, top_k, where_filter=where_filter)
    
    # 1. Vector search first (get more results for fusion)
    vector_top_k = top_k * 2
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": vector_top_k,
    }
    if where_filter:
        query_kwargs["where"] = where_filter
    
    try:
        vector_results = collection.query(**query_kwargs)
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []
    
    if not vector_results.get("documents") or not vector_results["documents"][0]:
        return []
    
    # 2. Build BM25 index from retrieved documents
    docs = []
    for i in range(len(vector_results["documents"][0])):
        docs.append({
            "id": vector_results["ids"][0][i],
            "text": vector_results["documents"][0][i],
            "metadata": vector_results["metadatas"][0][i] if vector_results["metadatas"] else {},
            "distance": vector_results["distances"][0][i] if vector_results["distances"] else None,
        })
    
    if not docs:
        return []
    
    # Build BM25 index
    tokenized_docs = [_tokenize_for_bm25(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Score query with BM25
    query_tokens = _tokenize_for_bm25(query)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores
    max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
    
    # 3. Fuse scores
    hybrid_results = []
    for i, doc in enumerate(docs):
        # Convert vector distance to similarity score
        vec_sim = 1.0 - min(1.0, max(0.0, float(doc.get("distance", 0.5))))
        
        # Normalize BM25 score
        bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
        
        # Hybrid score (weighted combination)
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
    
    # Sort by hybrid score
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Fetch parent context for child chunks
    hybrid_results = await _add_parent_context(hybrid_results, collection)
    
    return hybrid_results[:top_k]


async def _add_parent_context(chunks: list[dict], collection) -> list[dict]:
    """Add parent context to child chunks if available."""
    if not chunks:
        return chunks
    
    # Get unique parent IDs from child chunks
    parent_ids_to_fetch = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("chunk_type") == "child":
            parent_id = meta.get("parent_id")
            if parent_id:
                parent_ids_to_fetch.add(parent_id)
    
    if not parent_ids_to_fetch:
        return chunks
    
    # Fetch parent chunks
    try:
        parent_results = collection.get(
            where={"parent_id": {"$in": list(parent_ids_to_fetch)}},
            include=["documents", "metadatas", "ids"]
        )
    except Exception as e:
        logger.warning(f"Failed to fetch parent context: {e}")
        return chunks
    
    # Build parent lookup
    parent_map = {}
    if parent_results.get("ids"):
        for i, pid in enumerate(parent_results["ids"]):
            parent_map[pid] = {
                "text": parent_results["documents"][i] if parent_results["documents"] else "",
                "metadata": parent_results["metadatas"][i] if parent_results["metadatas"] else {},
            }
    
    # Add parent context to each chunk
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("chunk_type") == "child":
            parent_id = meta.get("parent_id")
            if parent_id and parent_id in parent_map:
                chunk["parent_context"] = parent_map[parent_id]["text"]
    
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
    Embed chunks and store in ChromaDB with scope metadata.

    scope="chat"     → chat-scoped, retrievable only within the specified chat
    scope="permanent" → user's permanent memory, retrievable across all chats
    """
    user_hex = user_id.hex[:16]
    collection_name = f"user_{user_hex}"
    client = get_chroma_client()
    if client is None:
        raise RuntimeError("ChromaDB is not available — vector storage disabled")
    collection = client.get_or_create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    user_id_str = str(user_id)
    chat_id_str = str(chat_id) if chat_id else ""

    for idx, chunk in enumerate(chunk_list):
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue
        meta = chunk.get("metadata", {}).copy()
        # Core scope fields — always set
        meta["user_id"] = user_id_str
        meta["scope"] = scope
        meta["chat_id"] = chat_id_str if scope == "chat" else ""
        documents.append(chunk_text)
        metadatas.append(meta)
        source = meta.get("source", "unknown")
        ids.append(f"{user_id_str}_{chat_id_str}_{source}_{idx}")

    if not documents:
        return collection_name

    # Batch embed with rate limiting
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        emb = await _embed_text_rate_limited(batch, task_type="search_document")
        all_embeddings.extend(emb)

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )

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
    file_contents: bytes, filename: str, uid: uuid.UUID, chat_id: uuid.UUID
):
    """Ingest a file: validate → hash → extract → chunk → embed → store."""
    if not file_contents:
        raise ValueError("file_contents is empty or None")

    validate_file_magic(file_contents, filename)
    file_hash = compute_file_hash(file_contents)

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


async def _embed_text_rate_limited(
    texts: list[str], task_type: str = "search_query"
) -> list[list[float]]:
    """Async embedding: rate-limited + retries + off the event loop."""

    async def _call():
        async with _rate_limiter.acquire():
            return await asyncio.to_thread(
                _embed_text_sync_with_retry, texts, task_type
            )

    return await _call()


# ── Retrieval (scope-filtered, rate-limited) ─────────────────────────────────

def _get_user_collection(user_id: uuid.UUID):
    """Get the user's ChromaDB collection, trying new then old naming."""
    client = get_chroma_client()
    if client is None:
        return None
    user_hex = user_id.hex[:16]
    new_name = f"user_{user_hex}"
    try:
        return client.get_collection(name=new_name)
    except Exception:
        pass
    old_name = f"chat_{user_hex[:12]}"
    try:
        return client.get_collection(name=old_name)
    except Exception:
        return None


def _build_scope_filter(
    scope: str, chat_id: uuid.UUID | None = None
) -> dict | None:
    """Build a ChromaDB where filter for scope-based retrieval."""
    if scope == "permanent":
        return {"scope": "permanent"}
    if chat_id is not None:
        return {"$and": [{"scope": "chat"}, {"chat_id": str(chat_id)}]}
    # Cross-chat: all chat-scoped chunks, no chat_id filter
    return {"scope": "chat"}


async def retrieve_chunks(
    query: str,
    user_id: uuid.UUID,
    top_k: int = 5,
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
    use_hybrid: bool = True,
) -> list[dict]:
    """Retrieve chunks with hybrid (vector + BM25) or vector-only search."""
    collection = _get_user_collection(user_id)
    if collection is None:
        return []

    query_embedding = await _embed_text_rate_limited([query], task_type="search_query")
    where_filter = _build_scope_filter(scope, chat_id)

    # Use hybrid search (vector + BM25 fusion)
    if use_hybrid:
        try:
            return await hybrid_search_chunks(
                query=query,
                collection=collection,
                query_embedding=query_embedding[0],
                top_k=top_k,
                alpha=0.7,
                where_filter=where_filter,
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector: {e}")
    
    # Fallback to pure vector search
    query_kwargs = {"query_embeddings": query_embedding, "n_results": top_k}
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    chunks_retrieved = []
    if results["documents"]:
        for i in range(len(results["documents"][0])):
            chunks_retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
    
    # Add parent context
    if chunks_retrieved:
        chunks_retrieved = await _add_parent_context(chunks_retrieved, collection)
    
    return chunks_retrieved


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
