"""pgvector-backed vector store — replaces the embedded ChromaDB client.

Rationale: Chroma persisted to local disk, which made the backend
state-hostile (docs vanish on container restart — breaks HF Spaces / Cloud
Run deploys). Chunks now live in Postgres alongside everything else.

Embeddings stay Nomic `nomic-embed-text-v1.5` (768 dims) — unchanged
semantics, only the storage engine moved.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import delete, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncLocalSession
from app.documents.models import DocumentChunk

logger = logging.getLogger(__name__)


def _vec_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector literal."""
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


async def insert_chunks(
    rows: list[dict],
    session_factory=None,
) -> int:
    """Insert chunk rows. Each row: {user_id, chat_id, scope, text, embedding, metadata}.

    Uses COPY-style batched inserts via the ORM; returns count inserted.
    """
    if not rows:
        return 0
    factory = session_factory or AsyncLocalSession
    async with factory() as db:  # type: AsyncSession
        db.add_all(
            DocumentChunk(
                user_id=r["user_id"],
                chat_id=r.get("chat_id"),
                scope=r.get("scope", "chat"),
                text=r["text"],
                embedding=r["embedding"],
                metadata_=r.get("metadata") or {},
            )
            for r in rows
        )
        await db.commit()
    return len(rows)


async def search_similar(
    user_id: uuid.UUID,
    query_embedding: list[float],
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
    limit: int = 60,
    session_factory=None,
) -> list[dict]:
    """Cosine-distance vector search, owner-scoped. Returns rows shaped like
    the old Chroma results: {id, text, metadata, distance}."""
    factory = session_factory or AsyncLocalSession
    vec = _vec_literal(query_embedding)

    owner = "user_id = :user_id"
    scope_clause = "scope = :scope"
    chat_clause = "chat_id = :chat_id" if scope == "chat" else "TRUE"
    sql = sa_text(
        f"""
        SELECT id::text, text, metadata,
               (embedding <=> (:vec)::vector) AS distance
        FROM document_chunks
        WHERE {owner} AND {scope_clause} AND {chat_clause}
        ORDER BY embedding <=> (:vec)::vector
        LIMIT :limit
        """
    )
    params = {
        "vec": vec,
        "user_id": str(user_id),
        "scope": scope,
        "limit": limit,
    }
    if scope == "chat":
        params["chat_id"] = str(chat_id) if chat_id else ""

    async with factory() as db:
        result = await db.execute(sql, params)
        rows = result.mappings().all()

    return [
        {
            "id": r["id"],
            "text": r["text"],
            "metadata": dict(r["metadata"] or {}),
            "distance": float(r["distance"]) if r["distance"] is not None else None,
        }
        for r in rows
    ]


async def fetch_parents(
    user_id: uuid.UUID,
    parent_id_keys: list[str],
    scope: str = "chat",
    chat_id: uuid.UUID | None = None,
    session_factory=None,
) -> list[dict]:
    """Fetch parent chunks by their metadata parent_id key ("parent_N" from
    chunking). Mirrors the old Chroma metadata lookup."""
    if not parent_id_keys:
        return []
    factory = session_factory or AsyncLocalSession
    chat_clause = "chat_id = :chat_id" if scope == "chat" else "TRUE"
    sql = sa_text(
        f"""
        SELECT id::text, text, metadata
        FROM document_chunks
        WHERE user_id = :user_id AND scope = :scope AND {chat_clause}
          AND metadata->>'parent_id' = ANY(:pids)
        """
    )
    params: dict = {
        "user_id": str(user_id),
        "scope": scope,
        "pids": list(parent_id_keys),
    }
    if scope == "chat":
        params["chat_id"] = str(chat_id) if chat_id else ""

    async with factory() as db:
        result = await db.execute(sql, params)
        rows = result.mappings().all()

    return [
        {"id": r["id"], "text": r["text"], "metadata": dict(r["metadata"] or {}), "distance": None}
        for r in rows
    ]


async def delete_chat_chunks(chat_id: uuid.UUID, user_id: uuid.UUID, session_factory=None) -> int:
    """Remove all chat-scoped chunks for a chat (purge paths)."""
    factory = session_factory or AsyncLocalSession
    async with factory() as db:
        result = await db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.chat_id == chat_id,
                DocumentChunk.user_id == user_id,
            )
        )
        await db.commit()
        return result.rowcount or 0


async def chunk_stats(user_id: uuid.UUID, session_factory=None) -> dict:
    """Lightweight diagnostics: total + per-chat counts for a user."""
    factory = session_factory or AsyncLocalSession
    async with factory() as db:
        total = (
            await db.execute(
                sa_text("SELECT count(*) FROM document_chunks WHERE user_id = :u"),
                {"u": str(user_id)},
            )
        ).scalar()
    return {"total_chunks": int(total or 0)}
