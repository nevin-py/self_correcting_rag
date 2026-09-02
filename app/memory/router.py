"""Memory API — list pgvector chunks for the current user."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import User
from app.auth.router import get_current_user
from app.core.database import get_db
from app.documents.models import DocumentChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory"])


class ChunkOut(BaseModel):
    id: str
    document_preview: str
    filename: str | None = None
    chat_id: str | None = None
    chunk_index: int | None = None
    parent_id: str | None = None
    file_hash: str | None = None
    chunk_type: str | None = None


class ChunkListResponse(BaseModel):
    collection: str
    total: int
    limit: int
    offset: int
    chunks: list[ChunkOut]


class FileAggregate(BaseModel):
    filename: str
    chat_id: str | None = None
    chunk_count: int
    file_hash: str | None = None


class MemoryStatsResponse(BaseModel):
    collection: str
    chunk_count: int
    files: list[FileAggregate]


@router.get("/chunks", response_model=ChunkListResponse)
async def list_chunks(
    current_user: User = Depends(get_current_user),
    chat_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(default=None, description="Substring filter on document text"),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.user_id == current_user.user_id)
        .order_by(DocumentChunk.created_at, DocumentChunk.id)
    )
    if chat_id:
        stmt = stmt.where(DocumentChunk.chat_id == chat_id)
    if q:
        stmt = stmt.where(DocumentChunk.text.ilike(f"%{q}%"))

    total = (
        await db.execute(
            select(func.count()).select_from(stmt.subquery())
        )
    ).scalar() or 0
    rows = (await db.execute(stmt.offset(offset).limit(limit))).scalars().all()

    def _out(c: DocumentChunk) -> ChunkOut:
        meta = c.metadata_ or {}
        return ChunkOut(
            id=str(c.id),
            document_preview=(c.text or "")[:200],
            filename=meta.get("filename") or meta.get("source"),
            chat_id=str(c.chat_id) if c.chat_id else None,
            chunk_index=meta.get("chunk_id"),
            parent_id=meta.get("parent_id"),
            file_hash=meta.get("file_hash"),
            chunk_type=meta.get("chunk_type"),
        )

    return ChunkListResponse(
        collection=f"user_{current_user.user_id.hex[:16]}",
        total=int(total),
        limit=limit,
        offset=offset,
        chunks=[_out(c) for c in rows],
    )


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(
                DocumentChunk.metadata_,
                func.count().label("cnt"),
            )
            .where(DocumentChunk.user_id == current_user.user_id)
            .group_by(DocumentChunk.metadata_)
        )
    ).all()

    agg: dict[tuple[str, str | None], dict] = {}
    total = 0
    for meta, cnt in rows:
        meta = meta or {}
        fname = meta.get("filename") or meta.get("source") or "unknown"
        cid = str(meta["chat_id"]) if meta.get("chat_id") else None
        key = (fname, cid)
        if key not in agg:
            agg[key] = {
                "filename": fname,
                "chat_id": cid,
                "chunk_count": 0,
                "file_hash": meta.get("file_hash"),
            }
        agg[key]["chunk_count"] += int(cnt)
        total += int(cnt)

    files = [FileAggregate(**v) for v in agg.values()]
    return MemoryStatsResponse(
        collection=f"user_{current_user.user_id.hex[:16]}",
        chunk_count=total,
        files=files,
    )
