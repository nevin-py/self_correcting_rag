"""Memory API — list real Chroma chunks for the current user."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth.models import User
from app.auth.router import get_current_user
from app.documents.clients import get_chroma_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory"])


def _collection_name(user_id: uuid.UUID) -> str:
    return f"user_{user_id.hex[:16]}"


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
):
    client = get_chroma_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    name = _collection_name(current_user.user_id)
    try:
        collection = client.get_or_create_collection(name=name)
    except Exception as exc:
        logger.exception("Failed to open collection %s", name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    where = {"chat_id": chat_id} if chat_id else None

    try:
        kwargs: dict = {
            "include": ["documents", "metadatas"],
            "limit": limit,
            "offset": offset,
        }
        if where:
            kwargs["where"] = where
        raw = collection.get(**kwargs)
    except TypeError:
        kwargs = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        raw = collection.get(**kwargs)
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        sliced = list(zip(ids, docs, metas))[offset : offset + limit]
        raw = {
            "ids": [s[0] for s in sliced],
            "documents": [s[1] for s in sliced],
            "metadatas": [s[2] for s in sliced],
        }

    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    chunks: list[ChunkOut] = []
    for i, doc_id in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {} or {}
        preview = (doc or "")[:400]
        if q and q.lower() not in (doc or "").lower():
            continue
        chunk_index = meta.get("chunk_id")
        try:
            chunk_index = int(chunk_index) if chunk_index is not None else None
        except (TypeError, ValueError):
            chunk_index = None
        chunks.append(
            ChunkOut(
                id=str(doc_id),
                document_preview=preview,
                filename=meta.get("filename") or meta.get("source"),
                chat_id=str(meta["chat_id"]) if meta.get("chat_id") else None,
                chunk_index=chunk_index,
                parent_id=str(meta["parent_id"]) if meta.get("parent_id") else None,
                file_hash=meta.get("file_hash"),
                chunk_type=meta.get("chunk_type"),
            )
        )

    try:
        total = collection.count()
    except Exception:
        total = len(chunks) + offset

    return ChunkListResponse(
        collection=name,
        total=total,
        limit=limit,
        offset=offset,
        chunks=chunks,
    )


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats(current_user: User = Depends(get_current_user)):
    client = get_chroma_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    name = _collection_name(current_user.user_id)
    try:
        collection = client.get_or_create_collection(name=name)
        raw = collection.get(include=["metadatas"])
    except Exception as exc:
        logger.exception("Failed stats for %s", name)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    metas = raw.get("metadatas") or []
    agg: dict[tuple[str, str | None], dict] = {}
    for meta in metas:
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
        agg[key]["chunk_count"] += 1

    files = [
        FileAggregate(**v)
        for v in sorted(agg.values(), key=lambda x: (-x["chunk_count"], x["filename"]))
    ]
    return MemoryStatsResponse(
        collection=name,
        chunk_count=len(metas),
        files=files,
    )
