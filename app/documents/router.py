import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, status, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db, AsyncLocalSession
from app.core.config import settings
from app.core.usage import enforce_ingest_budget, record_usage
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats
from app.agent.message_models import ChatMessage
from app.documents.models import IngestionLog
from app.documents.signing import signed_file_path, verify_file_sig
from app.documents.service import (
    full_pipeline,
    compute_file_hash,
    estimate_tokens,
    ingestion_pipeline,
    chunking,
)
from app.documents.schemas import IngestionLogResponse, IngestionStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


async def _record_ingest_message(session: AsyncSession, chat_id: uuid.UUID, filename: str) -> None:
    """Persist an ingest event so later turns can scope to this file without keyword lists."""
    from sqlalchemy import func

    result = await session.execute(
        select(func.coalesce(func.max(ChatMessage.sequence), 0)).where(ChatMessage.chat_id == chat_id)
    )
    next_seq = int(result.scalar_one() or 0) + 1
    session.add(
        ChatMessage(
            chat_id=chat_id,
            role="system",
            content=f"Document ingested: {filename}",
            sequence=next_seq,
        )
    )
    await session.commit()


async def _run_ingestion(
    ingestion_id: uuid.UUID,
    file_contents: bytes,
    filename: str,
    uid: uuid.UUID,
    chat_id: uuid.UUID,
    token_estimate: int,
    pre_extracted: tuple[str, dict] | None = None,
):
    """Run the full ingestion pipeline (async) with status tracking."""
    async with AsyncLocalSession() as session:
        log = await session.get(IngestionLog, ingestion_id)
        if log:
            log.status = "processing"
            await session.commit()
    logger.info("Ingestion started: id=%s file=%s user=%s chat=%s", ingestion_id, filename, uid, chat_id)

    try:
        await full_pipeline(
            file_contents=file_contents,
            filename=filename,
            uid=uid,
            chat_id=chat_id,
            pre_extracted=pre_extracted,
        )
        async with AsyncLocalSession() as session:
            log = await session.get(IngestionLog, ingestion_id)
            if log:
                log.status = "completed"
                log.ingest_token_count = token_estimate
                await session.commit()
        async with AsyncLocalSession() as session:
            await _record_ingest_message(session, chat_id, filename)
        async with AsyncLocalSession() as session:
            await record_usage(session, uid, "ingest_tokens", amount=token_estimate)
        logger.info("Ingestion completed: id=%s file=%s tokens=%s", ingestion_id, filename, token_estimate)
    except Exception as exc:
        logger.exception("Ingestion failed: id=%s file=%s user=%s chat=%s", ingestion_id, filename, uid, chat_id)
        async with AsyncLocalSession() as session:
            log = await session.get(IngestionLog, ingestion_id)
            if log:
                log.status = "failed"
                log.error_message = str(exc)[:1000]
                await session.commit()


@router.post("/upload_file", response_model=IngestionLogResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload(
    background_tasks: BackgroundTasks,
    chat_id: uuid.UUID = Query(..., description="Chat ID to associate the document with"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
):
    """Upload a document for ingestion. Returns an ingestion_id for status tracking."""
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    file_content = await file.read()
    await file.close()

    if not file_content:
        raise HTTPException(status_code=400, detail="File is empty")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    max_size = 50 * 1024 * 1024  # 50MB
    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # Parsing (PDF/DOCX) is CPU-bound — run it off the event loop so a 50MB
    # upload doesn't freeze every concurrent request.
    try:
        text, metadata = await asyncio.to_thread(ingestion_pipeline, file_content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    token_estimate = estimate_tokens(text)
    if token_estimate > settings.MAX_FILE_TOKENS:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (~{token_estimate} tokens; max {settings.MAX_FILE_TOKENS}).",
        )

    chunked = await asyncio.to_thread(chunking, text, metadata)
    child_chunks = [c for c in chunked if c.get("metadata", {}).get("chunk_type") != "parent"]
    if len(child_chunks) > settings.MAX_CHUNKS_PER_FILE:
        raise HTTPException(
            status_code=413,
            detail=f"Too many chunks ({len(child_chunks)}; max {settings.MAX_CHUNKS_PER_FILE}).",
        )

    await enforce_ingest_budget(db, current_user.user_id, token_estimate)

    file_hash = compute_file_hash(file_content)
    # Same-chat hash dedupe only (allowed across chats)
    existing = await db.execute(
        select(IngestionLog).where(
            IngestionLog.user_id == current_user.user_id,
            IngestionLog.chat_id == chat_id,
            IngestionLog.file_hash == file_hash,
            IngestionLog.status == "completed",
        )
    )
    existing_log = existing.scalar_one_or_none()
    if existing_log:
        logger.info(
            "Duplicate upload skipped (same chat): file=%s hash=%s chat=%s",
            file.filename,
            file_hash[:12],
            chat_id,
        )
        return existing_log

    log = IngestionLog(
        chat_id=chat_id,
        user_id=current_user.user_id,
        filename=file.filename,
        file_hash=file_hash,
        ingest_token_count=token_estimate,
        status="pending",
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    # Persist the original so citations can hyperlink back to the source.
    # Storage failure is logged but non-fatal: ingestion itself only needs
    # the extracted text, which we already have in memory.
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename).suffix.lower()[:16] or ".bin"
        storage_path = upload_dir / f"{log.id}{ext}"
        storage_path.write_bytes(file_content)
        log.storage_path = str(storage_path)
        log.content_type = file.content_type or "application/octet-stream"
        log.size_bytes = len(file_content)
        await db.commit()
        await db.refresh(log)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist original file (%s): %s", log.id, exc)

    background_tasks.add_task(
        _run_ingestion,
        ingestion_id=log.id,
        file_contents=file_content,
        filename=file.filename,
        uid=current_user.user_id,
        chat_id=chat_id,
        token_estimate=token_estimate,
        pre_extracted=(text, metadata),
    )

    logger.info(
        "Upload accepted: ingestion_id=%s file=%s chat_id=%s user_id=%s tokens=%s",
        log.id,
        file.filename,
        chat_id,
        current_user.user_id,
        token_estimate,
    )
    return log


@router.get("/{ingestion_id}/file")
async def get_document_file(
    ingestion_id: uuid.UUID,
    exp: int | None = None,
    sig: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Serve a stored original. Signed-URL auth (no bearer): <a href> links
    from the chat transcript can't carry headers. Signature is scoped to this
    ingestion id and expires (see signing.py)."""
    if not verify_file_sig(str(ingestion_id), exp, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired link")
    result = await db.execute(
        select(IngestionLog).where(
            IngestionLog.id == ingestion_id,
            IngestionLog.storage_path.isnot(None),
        )
    )
    log = result.scalar_one_or_none()
    if not log or not log.storage_path:
        raise HTTPException(status_code=404, detail="File not stored")
    path = Path(log.storage_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File missing from storage")
    return FileResponse(
        path,
        media_type=log.content_type or "application/octet-stream",
        filename=log.filename,
        content_disposition_type="inline",
    )


@router.get("/ingestions/{ingestion_id}", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    ingestion_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check the status of a document ingestion."""
    result = await db.execute(
        select(IngestionLog).where(
            IngestionLog.id == ingestion_id,
            IngestionLog.user_id == current_user.user_id,
        )
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="Ingestion not found")
    return log
