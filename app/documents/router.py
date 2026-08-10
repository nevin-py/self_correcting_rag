import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, status, BackgroundTasks, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db, AsyncLocalSession
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats
from app.documents.models import IngestionLog
from app.documents.service import full_pipeline
from app.documents.schemas import IngestionLogResponse, IngestionStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Background worker ────────────────────────────────────────────────────────

async def _run_ingestion(ingestion_id: uuid.UUID, file_contents: bytes, filename: str, uid: uuid.UUID, chat_id: uuid.UUID):
    """Run the blocking ingestion pipeline in a thread pool, with status tracking."""
    # Mark as processing
    async with AsyncLocalSession() as session:
        log = await session.get(IngestionLog, ingestion_id)
        if log:
            log.status = "processing"
            await session.commit()

    try:
        await asyncio.to_thread(
            full_pipeline,
            file_contents=file_contents,
            filename=filename,
            uid=uid,
            chat_id=chat_id,
        )
        # Mark as completed
        async with AsyncLocalSession() as session:
            log = await session.get(IngestionLog, ingestion_id)
            if log:
                log.status = "completed"
                await session.commit()
    except Exception as exc:
        logger.exception("Ingestion failed for file=%s user=%s chat=%s", filename, uid, chat_id)
        # Mark as failed with error detail
        async with AsyncLocalSession() as session:
            log = await session.get(IngestionLog, ingestion_id)
            if log:
                log.status = "failed"
                log.error_message = str(exc)[:1000]
                await session.commit()


# ── Upload endpoint ──────────────────────────────────────────────────────────

@router.post("/upload_file", response_model=IngestionLogResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload(
    chat_id: uuid.UUID = Query(..., description="Chat ID to associate the document with"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
):
    """Upload a document for ingestion. Returns an ingestion_id for status tracking."""
    # ── Verify chat ownership ────────────────────────────────────────────
    result = await db.execute(
        select(Chats).where(
            Chats.chat_id == chat_id,
            Chats.user_id == current_user.user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Chat not found")

    # ── Validate file ────────────────────────────────────────────────────
    file_content = await file.read()
    await file.close()

    if not file_content:
        raise HTTPException(status_code=400, detail="File is empty")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    max_size = 50 * 1024 * 1024  # 50MB
    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # ── Create ingestion log (pending) ───────────────────────────────────
    log = IngestionLog(
        chat_id=chat_id,
        user_id=current_user.user_id,
        filename=file.filename,
        status="pending",
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    # ── Queue background ingestion ───────────────────────────────────────
    background_tasks.add_task(
        _run_ingestion,
        ingestion_id=log.id,
        file_contents=file_content,
        filename=file.filename,
        uid=current_user.user_id,
        chat_id=chat_id,
    )

    return log


# ── Status endpoint ──────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=404, detail="Ingestion log not found")

    return IngestionStatusResponse(
        ingestion_id=log.id,
        status=log.status,
        error_message=log.error_message,
        filename=log.filename,
        chat_id=log.chat_id,
    )
