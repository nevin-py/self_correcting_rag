import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, status, BackgroundTasks, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.auth.models import User
from app.auth.router import get_current_user
from app.agent.models import Chats
from app.documents.service import full_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


async def _run_ingestion(file_contents: bytes, filename: str, uid: uuid.UUID, chat_id: uuid.UUID):
    """Wrapper to run the blocking ingestion pipeline in a thread pool."""
    try:
        await asyncio.to_thread(
            full_pipeline,
            file_contents=file_contents,
            filename=filename,
            uid=uid,
            chat_id=chat_id,
        )
    except Exception:
        logger.exception("Ingestion failed for file=%s user=%s chat=%s", filename, uid, chat_id)


@router.post("/upload_file", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    chat_id: uuid.UUID = Query(..., description="Chat ID to associate the document with"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
):
    """Upload a document for ingestion into the vector store."""
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

    # ── Queue background ingestion ───────────────────────────────────────
    background_tasks.add_task(
        _run_ingestion,
        file_contents=file_content,
        filename=file.filename,
        uid=current_user.user_id,
        chat_id=chat_id,
    )

    return {
        "message": "File accepted for processing.",
        "chat_id": str(chat_id),
        "filename": file.filename,
    }
