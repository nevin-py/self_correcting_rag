import asyncio
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, status, BackgroundTasks, HTTPException, Query

from app.auth.models import User
from app.auth.router import get_current_user
from app.documents.service import full_pipeline

router = APIRouter(prefix="/documents", tags=["Documents"])


async def _run_ingestion(file_contents: bytes, filename: str, uid: uuid.UUID, chat_id: uuid.UUID):
    """Wrapper to run the blocking ingestion pipeline in a thread pool."""
    await asyncio.to_thread(
        full_pipeline,
        file_contents=file_contents,
        filename=filename,
        uid=uid,
        chat_id=chat_id,
    )


@router.post("/upload_file", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    chat_id: uuid.UUID = Query(..., description="Chat ID to associate the document with"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
):
    """Upload a document for ingestion into the vector store."""
    file_content = await file.read()
    await file.close()

    if not file_content:
        raise HTTPException(status_code=400, detail="File is empty")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    # Validate file size (50MB max)
    max_size = 50 * 1024 * 1024
    if len(file_content) > max_size:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

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
