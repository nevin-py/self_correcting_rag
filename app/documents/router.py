
from fastapi import Depends
from app.auth.models import User
from app.auth.router import get_current_user
from fastapi import APIRouter,UploadFile,File,status,BackgroundTasks, HTTPException, status
from app.documents.service import full_pipeline
from app.documents import service
import uuid
router=APIRouter(prefix="/documents", tags=["Documents"])

@router.post('/upload_file', status_code=status.HTTP_202_ACCEPTED)
async def upload(chat_id:uuid.UUID,background_tasks:BackgroundTasks,current_user:User=Depends(get_current_user),file:UploadFile=File(...)):
    file_content=await file.read()
    await file.close()
    if not file_content:
        raise HTTPException(status_code=400, 
        detail='file not found')
    if not file.filename:
        raise HTTPException(
        status_code=400,
        detail="Filename is missing."
    )
    background_tasks.add_task(
                                full_pipeline,
                                file_contents=file_content,
                                filename=file.filename,
                                uid=current_user.user_id,
                                chat_id=chat_id,
                                )

    return {
        "message": "File accepted for processing.",
        "chat_id": str(chat_id),
    }