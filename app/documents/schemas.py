import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class IngestionLogResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    filename: str
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class IngestionStatusResponse(BaseModel):
    ingestion_id: uuid.UUID
    status: str
    error_message: str | None = None
    filename: str
    chat_id: uuid.UUID
