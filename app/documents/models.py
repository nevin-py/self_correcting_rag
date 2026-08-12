import uuid
from datetime import datetime
from sqlalchemy import ForeignKey, String, DateTime, Text, Integer, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chats.chat_id"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.user_id"))
    filename: Mapped[str] = mapped_column(nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ingest_token_count: Mapped[int | None] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | processing | completed | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
