#agents/models.py
from sqlalchemy import ForeignKey,String,DateTime,func
from sqlalchemy.orm import Mapped,mapped_column,relationship
import uuid
from typing import Literal,TYPE_CHECKING
from datetime import datetime
from app.core.database import Base
if TYPE_CHECKING:
    from app.auth.models import User

class Agent_interact(Base):
    __tablename__='agents'
    id:Mapped[uuid.UUID]=mapped_column(primary_key=True,default=uuid.uuid4)
    chat_id:Mapped[uuid.UUID]=mapped_column(ForeignKey('chats.chat_id'))
    user_input:Mapped[str]=mapped_column(nullable=False)
    agent_output:Mapped[str]=mapped_column(nullable=False)
    routing_path:Mapped[str]=mapped_column(String(50))
    token_metric:Mapped[int]=mapped_column(nullable=False)
    latency:Mapped[float]=mapped_column(nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
    chat:Mapped["Chats"]=relationship(back_populates="interactions")
class Chats(Base):
    __tablename__='chats'
    user_id:Mapped[uuid.UUID]=mapped_column(ForeignKey('users.user_id'))
    chat_id:Mapped[uuid.UUID]=mapped_column(primary_key=True,default=uuid.uuid4)
    title:Mapped[str]=mapped_column(nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
    user:Mapped["User"]=relationship(back_populates="chats")
    interactions:Mapped[list["Agent_interact"]]=relationship(
        back_populates="chat",
        order_by=lambda: Agent_interact.created_at.asc(),
        cascade="all, delete-orphan"
    )