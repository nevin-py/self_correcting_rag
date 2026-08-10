#auth/models.py
from typing import TYPE_CHECKING
from sqlalchemy import DateTime,func
import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped,mapped_column,relationship
from app.core.database import Base
if TYPE_CHECKING:
    from app.agent.models import Chats
class User(Base):
    __tablename__='users'
    user_id:Mapped[uuid.UUID]=mapped_column(primary_key=True,default=uuid.uuid4)
    email:Mapped[str]=mapped_column(unique=True,nullable=False,index=True)
    hashed_password:Mapped[str]=mapped_column(nullable=False)
    is_active:Mapped[bool]=mapped_column(default=True)
    create_time:Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
    update_time:Mapped[datetime]=mapped_column(DateTime,server_default=func.now(),onupdate=func.now())
    chats:Mapped[list["Chats"]]=relationship(back_populates='user')

