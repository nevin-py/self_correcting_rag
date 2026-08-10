import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ChatCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    chat_id: uuid.UUID
    title: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ChatListResponse(BaseModel):
    chats: list[ChatResponse]


class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class QueryResponse(BaseModel):
    answer: str
    chat_id: uuid.UUID
    latency_ms: float


class InteractionResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    user_input: str
    agent_output: str
    routing_path: str | None = None
    token_metric: int | None = None
    latency: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class InteractionListResponse(BaseModel):
    interactions: list[InteractionResponse]
