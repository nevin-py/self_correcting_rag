import uuid
from datetime import datetime
from typing import Literal
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
    provider: Literal["groq", "openrouter", "auto"] = Field(
        default="auto",
        description=(
            "LLM provider to use. "
            "'groq' = Groq only (fast, free tier). "
            "'openrouter' = OpenRouter only (paid, higher quality). "
            "'auto' = Try Groq first, fall back to OpenRouter on rate limit."
        ),
    )


class CitationResponse(BaseModel):
    evidence_id: str
    text: str
    source_type: str
    source_name: str
    source_url: str | None = None
    source_date: datetime | None = None
    authority_score: float
    recency_score: float
    # New structured metadata fields
    metric_type: str = "unknown"
    metric_value: str = ""
    geographic_scope: str = "unknown"
    geography: str = ""
    year_period: str = ""
    temporal_qualifier: str = "unknown"
    source_quality: str = "unknown"


class ClaimResponse(BaseModel):
    claim_id: str
    text: str
    status: str
    claim_type: str = "fact"          # fact / inference / speculation
    evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    reasoning: str


class QueryResponse(BaseModel):
    answer: str
    chat_id: uuid.UUID
    latency_ms: float
    provider_used: str | None = None
    final_status: str | None = None
    claims: list[ClaimResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    verification_errors: list[dict] = Field(default_factory=list)


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


class MessageResponse(BaseModel):
    role: str
    content: str
    sequence: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
