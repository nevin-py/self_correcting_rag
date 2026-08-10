#app/agent/schemas
import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import uuid
class ChatCreate(BaseModel):
    title:str=Field(min_length=1,max_length=100)

class ChatResponse(BaseModel):
    chat_id:uuid.UUID
    title:str
    created_at:datetime
    user_id:uuid.UUID
    model_config=ConfigDict(from_attributes=True)
class InteractRequest(BaseModel):
    chat_id:uuid.UUID
    message:str=Field(min_length=1)
class InteractResponse(BaseModel):
    id:uuid.UUID
    chat_id:uuid.UUID
    user_input:str
    agent_output:str
    routing_path:str
    token_metric:int
    latency:float
    created_at:datetime
    model_config=ConfigDict(from_attributes=True)

