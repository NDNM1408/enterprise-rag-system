"""Request DTOs for agent operations."""

from typing import Optional
from pydantic import BaseModel, Field


class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""

    name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    llm_model: str = Field(..., min_length=1, description="LLM model to use")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    created_by: Optional[str] = Field(None, description="Creator user ID")


class UpdateAgentRequest(BaseModel):
    """Request to update an agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    llm_model: Optional[str] = Field(None, min_length=1, description="LLM model to use")
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    is_active: Optional[bool] = Field(None, description="Whether the agent is active")


class LinkKnowledgeBaseRequest(BaseModel):
    """Request to link a knowledge base to an agent."""

    kb_id: str = Field(..., min_length=1, description="Knowledge base ID to link")


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, description="User message")
    user_id: str = Field(..., min_length=1, description="User ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn chat")
