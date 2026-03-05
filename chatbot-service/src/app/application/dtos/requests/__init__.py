"""Request DTOs module."""

from .agent_requests import (
    CreateAgentRequest,
    UpdateAgentRequest,
    LinkKnowledgeBaseRequest,
    ChatRequest,
)

__all__ = [
    "CreateAgentRequest",
    "UpdateAgentRequest",
    "LinkKnowledgeBaseRequest",
    "ChatRequest",
]
