"""
Chatbot API controller.

Provides endpoints for agent management and chat operations.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Path, Request as FastAPIRequest, Depends, Query
from fastapi.responses import StreamingResponse

from app.application.dtos.requests.agent_requests import (
    CreateAgentRequest,
    UpdateAgentRequest,
    LinkKnowledgeBaseRequest,
    ChatRequest,
)
from app.application.dtos.responses.success_response import create_success_response
from app.application.services.agent_service import AgentService, get_agent_service, serialize_agent
from app.application.services.chatbot_service import ChatbotService, get_chatbot_service


_path = "/api/v1"
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=_path,
    tags=["chatbot"]
)


# ============ Agent Management Endpoints ============

@router.post(
    "/agents",
    summary="Create agent",
    description="Create a new chatbot agent with LLM configuration",
)
async def create_agent(
    request: FastAPIRequest,
    body: CreateAgentRequest,
    service: AgentService = Depends(get_agent_service),
):
    """Create a new agent."""
    agent = await service.create_agent(
        name=body.name,
        llm_model=body.llm_model,
        description=body.description,
        system_prompt=body.system_prompt,
        llm_temperature=body.llm_temperature,
        tenant_id=body.tenant_id,
        created_by=body.created_by,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=serialize_agent(agent), request_id=request_id)


@router.get(
    "/agents",
    summary="List agents",
    description="List all agents with optional filters",
)
async def list_agents(
    request: FastAPIRequest,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
    tenant_id: Optional[str] = Query(default=None),
    is_active: Optional[bool] = Query(default=None),
    service: AgentService = Depends(get_agent_service),
):
    """List agents."""
    agents = await service.list_agents(
        skip=skip,
        limit=limit,
        tenant_id=tenant_id,
        is_active=is_active,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data=[serialize_agent(a) for a in agents],
        request_id=request_id,
    )


@router.get(
    "/agents/{agent_id}",
    summary="Get agent",
    description="Get agent details by ID",
)
async def get_agent(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    service: AgentService = Depends(get_agent_service),
):
    """Get agent by ID."""
    agent = await service.get_agent_with_kb_ids(agent_id)
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=agent, request_id=request_id)


@router.put(
    "/agents/{agent_id}",
    summary="Update agent",
    description="Update agent configuration",
)
async def update_agent(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    body: UpdateAgentRequest = ...,
    service: AgentService = Depends(get_agent_service),
):
    """Update an agent."""
    agent = await service.update_agent(
        agent_id=agent_id,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        llm_model=body.llm_model,
        llm_temperature=body.llm_temperature,
        is_active=body.is_active,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=serialize_agent(agent), request_id=request_id)


@router.delete(
    "/agents/{agent_id}",
    summary="Delete agent",
    description="Delete an agent by ID",
)
async def delete_agent(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    service: AgentService = Depends(get_agent_service),
):
    """Delete an agent."""
    await service.delete_agent(agent_id)
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data={"deleted": True}, request_id=request_id)


# ============ Knowledge Base Linking Endpoints ============

@router.post(
    "/agents/{agent_id}/kb",
    summary="Link knowledge base",
    description="Link a knowledge base to an agent",
)
async def link_knowledge_base(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    body: LinkKnowledgeBaseRequest = ...,
    service: AgentService = Depends(get_agent_service),
):
    """Link a knowledge base to an agent."""
    result = await service.link_knowledge_base(
        agent_id=agent_id,
        kb_id=body.kb_id,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=result, request_id=request_id)


@router.delete(
    "/agents/{agent_id}/kb/{kb_id}",
    summary="Unlink knowledge base",
    description="Unlink a knowledge base from an agent",
)
async def unlink_knowledge_base(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    kb_id: str = Path(..., description="Knowledge base ID"),
    service: AgentService = Depends(get_agent_service),
):
    """Unlink a knowledge base from an agent."""
    await service.unlink_knowledge_base(
        agent_id=agent_id,
        kb_id=kb_id,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data={"unlinked": True}, request_id=request_id)


# ============ Chat Endpoints ============

@router.post(
    "/agents/{agent_id}/chat",
    summary="Chat with agent",
    description="Send a message to an agent and get a response",
)
async def chat(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    body: ChatRequest = ...,
    service: ChatbotService = Depends(get_chatbot_service),
):
    """Send a message to an agent."""
    result = await service.chat(
        agent_id=agent_id,
        message=body.message,
        user_id=body.user_id,
        conversation_id=body.conversation_id,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=result, request_id=request_id)


@router.post(
    "/agents/{agent_id}/chat/stream",
    summary="Stream chat with agent",
    description="Send a message and stream the response",
)
async def stream_chat(
    request: FastAPIRequest,
    agent_id: str = Path(..., description="Agent ID"),
    body: ChatRequest = ...,
    service: ChatbotService = Depends(get_chatbot_service),
):
    """Stream a response from an agent."""

    async def generate():
        # Service yields structured events; serialise as JSON SSE frames so
        # newlines inside model output never confuse the SSE parser on the
        # client. The final ``[DONE]`` sentinel stays a plain string for
        # backwards compatibility with the EventSource convention.
        async for event in service.stream_chat(
            agent_id=agent_id,
            message=body.message,
            user_id=body.user_id,
            conversation_id=body.conversation_id,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============ Conversation Endpoints ============

@router.get(
    "/conversations",
    summary="List conversations",
    description="List conversations for a user",
)
async def list_conversations(
    request: FastAPIRequest,
    user_id: str = Query(..., description="User ID"),
    agent_id: Optional[str] = Query(default=None, description="Filter by agent"),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    service: ChatbotService = Depends(get_chatbot_service),
):
    """List conversations for a user."""
    conversations = await service.get_conversations(
        user_id=user_id,
        agent_id=agent_id,
        skip=skip,
        limit=limit,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=conversations, request_id=request_id)


@router.get(
    "/conversations/{conversation_id}/messages",
    summary="Get conversation history",
    description="Get messages for a conversation",
)
async def get_conversation_history(
    request: FastAPIRequest,
    conversation_id: str = Path(..., description="Conversation ID"),
    user_id: str = Query(..., description="User ID"),
    service: ChatbotService = Depends(get_chatbot_service),
):
    """Get conversation history."""
    messages = await service.get_conversation_history(
        conversation_id=conversation_id,
        user_id=user_id,
    )
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=messages, request_id=request_id)
