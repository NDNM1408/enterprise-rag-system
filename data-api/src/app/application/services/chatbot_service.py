"""Service for chatbot operations."""

import logging
from typing import List, Optional, Dict, Any, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.core.agents import ChatbotAgent
from app.infrastructure.connectors.postgres.repositories.agent_repository import AgentRepository
from app.infrastructure.connectors.postgres.repositories.conversation_repository import ConversationRepository
from app.exceptions import ResourceNotFoundError


logger = logging.getLogger(__name__)


class ChatbotService:
    """Service for chatbot operations."""

    def __init__(self):
        self.agent_repository = AgentRepository()
        self.conversation_repository = ConversationRepository()
        self._agent_cache: Dict[str, ChatbotAgent] = {}

    def _get_or_create_agent(
        self,
        model: str,
        temperature: float,
        system_prompt: str,
    ) -> ChatbotAgent:
        """Get or create a ChatbotAgent instance."""
        cache_key = f"{model}:{temperature}:{hash(system_prompt)}"
        if cache_key not in self._agent_cache:
            self._agent_cache[cache_key] = ChatbotAgent(
                model=model,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        return self._agent_cache[cache_key]

    async def chat(
        self,
        agent_id: str,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message to an agent and get a response."""
        agent_config = await self._get_agent_config(agent_id)

        conversation = await self.conversation_repository.get_or_create_conversation(
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        history = await self._get_conversation_history(conversation.id)
        kb_ids = await self.agent_repository.get_linked_kb_ids(agent_id)

        chatbot_agent = self._get_or_create_agent(
            model=agent_config["llm_model"],
            temperature=agent_config["llm_temperature"],
            system_prompt=agent_config["system_prompt"],
        )

        result = await chatbot_agent.chat(
            message=message,
            kb_ids=kb_ids,
            conversation_id=conversation.id,
            history=history,
        )

        await self.conversation_repository.add_message(
            conversation_id=conversation.id,
            role="human",
            content=message,
        )
        await self.conversation_repository.add_message(
            conversation_id=conversation.id,
            role="ai",
            content=result["response"],
        )

        return {
            "response": result["response"],
            "conversation_id": conversation.id,
            "context": result.get("context"),
        }

    async def stream_chat(
        self,
        agent_id: str,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a response from an agent.

        Yields structured events the controller serialises as JSON SSE frames:

          - ``{"type": "meta", "conversation_id": "..."}``  emitted first
          - ``{"type": "delta", "content": "..."}``        for every token
          - (``[DONE]`` sentinel is appended by the controller)
        """
        agent_config = await self._get_agent_config(agent_id)

        conversation = await self.conversation_repository.get_or_create_conversation(
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # Emit the conversation id immediately so the client can persist it
        # for follow-up turns even before the first token arrives.
        yield {"type": "meta", "conversation_id": conversation.id}

        history = await self._get_conversation_history(conversation.id)
        kb_ids = await self.agent_repository.get_linked_kb_ids(agent_id)

        chatbot_agent = self._get_or_create_agent(
            model=agent_config["llm_model"],
            temperature=agent_config["llm_temperature"],
            system_prompt=agent_config["system_prompt"],
        )

        await self.conversation_repository.add_message(
            conversation_id=conversation.id,
            role="human",
            content=message,
        )

        full_response = []
        full_thinking = []

        async for event in chatbot_agent.stream(
            message=message,
            kb_ids=kb_ids,
            conversation_id=conversation.id,
            history=history,
        ):
            # Agent emits tagged dicts:
            #   {"type": "content",  "delta": "..."}  → answer tokens
            #   {"type": "thinking", "delta": "..."}  → reasoning tokens
            #   {"type": "agentic", "phase": "...", ...}  → planner-hop progress
            # Plain strings are tolerated as legacy "content" fallback.
            if isinstance(event, dict):
                ev_type = event.get("type", "content")
            else:
                ev_type = "content"
                event = {"type": "content", "delta": str(event)}

            # Agentic progress: forward the whole payload to the client. No
            # text accumulation — these are UI-only progress signals.
            if ev_type == "agentic":
                yield event
                continue

            delta = event.get("delta") or event.get("content") or ""
            if not delta:
                continue

            if ev_type == "thinking":
                full_thinking.append(delta)
                yield {"type": "thinking", "delta": delta}
            else:
                full_response.append(delta)
                # Keep the legacy ``{type:"delta",content:...}`` shape for the
                # content stream so the frontend hook doesn't need a flag day.
                yield {"type": "delta", "content": delta}

        await self.conversation_repository.add_message(
            conversation_id=conversation.id,
            role="ai",
            content="".join(full_response),
        )

    async def _get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get agent configuration."""
        agent = await self.agent_repository.get(id=agent_id)
        return {
            "llm_model": agent.llm_model,
            "llm_temperature": float(agent.llm_temperature) if agent.llm_temperature else 0.7,
            "system_prompt": agent.system_prompt,
            "is_active": agent.is_active,
        }

    async def _get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> List[BaseMessage]:
        """Get conversation history as LangChain messages."""
        messages = await self.conversation_repository.get_messages(
            conversation_id=conversation_id,
            limit=limit,
        )

        history = []
        for msg in messages:
            if msg.role == "human":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                history.append(AIMessage(content=msg.content))

        return history

    async def get_conversations(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get conversations for a user."""
        conversations = await self.conversation_repository.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            skip=skip,
            limit=limit,
        )
        return [c.to_dict() for c in conversations]

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        conversation = await self.conversation_repository.get(id=conversation_id)
        if conversation.user_id != user_id:
            raise ResourceNotFoundError("Conversation", conversation_id)

        messages = await self.conversation_repository.get_messages(conversation_id)
        return [m.to_dict() for m in messages]


def get_chatbot_service() -> ChatbotService:
    """Dependency injection for ChatbotService."""
    return ChatbotService()
