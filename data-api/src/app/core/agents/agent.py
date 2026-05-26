"""
Chatbot agent wrapper.

Provides a simple interface to interact with the LangGraph agent.
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage

from app.core.agents.state import ChatbotState
from app.core.agents.graph_builder import build_chatbot_graph
from app.core.agents.nodes.guardrail_node import GuardrailNode
from app.core.agents.nodes.rag_node import RAGNode
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.clients.data_api_client import DataApiClient


logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Wrapper for the chatbot LangGraph agent."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        system_prompt: str = None,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

        self.llm_client = LiteLLMClient()
        self.data_api_client = DataApiClient()

        # Keep raw node instances around for the streaming path. The compiled
        # graph wraps them in PregelNode objects that aren't directly
        # callable, so we can't reach back through ``graph.nodes`` to invoke
        # them token-by-token.
        self._guardrail_node = GuardrailNode()
        self._rag_node = RAGNode(
            llm_client=self.llm_client,
            data_api_client=self.data_api_client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt or RAGNode.__dict__.get("DEFAULT_SYSTEM_PROMPT", ""),
        )

        self.graph = build_chatbot_graph(
            llm_client=self.llm_client,
            data_api_client=self.data_api_client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            guardrail_node=self._guardrail_node,
            rag_node=self._rag_node,
        )

    async def chat(
        self,
        message: str,
        kb_ids: list,
        conversation_id: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Send a message and get a response."""
        messages = history or []
        messages.append(HumanMessage(content=message))

        initial_state: ChatbotState = {
            "messages": messages,
            "agent_id": "",
            "user_id": "",
            "kb_ids": kb_ids,
            "guardrail_passed": False,
            "context": None,
        }

        config = {}
        if conversation_id:
            config = {"configurable": {"thread_id": conversation_id}}

        result = await self.graph.ainvoke(initial_state, config=config)

        ai_message = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break

        return {
            "response": ai_message,
            "context": result.get("context"),
        }

    async def stream(
        self,
        message: str,
        kb_ids: list,
        conversation_id: Optional[str] = None,
        history: Optional[list] = None,
    ) -> AsyncIterator[str]:
        """Stream a response.

        Bypasses the compiled graph for token streaming — runs guardrail
        then RAG directly so we can yield deltas as they arrive from the
        LLM (LangGraph's streaming API doesn't surface LLM token streams
        through its node interface).
        """
        messages = history or []
        messages.append(HumanMessage(content=message))

        initial_state: ChatbotState = {
            "messages": messages,
            "agent_id": "",
            "user_id": "",
            "kb_ids": kb_ids,
            "guardrail_passed": False,
            "context": None,
        }

        # Guardrail mutates state in-place via its return dict; merge so the
        # RAG node sees the same flags it would inside the graph.
        guardrail_update = await self._guardrail_node(initial_state)
        if isinstance(guardrail_update, dict):
            initial_state.update(guardrail_update)

        async for chunk in self._rag_node.stream(initial_state):
            yield chunk
