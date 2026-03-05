"""
Chatbot agent wrapper.

Provides a simple interface to interact with the LangGraph agent.
"""

import logging
from typing import Dict, Any, Optional, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage

from app.core.agents.state import ChatbotState
from app.core.agents.graph_builder import build_chatbot_graph, get_rag_node_from_graph
from app.infrastructure.clients import LiteLLMClient, DataApiClient


logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Wrapper for the chatbot LangGraph agent."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        system_prompt: str = None,
    ):
        """
        Initialize the chatbot agent.

        Args:
            model: LLM model name
            temperature: LLM temperature
            system_prompt: Custom system prompt
        """
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

        # Initialize clients
        self.llm_client = LiteLLMClient()
        self.data_api_client = DataApiClient()

        # Build the graph
        self.graph = build_chatbot_graph(
            llm_client=self.llm_client,
            data_api_client=self.data_api_client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    async def chat(
        self,
        message: str,
        kb_ids: list[str],
        conversation_id: Optional[str] = None,
        history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Send a message and get a response.

        Args:
            message: User message
            kb_ids: List of knowledge base IDs to query
            conversation_id: Thread/conversation ID for state persistence
            history: Optional chat history as list of messages

        Returns:
            Response dict with 'response' and 'context' keys
        """
        # Build initial messages
        messages = history or []
        messages.append(HumanMessage(content=message))

        # Build initial state
        initial_state: ChatbotState = {
            "messages": messages,
            "agent_id": "",  # Not used in graph
            "user_id": "",  # Not used in graph
            "kb_ids": kb_ids,
            "guardrail_passed": False,
            "context": None,
        }

        # Configuration for thread persistence
        config = {}
        if conversation_id:
            config = {"configurable": {"thread_id": conversation_id}}

        # Invoke the graph
        result = await self.graph.ainvoke(initial_state, config=config)

        # Extract the AI response
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
        kb_ids: list[str],
        conversation_id: Optional[str] = None,
        history: Optional[list] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response.

        Args:
            message: User message
            kb_ids: List of knowledge base IDs to query
            conversation_id: Thread/conversation ID for state persistence
            history: Optional chat history as list of messages

        Yields:
            Chunks of the response
        """
        # Build initial messages
        messages = history or []
        messages.append(HumanMessage(content=message))

        # Build initial state
        initial_state: ChatbotState = {
            "messages": messages,
            "agent_id": "",
            "user_id": "",
            "kb_ids": kb_ids,
            "guardrail_passed": False,
            "context": None,
        }

        # Get the RAG node for streaming
        rag_node = get_rag_node_from_graph(self.graph)
        if rag_node is None:
            # Fallback to non-streaming
            result = await self.chat(message, kb_ids, conversation_id, history)
            yield result["response"]
            return

        # First run guardrail
        guardrail_node = self.graph.nodes.get("guardrail")
        if guardrail_node:
            await guardrail_node(initial_state)

        # Stream from RAG node
        async for chunk in rag_node.stream(initial_state):
            yield chunk
