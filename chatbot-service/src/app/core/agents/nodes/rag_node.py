"""
RAG node for retrieving context and generating responses.

This node queries linked knowledge bases and generates a response using the LLM.
"""

import logging
from typing import Dict, Any, List, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.agents.state import ChatbotState
from app.infrastructure.clients import LiteLLMClient, DataApiClient
from app.exceptions import ExternalServiceError


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely.

If the context doesn't contain relevant information, say so and provide a general response based on your knowledge.

Always be helpful, accurate, and professional."""


class RAGNode:
    """Node for RAG-based response generation."""

    def __init__(
        self,
        llm_client: LiteLLMClient,
        data_api_client: DataApiClient,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.name = "rag"
        self.llm_client = llm_client
        self.data_api_client = data_api_client
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

    async def __call__(self, state: ChatbotState) -> Dict[str, Any]:
        """
        Retrieve context from knowledge bases and generate response.

        Args:
            state: Current chatbot state

        Returns:
            Updated state with AI response message
        """
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        # Get the latest user query
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            return {"messages": [AIMessage(content="I didn't receive a question. How can I help you?")]}

        # Retrieve context from linked knowledge bases
        context = await self._retrieve_context(kb_ids, user_query)

        # Build messages for LLM
        llm_messages = self._build_llm_messages(messages, context)

        # Generate response
        try:
            response = await self.llm_client.chat(
                messages=llm_messages,
                model=self.model,
                temperature=self.temperature,
            )
            return {
                "messages": [AIMessage(content=response)],
                "context": context,
            }
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise ExternalServiceError("LLM", str(e))

    async def stream(self, state: ChatbotState) -> AsyncIterator[str]:
        """
        Stream response from LLM.

        Args:
            state: Current chatbot state

        Yields:
            Chunks of the response
        """
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        # Get the latest user query
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            yield "I didn't receive a question. How can I help you?"
            return

        # Retrieve context
        context = await self._retrieve_context(kb_ids, user_query)

        # Build messages for LLM
        llm_messages = self._build_llm_messages(messages, context)

        # Stream response
        try:
            async for chunk in self.llm_client.stream_chat(
                messages=llm_messages,
                model=self.model,
                temperature=self.temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise ExternalServiceError("LLM", str(e))

    async def _retrieve_context(self, kb_ids: List[str], query: str) -> str:
        """Retrieve context from linked knowledge bases."""
        if not kb_ids:
            return ""

        context_parts = []
        results = await self.data_api_client.batch_query_knowledge_bases(
            kb_ids=kb_ids,
            query_text=query,
            top_k=5,
        )

        for result in results:
            if result.get("success"):
                data = result.get("data", {})
                # Handle different response formats from data-api
                if "chunks" in data:
                    for chunk in data["chunks"][:3]:
                        if isinstance(chunk, dict):
                            text = chunk.get("content") or chunk.get("text", "")
                            if text:
                                context_parts.append(text)
                elif "context" in data:
                    context_parts.append(data["context"])
                elif "results" in data:
                    for item in data["results"][:3]:
                        if isinstance(item, dict):
                            text = item.get("content") or item.get("text", "")
                            if text:
                                context_parts.append(text)

        if context_parts:
            return "\n\n---\n\n".join(context_parts)
        return ""

    def _build_llm_messages(
        self,
        messages: List,
        context: str,
    ) -> List[Dict[str, str]]:
        """Build messages list for LLM API call."""
        llm_messages = []

        # Add system prompt with context
        if context:
            system_content = f"{self.system_prompt}\n\n--- Retrieved Context ---\n{context}"
        else:
            system_content = self.system_prompt

        llm_messages.append({"role": "system", "content": system_content})

        # Add conversation history (limit to last 10 messages)
        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                pass  # Skip system messages from history

        return llm_messages
