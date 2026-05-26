"""
RAG node for retrieving context and generating responses.
"""

import logging
from typing import Dict, Any, List, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.core.agents.state import ChatbotState
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.clients.data_api_client import DataApiClient
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
        """Retrieve context from knowledge bases and generate response."""
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            return {"messages": [AIMessage(content="I didn't receive a question. How can I help you?")]}

        context = await self._retrieve_context(kb_ids, user_query)
        llm_messages = self._build_llm_messages(messages, context)

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
        """Stream response from LLM."""
        messages = state.get("messages", [])
        kb_ids = state.get("kb_ids", [])

        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            yield "I didn't receive a question. How can I help you?"
            return

        context = await self._retrieve_context(kb_ids, user_query)
        llm_messages = self._build_llm_messages(messages, context)

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

    # Number of child chunks to retrieve per knowledge base. After global
    # dedupe by parent, the actual number of parent blocks fed to the LLM
    # will be <= TOP_K_CHILD (parents sharing children collapse to one).
    TOP_K_CHILD = 10

    async def _retrieve_context(self, kb_ids: List[str], query: str) -> str:
        """Retrieve context using parent-child dedupe.

        Pipeline (matches v4-style retrieval):
          1. Fetch top-K child chunks per KB (vector similarity).
          2. Flatten across KBs and sort by similarity desc.
          3. Dedupe by ``parent_text`` — keep the first (highest-scored)
             occurrence; collapse all sibling children of the same parent.
          4. Concatenate the surviving parent blocks into the LLM context.

        For graph-RAG knowledge bases the response shape has no chunk list;
        we fall back to the pre-built ``context`` string.
        """
        if not kb_ids:
            return ""

        per_kb_results = await self.data_api_client.batch_query_knowledge_bases(
            kb_ids=kb_ids,
            query_text=query,
            top_k=self.TOP_K_CHILD,
        )

        all_chunks: List[Dict[str, Any]] = []
        graph_contexts: List[str] = []

        for kb_result in per_kb_results:
            if not kb_result.get("success"):
                continue
            data = kb_result.get("data", {}) or {}
            chunks = data.get("results") or data.get("chunks")
            if isinstance(chunks, list):
                for c in chunks:
                    if isinstance(c, dict):
                        all_chunks.append(c)
            elif isinstance(data.get("context"), str) and data["context"].strip():
                graph_contexts.append(data["context"])

        # Children come back per-KB already sorted; merging multiple KBs needs
        # a global re-sort so the dedupe always keeps the highest-scored
        # representative of each parent.
        all_chunks.sort(key=lambda c: float(c.get("similarity") or 0.0), reverse=True)

        seen_parents: set = set()
        parent_blocks: List[str] = []
        for chunk in all_chunks:
            parent_text = chunk.get("parent_text") or chunk.get("content") or chunk.get("text", "")
            if not parent_text:
                continue
            # parent_text can be long — hash to keep the set small.
            key = hash(parent_text)
            if key in seen_parents:
                continue
            seen_parents.add(key)
            parent_blocks.append(parent_text)

        context_parts = parent_blocks + graph_contexts
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

        if context:
            system_content = f"{self.system_prompt}\n\n--- Retrieved Context ---\n{context}"
        else:
            system_content = self.system_prompt

        llm_messages.append({"role": "system", "content": system_content})

        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                pass

        return llm_messages
