"""LangGraph chatbot graph builder.

(This is the LangGraph state graph for the chatbot agent — unrelated to the
removed GraphRAG knowledge-graph code.)

Graph: START -> Guardrail -> RAG -> END
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.core.agents.state import ChatbotState
from app.core.agents.nodes.guardrail_node import GuardrailNode
from app.core.agents.nodes.rag_node import RAGNode
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.clients.data_api_client import DataApiClient


logger = logging.getLogger(__name__)


def build_chatbot_graph(
    llm_client: LiteLLMClient,
    data_api_client: DataApiClient,
    model: str = "gemini/gemini-2.0-flash",
    temperature: float = 0.7,
    system_prompt: str = None,
    checkpointer: Optional[MemorySaver] = None,
    guardrail_node: Optional[GuardrailNode] = None,
    rag_node: Optional[RAGNode] = None,
) -> CompiledStateGraph:
    """Build the chatbot agent graph.

    Args:
        llm_client: LiteLLM client for LLM calls.
        data_api_client: Data API client for KB queries.
        model: LLM model name.
        temperature: LLM temperature.
        system_prompt: Custom system prompt.
        checkpointer: Optional state persistence.
        guardrail_node: Pre-instantiated guardrail node. Pass in when the
            caller needs a direct (non-graph) reference (e.g. for token
            streaming that bypasses the compiled graph).
        rag_node: Same as above for the RAG node.
    """
    if guardrail_node is None:
        guardrail_node = GuardrailNode()
    if rag_node is None:
        rag_node = RAGNode(
            llm_client=llm_client,
            data_api_client=data_api_client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt or RAGNode.__dict__.get("DEFAULT_SYSTEM_PROMPT", ""),
        )

    builder = StateGraph(ChatbotState)
    builder.add_node(guardrail_node.name, guardrail_node)
    builder.add_node(rag_node.name, rag_node)
    builder.add_edge(START, guardrail_node.name)
    builder.add_edge(guardrail_node.name, rag_node.name)
    builder.add_edge(rag_node.name, END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    logger.info(f"Chatbot graph built with model: {model}")
    return graph
