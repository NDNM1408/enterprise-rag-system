"""
LangGraph graph builder for the chatbot agent.

Simple graph: START → Guardrail → RAG → END
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.core.agents.state import ChatbotState
from app.core.agents.nodes.guardrail_node import GuardrailNode
from app.core.agents.nodes.rag_node import RAGNode
from app.infrastructure.clients import LiteLLMClient, DataApiClient


logger = logging.getLogger(__name__)


def build_chatbot_graph(
    llm_client: LiteLLMClient,
    data_api_client: DataApiClient,
    model: str = "gemini/gemini-2.0-flash",
    temperature: float = 0.7,
    system_prompt: str = None,
    checkpointer: Optional[MemorySaver] = None,
) -> CompiledStateGraph:
    """
    Build the chatbot agent graph.

    Args:
        llm_client: LiteLLM client for LLM calls
        data_api_client: Data API client for KB queries
        model: LLM model name
        temperature: LLM temperature
        system_prompt: Custom system prompt
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled state graph
    """
    # Initialize nodes
    guardrail_node = GuardrailNode()
    rag_node = RAGNode(
        llm_client=llm_client,
        data_api_client=data_api_client,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt or RAGNode.__dict__.get("DEFAULT_SYSTEM_PROMPT", ""),
    )

    # Build graph
    builder = StateGraph(ChatbotState)

    # Add nodes
    builder.add_node(guardrail_node.name, guardrail_node)
    builder.add_node(rag_node.name, rag_node)

    # Add edges
    builder.add_edge(START, guardrail_node.name)
    builder.add_edge(guardrail_node.name, rag_node.name)
    builder.add_edge(rag_node.name, END)

    # Compile with optional checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    logger.info(f"Chatbot graph built with model: {model}")

    return graph


def get_rag_node_from_graph(graph: CompiledStateGraph) -> Optional[RAGNode]:
    """Extract the RAG node from a compiled graph for streaming."""
    # The RAG node is stored in the graph's nodes dict
    if hasattr(graph, "nodes") and "rag" in graph.nodes:
        return graph.nodes["rag"]
    return None
