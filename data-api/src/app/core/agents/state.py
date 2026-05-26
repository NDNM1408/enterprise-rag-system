"""
LangGraph state definition for the chatbot agent.
"""

from typing import List, Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ChatbotState(TypedDict):
    """
    State for the chatbot agent.

    Attributes:
        messages: Chat history with automatic message aggregation
        agent_id: ID of the agent being used
        user_id: ID of the user
        kb_ids: List of linked knowledge base IDs
        guardrail_passed: Whether the input passed guardrail validation
        context: Retrieved context from RAG (optional)
    """
    messages: Annotated[List[BaseMessage], add_messages]
    agent_id: str
    user_id: str
    kb_ids: List[str]
    guardrail_passed: bool
    context: Optional[str]
