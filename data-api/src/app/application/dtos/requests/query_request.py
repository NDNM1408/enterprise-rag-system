"""
Query request model for all knowledge base types.

Supports both classic RAG (semantic/hybrid/fuzzy search) and graph RAG
(LightRAG-style graph traversal).

This is a query-only API that returns retrieved context/chunks.
"""

from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Query request for all knowledge base types.

    The API automatically detects the KB type (classic vs graphrag) based on
    the knowledge base configuration and uses the appropriate search method.

    Returns retrieved context/chunks only (no LLM-generated answer).
    """

    # ------------------------------------------------------------------
    # Common parameters (apply to both classic and graph RAG)
    # ------------------------------------------------------------------
    query_text: str = Field(
        ...,
        description="The query text to search for",
        min_length=1
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Number of results to return (classic) or entities/relations to retrieve (graph)"
    )

    # ------------------------------------------------------------------
    # Classic RAG parameters (used when KB has rag_mode='classic')
    # ------------------------------------------------------------------
    search_type: Literal["semantic", "hybrid", "fuzzy"] = Field(
        default="semantic",
        description=(
            "Search type for classic RAG: "
            "'semantic' — vector similarity search; "
            "'hybrid' — combines vector + full-text search; "
            "'fuzzy' — full-text search only (trigram similarity)"
        ),
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid search (0-1, 1-alpha for text rank)"
    )

    # ------------------------------------------------------------------
    # Graph RAG parameters (used when KB has rag_mode='graphrag')
    # Modeled after LightRAG's QueryParam
    # ------------------------------------------------------------------
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="hybrid",
        description=(
            "GraphRAG retrieval mode: "
            "'local' — entity-centric search; "
            "'global' — community/relationship-centric; "
            "'hybrid' — combines local + global results; "
            "'naive' — plain vector chunk search (no graph traversal); "
            "'mix' — hybrid + naive (includes both graph data AND chunks)"
        ),
    )
    chunk_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of document chunks to retrieve in graph RAG"
    )
    max_entity_tokens: int = Field(
        default=4000,
        ge=100,
        le=16000,
        description="Maximum tokens allocated for entity context in graph RAG"
    )
    max_relation_tokens: int = Field(
        default=4000,
        ge=100,
        le=16000,
        description="Maximum tokens allocated for relationship context in graph RAG"
    )
    max_total_tokens: int = Field(
        default=16000,
        ge=1000,
        le=64000,
        description="Maximum total tokens budget for the entire query context in graph RAG"
    )
