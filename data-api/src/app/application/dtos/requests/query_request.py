"""Query request model — routes by KB ``rag_mode``."""

from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Returns retrieved chunks only (no LLM answer).

    The API auto-detects the KB type from ``parser_config.rag_mode``:
      - ``classic``  uses semantic / hybrid / fuzzy pgvector search.
      - ``llm-wiki`` uses Elasticsearch BM25 + kNN with client-side RRF.
    """

    query_text: str = Field(..., description="The query text to search for", min_length=1)
    top_k: int = Field(default=10, ge=1, le=200, description="Number of results to return")

    # Classic-only parameters — ignored for llm-wiki KBs.
    search_type: Literal["semantic", "hybrid", "fuzzy"] = Field(
        default="semantic",
        description=(
            "Search type for classic RAG: "
            "'semantic' — vector similarity; "
            "'hybrid' — vector + full-text; "
            "'fuzzy' — trigram only"
        ),
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid search (1-alpha = text rank)",
    )
