from .base_query_request import BaseQueryRequest
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional


class ParserConfig(BaseModel):
    """Optional per-KB configuration for document parsing and RAG mode."""

    rag_mode: Literal["classic", "llm-wiki"] = Field(
        default="classic",
        description=(
            "'classic' — chunk → embed → pgvector pipeline; "
            "'llm-wiki' — legal-article-aware chunking + Elasticsearch hybrid "
            "(BM25 + kNN with RRF), tuned for Vietnamese legal corpora"
        ),
    )
    agentic_search: bool = Field(
        default=False,
        description=(
            "When true the /query endpoint runs the basin-pivot agentic loop "
            "(planner LLM fan-out across iterations + per-iter merge) before "
            "the selector; otherwise single-shot retrieve + selector."
        ),
    )
    # Tunable knobs — None means "fall back to the global default in
    # ``settings``". Stored verbatim on the KB row so each KB can be tuned
    # without restarting any service.
    top_n: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Final chunks the selector returns (default: SELECTOR_TOP_N=10).",
    )
    agentic_max_iter: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Hard cap on planner iterations (default: AGENTIC_MAX_ITER=5).",
    )
    agentic_top_k_per_iter: Optional[int] = Field(
        default=None, ge=1, le=50,
        description="Per-sub-query vector top_k per iteration (default: AGENTIC_TOP_K_PER_ITER=5).",
    )

    model_config = {"extra": "allow"}  # forward-compatible: unknown keys are preserved


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(..., description="Name of the data source")
    description: Optional[str] = Field(None, description="Description of the data source")
    embed_id: Optional[str] = Field(
        default="rag-embedding-model",
        alias="embeddedModelId",
        description="Embedded Model ID (defaults to 'rag-embedding-model')"
    )
    parser_config: Optional[ParserConfig] = Field(
        default=None,
        description="Optional parser / RAG-mode configuration.",
    )

    model_config = {"populate_by_name": True}


class UpdateKnowledgeBaseRequest(BaseModel):
    """Partial update — only the fields actually sent are written."""
    name: Optional[str] = Field(None, description="Rename the KB")
    description: Optional[str] = Field(None, description="Update description")
    parser_config: Optional[ParserConfig] = Field(
        None, description="Replace parser/RAG-mode configuration (incl. agentic_search)",
    )

    model_config = {"extra": "ignore"}


class QueryDataKnowledgeBaseRequest(BaseQueryRequest):
    pass
