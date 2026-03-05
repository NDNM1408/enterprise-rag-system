from .base_query_request import BaseQueryRequest
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional


class ParserConfig(BaseModel):
    """Optional per-KB configuration for document parsing and RAG mode."""

    rag_mode: Literal["classic", "graphrag"] = Field(
        default="classic",
        description=(
            "'classic' — chunk → embed → pgvector pipeline; "
            "'graphrag' — parse → entity extraction → Neo4j graph"
        ),
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


class QueryDataKnowledgeBaseRequest(BaseQueryRequest):
    pass
