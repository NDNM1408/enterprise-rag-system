"""
Query controller — single endpoint for all knowledge base types.

Automatically detects KB type (classic vs graphrag) and routes to the
appropriate search implementation.
"""

import logging
from fastapi import APIRouter, Path, Request as FastAPIRequest, Depends

from app.application.dtos.requests.query_request import QueryRequest
from app.application.dtos.responses.success_response import create_success_response
from app.application.services.query_service import (
    QueryService,
    get_query_service,
)

_path = "/api/v1/query"
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=_path,
    tags=["query"]
)


@router.post(
    "/{kb_id}",
    summary="Query endpoint",
    description=(
        "Query a knowledge base using the appropriate method based on its type. "
        "Returns retrieved context/chunks only (no LLM-generated answer). "
        "For classic RAG KBs, supports semantic/hybrid/fuzzy search. "
        "For graph RAG KBs, supports LightRAG-style graph traversal."
    ),
)
async def query(
    request: FastAPIRequest,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    query_request: QueryRequest = ...,
    service: QueryService = Depends(get_query_service),
):
    """
    Query endpoint for all knowledge base types.

    The API automatically detects the KB type based on the knowledge base
    configuration (parser_config.rag_mode) and uses the appropriate search method.

    **Classic RAG** (rag_mode='classic'):
    - `search_type='semantic'`: Vector similarity search
    - `search_type='hybrid'`: Combines vector + full-text search
    - `search_type='fuzzy'`: Trigram-based fuzzy matching

    **Graph RAG** (rag_mode='graphrag'):
    - `mode='local'`: Entity-centric search
    - `mode='global'`: Community/relationship-centric
    - `mode='hybrid'`: Combines local + global results
    - `mode='naive'`: Plain vector chunk search (no graph traversal)
    - `mode='mix'`: Hybrid + naive (includes both graph data AND chunks)

    Returns:
        Standardized success response with retrieved context/chunks.
    """
    result = await service.query(
        kb_id=kb_id,
        query_text=query_request.query_text,
        top_k=query_request.top_k,
        # Classic RAG params
        search_type=query_request.search_type,
        alpha=query_request.alpha,
        # Graph RAG params
        mode=query_request.mode,
        chunk_top_k=query_request.chunk_top_k,
        max_entity_tokens=query_request.max_entity_tokens,
        max_relation_tokens=query_request.max_relation_tokens,
        max_total_tokens=query_request.max_total_tokens,
    )

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=result, request_id=request_id)
