"""Query controller — single endpoint for all KB types."""

import logging
from fastapi import APIRouter, Path, Request as FastAPIRequest, Depends

from app.application.dtos.requests.query_request import QueryRequest
from app.application.dtos.responses.success_response import create_success_response
from app.application.services.query_service import QueryService, get_query_service

_path = "/api/v1/query"
logger = logging.getLogger(__name__)

router = APIRouter(prefix=_path, tags=["query"])


@router.post(
    "/{kb_id}",
    summary="Query endpoint",
    description=(
        "Query a knowledge base using the appropriate method based on its type. "
        "Returns retrieved chunks only (no LLM-generated answer)."
    ),
)
async def query(
    request: FastAPIRequest,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    query_request: QueryRequest = ...,
    service: QueryService = Depends(get_query_service),
):
    """Routes by ``parser_config.rag_mode``:

    - ``classic``: semantic / hybrid / fuzzy pgvector search.
    - ``llm-wiki``: Elasticsearch BM25 + kNN with client-side RRF.
    """
    result = await service.query(
        kb_id=kb_id,
        query_text=query_request.query_text,
        top_k=query_request.top_k,
        search_type=query_request.search_type,
        alpha=query_request.alpha,
    )

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=result, request_id=request_id)
