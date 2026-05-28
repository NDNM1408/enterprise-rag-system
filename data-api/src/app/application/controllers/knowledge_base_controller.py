"""Knowledge base management controller."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Path, Query, Request as FastAPIRequest

from app.application.dtos.requests.knowledge_base_request import (
    CreateKnowledgeBaseRequest,
    UpdateKnowledgeBaseRequest,
)
from app.application.services.knowledge_base_service import KnowledgeBaseService
from app.configurations.dependencies import get_knowledge_base_service
from app.application.dtos.responses.success_response import create_success_response
from app.exceptions import ValidationError


_path = "/api/v1/knowledge_base"
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=_path,
    tags=["knowledge_base"]
)


@router.get(
    "/",
    summary="List knowledge bases",
    description="Get paginated list of knowledge bases with optional filtering and sorting"
)
async def list_knowledge_base(
    request: FastAPIRequest,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (max 100)"),
    filter: Optional[str] = Query(None, description="Filter as JSON string, e.g., {\"name\": \"My KB\"}"),
    sort: Optional[str] = Query(None, description="Sort as JSON string, e.g., {\"create_time\": \"desc\"}"),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    List knowledge bases with pagination.

    Args:
        request: FastAPI request object (injected)
        page: Page number (1-indexed)
        page_size: Number of items per page
        filter: Optional JSON string for filtering
        sort: Optional JSON string for sorting
        knowledge_base_service: Knowledge base service (injected)

    Returns:
        Standardized success response with paginated knowledge base list

    Raises:
        ValidationError: If filter or sort JSON is invalid
    """
    # Parse filter and sort parameters
    filter_dict = None
    sort_dict = None

    if filter:
        try:
            filter_dict = json.loads(filter)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message="Invalid JSON in filter parameter",
                details={"error": str(e), "filter": filter}
            )

    if sort:
        try:
            sort_dict = json.loads(sort)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message="Invalid JSON in sort parameter",
                details={"error": str(e), "sort": sort}
            )

    # Get paginated results
    result = await knowledge_base_service.paging(
        page=page,
        page_size=page_size,
        where=filter_dict,
        order_by=sort_dict
    )

    # Return standardized response
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(data=result, request_id=request_id)


@router.post(
    "",
    summary="Create knowledge base",
    description="Create a new knowledge base"
)
async def create_knowledge_base(
    request: FastAPIRequest,
    kb_request: CreateKnowledgeBaseRequest,
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    Create a new knowledge base.

    Args:
        request: FastAPI request object (injected)
        kb_request: Knowledge base creation request
        knowledge_base_service: Knowledge base service (injected)

    Returns:
        Standardized success response with created knowledge base

    Raises:
        ValidationError: If embed_id is missing or invalid
    """
    kb = await knowledge_base_service.create(kb_request)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data=kb.to_dict() if hasattr(kb, 'to_dict') else kb,
        request_id=request_id
    )


@router.get(
    "/{kb_id}",
    summary="Get knowledge base",
    description="Get details of a specific knowledge base"
)
async def get_knowledge_base(
    request: FastAPIRequest,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    Get knowledge base by ID.

    Args:
        request: FastAPI request object (injected)
        kb_id: Knowledge base identifier
        knowledge_base_service: Knowledge base service (injected)

    Returns:
        Standardized success response with knowledge base details

    Raises:
        ResourceNotFoundError: If knowledge base not found
    """
    kb = await knowledge_base_service.get(kb_id)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data=kb.to_dict() if hasattr(kb, 'to_dict') else kb,
        request_id=request_id
    )


@router.patch(
    "/{kb_id}",
    summary="Update knowledge base",
    description=(
        "Partial update — send only the fields to change. "
        "``parser_config`` is replaced wholesale (incl. agentic_search)."
    ),
)
async def update_knowledge_base(
    request: FastAPIRequest,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    kb_request: UpdateKnowledgeBaseRequest = ...,
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
):
    """PATCH a KB. Only keys actually present in the body are written."""
    patch = kb_request.model_dump(exclude_unset=True, exclude_none=False)
    # Convert nested Pydantic ParserConfig to dict so the repo can write it
    # straight into the jsonb column.
    if isinstance(patch.get("parser_config"), dict) is False and patch.get("parser_config") is not None:
        patch["parser_config"] = patch["parser_config"].model_dump()
    kb = await knowledge_base_service.update(kb_id, patch)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data=kb.to_dict() if hasattr(kb, "to_dict") else kb,
        request_id=request_id,
    )


@router.delete(
    "/{kb_id}",
    summary="Delete knowledge base",
    description="Delete a knowledge base and all associated documents"
)
async def delete_knowledge_base(
    request: FastAPIRequest,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    knowledge_base_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    Delete a knowledge base.

    Args:
        request: FastAPI request object (injected)
        kb_id: Knowledge base identifier
        knowledge_base_service: Knowledge base service (injected)

    Returns:
        Standardized success response confirming deletion

    Raises:
        ResourceNotFoundError: If knowledge base not found
    """
    result = await knowledge_base_service.delete(kb_id)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={"kb_id": kb_id, "deleted": True},
        request_id=request_id
    )
