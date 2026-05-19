"""Document management controller."""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, Path, Request, Body

from app.configurations.dependencies import get_document_service
from app.application.services.document_service import DocumentsService
from app.application.dtos.responses.success_response import create_success_response
from app.exceptions import ValidationError


_path = "/api/v1"
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=_path,
    tags=["documents"]
)


@router.get(
    "/{kb_id}/documents",
    summary="List documents",
    description="List all documents in a knowledge base"
)
async def list_documents(
    request: Request,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    document_service: DocumentsService = Depends(get_document_service)
):
    """
    Retrieve all documents belonging to a knowledge base.

    Raises:
        ResourceNotFoundError: If the knowledge base does not exist
    """
    documents = await document_service.list_documents(kb_id=kb_id)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={"kb_id": kb_id, "count": len(documents), "documents": documents},
        request_id=request_id
    )


@router.post(
    "/{kb_id}/documents",
    summary="Upload documents",
    description="Upload one or more documents to a knowledge base for processing"
)
async def create_documents(
    request: Request,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    cmetadata: Optional[str] = Form(None, description="Custom metadata as JSON string"),
    files: List[UploadFile] = File(..., description="One or more files to upload"),
    document_service: DocumentsService = Depends(get_document_service)
):
    """
    Upload documents to a knowledge base.

    Args:
        request: FastAPI request object (injected)
        kb_id: Knowledge base identifier
        cmetadata: Optional custom metadata as JSON string
        files: List of files to upload
        document_service: Document service (injected)

    Returns:
        Standardized success response with upload confirmation

    Raises:
        ValidationError: If cmetadata is not valid JSON
        ResourceNotFoundError: If knowledge base not found
        ConflictError: If documents with same names already exist
    """
    # Parse and validate cmetadata if provided
    parsed_cmetadata = None
    if cmetadata:
        try:
            parsed_cmetadata = json.loads(cmetadata)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message="Invalid JSON in cmetadata field",
                details={"error": str(e), "cmetadata": cmetadata}
            )

    # Upload documents
    await document_service.add_documents(
        kb_id=kb_id,
        files=files,
        cmetadata=parsed_cmetadata,
    )

    # Return success response
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={
            "kb_id": kb_id,
            "uploaded_count": len(files),
            "filenames": [file.filename for file in files],
            "status": "processing"
        },
        request_id=request_id
    )


@router.post(
    "/internal/parse-callback",
    summary="Document-parsing service callback",
    description=(
        "Internal webhook the document-parsing worker calls to report progress, "
        "completion, or failure of a parse job. Trusted on internal network only."
    ),
)
async def parse_callback(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    document_service: DocumentsService = Depends(get_document_service),
):
    await document_service.handle_parse_callback(payload)
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={"ack": True, "job_id": payload.get("job_id") or payload.get("id")},
        request_id=request_id,
    )


@router.delete(
    "/{kb_id}/documents/{doc_id}",
    summary="Delete a document",
    description="Delete a document and all associated chunks from S3 and the database"
)
async def delete_document(
    request: Request,
    kb_id: str = Path(..., description="Knowledge Base ID"),
    doc_id: str = Path(..., description="Document ID"),
    document_service: DocumentsService = Depends(get_document_service)
):
    """
    Delete a document from a knowledge base.

    Removes:
    - The original document file from S3
    - All chunk files from S3
    - The document record and all associated chunks/events from the database

    Raises:
        ResourceNotFoundError: If the document does not exist in this knowledge base
        ExternalServiceError: If S3 deletion fails
    """
    await document_service.delete_document(kb_id=kb_id, doc_id=doc_id)

    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={"kb_id": kb_id, "doc_id": doc_id, "deleted": True},
        request_id=request_id
    )