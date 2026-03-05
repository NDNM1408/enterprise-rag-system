"""Unified error response model following RFC 9457 (Problem Details for HTTP APIs)."""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standardized error response following RFC 9457.
    """

    type: str = Field(
        ...,
        description="URI reference identifying the problem type"
    )
    title: str = Field(
        ...,
        description="Short, human-readable summary of the problem"
    )
    status: int = Field(
        ...,
        description="HTTP status code",
        ge=400,
        le=599
    )
    detail: str = Field(
        ...,
        description="Human-readable explanation specific to this occurrence"
    )
    instance: str = Field(
        ...,
        description="URI reference identifying the specific occurrence"
    )
    request_id: str = Field(
        ...,
        description="Unique identifier for request tracing"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of error"
    )
    errors: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details (e.g., validation errors)"
    )


def create_error_response(
    status: int,
    title: str,
    detail: str,
    request_id: str,
    instance: str,
    error_type: Optional[str] = None,
    errors: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Helper function to create an error response.
    """
    if error_type is None:
        error_type_map = {
            400: "bad-request",
            401: "unauthorized",
            403: "forbidden",
            404: "not-found",
            409: "conflict",
            422: "validation-error",
            500: "internal-server-error",
            502: "bad-gateway",
            503: "service-unavailable"
        }
        type_slug = error_type_map.get(status, "error")
        error_type = f"https://api.chatbot.com/errors/{type_slug}"

    return ErrorResponse(
        type=error_type,
        title=title,
        status=status,
        detail=detail,
        instance=instance,
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat(),
        errors=errors
    )
