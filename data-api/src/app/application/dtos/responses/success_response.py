"""Unified success response model."""

from typing import TypeVar, Generic, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


T = TypeVar('T')


class ResponseMeta(BaseModel):
    """Metadata included in all responses."""

    request_id: str = Field(..., description="Unique identifier for request tracing")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of response"
    )


class SuccessResponse(BaseModel, Generic[T]):
    """
    Standardized success response envelope.

    Example:
        {
            "success": true,
            "data": {...},
            "meta": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-02-18T10:30:00.000Z"
            }
        }
    """

    success: bool = Field(True, description="Always true for success responses")
    data: T = Field(..., description="Response payload")
    meta: ResponseMeta = Field(..., description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"id": "123", "name": "Example"},
                "meta": {
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2026-02-18T10:30:00.000Z"
                }
            }
        }


def create_success_response(data: Any, request_id: str) -> dict:
    """
    Helper function to create a success response.

    Args:
        data: Response payload
        request_id: Request ID from middleware

    Returns:
        Dictionary with standardized success response structure
    """
    return {
        "success": True,
        "data": data,
        "meta": {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
