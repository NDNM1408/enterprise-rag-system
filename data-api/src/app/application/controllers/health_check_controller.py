"""Health check controller."""

from fastapi import APIRouter, Request

from app.application.dtos.responses.success_response import create_success_response


router = APIRouter(tags=["health"])


@router.get(
    "/",
    summary="Health check",
    description="Check if the API is running and responsive"
)
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns:
        Standardized success response with status "OK"
    """
    request_id = getattr(request.state, "request_id", "unknown")
    return create_success_response(
        data={"status": "OK", "service": "data-api"},
        request_id=request_id
    )
