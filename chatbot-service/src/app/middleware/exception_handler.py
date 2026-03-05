"""Global exception handlers for consistent error responses."""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.exceptions import (
    DomainException,
    ResourceNotFoundError,
    ConflictError,
    ValidationError,
    DatabaseError,
    ExternalServiceError,
    GuardrailError,
)
from app.application.dtos.responses.error_response import create_error_response


logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    """Get request_id from request state, or generate fallback."""
    return getattr(request.state, "request_id", "unknown")


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""

    @app.exception_handler(ResourceNotFoundError)
    async def handle_not_found_error(request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        """Handle resource not found errors."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Resource not found: {exc.message}")

        error_response = create_error_response(
            status=status.HTTP_404_NOT_FOUND,
            title="Resource Not Found",
            detail=exc.message,
            request_id=request_id,
            instance=str(request.url),
            errors=exc.details if exc.details else None
        )

        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_response.model_dump()
        )

    @app.exception_handler(ConflictError)
    async def handle_conflict_error(request: Request, exc: ConflictError) -> JSONResponse:
        """Handle conflict errors."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Conflict error: {exc.message}")

        error_response = create_error_response(
            status=status.HTTP_409_CONFLICT,
            title="Conflict",
            detail=exc.message,
            request_id=request_id,
            instance=str(request.url),
            errors=exc.details if exc.details else None
        )

        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=error_response.model_dump()
        )

    @app.exception_handler(ValidationError)
    async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle domain validation errors."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Validation error: {exc.message}")

        error_response = create_error_response(
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Validation Error",
            detail=exc.message,
            request_id=request_id,
            instance=str(request.url),
            errors=exc.details if exc.details else None
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump()
        )

    @app.exception_handler(GuardrailError)
    async def handle_guardrail_error(request: Request, exc: GuardrailError) -> JSONResponse:
        """Handle guardrail errors."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Guardrail error: {exc.message}")

        error_response = create_error_response(
            status=status.HTTP_400_BAD_REQUEST,
            title="Guardrail Error",
            detail=exc.message,
            request_id=request_id,
            instance=str(request.url),
            errors=exc.details if exc.details else None
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI/Pydantic request validation errors."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] Request validation error: {exc.errors()}")

        validation_errors = {}
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            validation_errors[field] = error["msg"]

        error_response = create_error_response(
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Request Validation Error",
            detail="Invalid request parameters",
            request_id=request_id,
            instance=str(request.url),
            errors=validation_errors
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump()
        )

    @app.exception_handler(DatabaseError)
    async def handle_database_error(request: Request, exc: DatabaseError) -> JSONResponse:
        """Handle database errors."""
        request_id = get_request_id(request)
        logger.error(f"[{request_id}] Database error: {exc.message}", exc_info=True)

        error_response = create_error_response(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Database Error",
            detail="An error occurred while accessing the database",
            request_id=request_id,
            instance=str(request.url)
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump()
        )

    @app.exception_handler(SQLAlchemyError)
    async def handle_sqlalchemy_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Handle SQLAlchemy errors."""
        request_id = get_request_id(request)
        logger.error(f"[{request_id}] SQLAlchemy error: {str(exc)}", exc_info=True)

        error_response = create_error_response(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Database Error",
            detail="An error occurred while accessing the database",
            request_id=request_id,
            instance=str(request.url)
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump()
        )

    @app.exception_handler(ExternalServiceError)
    async def handle_external_service_error(
        request: Request, exc: ExternalServiceError
    ) -> JSONResponse:
        """Handle external service errors."""
        request_id = get_request_id(request)
        logger.error(f"[{request_id}] External service error: {exc.message}", exc_info=True)

        error_response = create_error_response(
            status=status.HTTP_502_BAD_GATEWAY,
            title="External Service Error",
            detail=exc.message,
            request_id=request_id,
            instance=str(request.url)
        )

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=error_response.model_dump()
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTPException."""
        request_id = get_request_id(request)
        logger.warning(f"[{request_id}] HTTP exception: {exc.status_code} - {exc.detail}")

        title_map = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            422: "Unprocessable Entity",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable"
        }

        error_response = create_error_response(
            status=exc.status_code,
            title=title_map.get(exc.status_code, "Error"),
            detail=str(exc.detail),
            request_id=request_id,
            instance=str(request.url)
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        request_id = get_request_id(request)
        logger.error(
            f"[{request_id}] Unexpected error: {type(exc).__name__}: {str(exc)}",
            exc_info=True
        )

        error_response = create_error_response(
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal Server Error",
            detail="An unexpected error occurred",
            request_id=request_id,
            instance=str(request.url)
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump()
        )
