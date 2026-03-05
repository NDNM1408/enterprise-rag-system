"""
Custom exception hierarchy for domain errors.
These exceptions should be raised by services and repositories, then caught by
middleware and converted to appropriate HTTP responses.
"""


class DomainException(Exception):
    """Base exception for all domain-level errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ResourceNotFoundError(DomainException):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource_type: str, resource_id: str, details: dict = None):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(DomainException):
    """Raised when an operation conflicts with existing state (e.g., duplicate resource)."""
    pass


class ValidationError(DomainException):
    """Raised when input validation fails."""
    pass


class DatabaseError(DomainException):
    """Raised when database operations fail."""
    pass


class ExternalServiceError(DomainException):
    """Raised when an external service (S3, RabbitMQ, etc.) fails."""

    def __init__(self, service_name: str, message: str, details: dict = None):
        super().__init__(f"{service_name} error: {message}", details)
        self.service_name = service_name
