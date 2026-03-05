"""API module."""

from .v1.chatbot_controller import router as chatbot_router

__all__ = ["chatbot_router"]
