"""
Guardrail node for input validation.
"""

import logging
from typing import Dict, Any

from app.core.agents.state import ChatbotState
from app.exceptions import GuardrailError


logger = logging.getLogger(__name__)

BLOCKED_PATTERNS = [
    "ignore all previous instructions",
    "disregard all prior",
    "you are now",
    "act as if",
]

MAX_MESSAGE_LENGTH = 10000


class GuardrailNode:
    """Node for validating user input."""

    def __init__(self):
        self.name = "guardrail"

    async def __call__(self, state: ChatbotState) -> Dict[str, Any]:
        """Validate the latest user message."""
        messages = state.get("messages", [])
        if not messages:
            return {"guardrail_passed": True}

        last_message = messages[-1]
        if not hasattr(last_message, "content"):
            return {"guardrail_passed": True}

        content = last_message.content.lower() if last_message.content else ""

        for pattern in BLOCKED_PATTERNS:
            if pattern in content:
                logger.warning(f"Guardrail blocked message containing: {pattern}")
                raise GuardrailError(
                    "Input contains potentially harmful content",
                    {"pattern": pattern}
                )

        if len(content) > MAX_MESSAGE_LENGTH:
            raise GuardrailError(
                f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters",
                {"length": len(content)}
            )

        logger.info("Guardrail validation passed")
        return {"guardrail_passed": True}
