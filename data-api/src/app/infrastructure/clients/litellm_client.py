"""LiteLLM client for LLM interactions."""

import logging
from typing import List, AsyncIterator, Dict, Optional, Any

from openai import AsyncOpenAI

from app.configurations.configurations import settings


logger = logging.getLogger(__name__)

# Per-turn budget for Gemini thinking. 8000 tokens covers most chain-of-
# thought on technical Q&A without running away on simple prompts.
THINKING_BUDGET_TOKENS = 8000


def _thinking_extra_body() -> Dict[str, Any]:
    """Anthropic-style thinking param. LiteLLM normalises this across
    Gemini/Vertex/Bedrock/Anthropic providers."""
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS,
        }
    }


class LiteLLMClient:
    """Client for interacting with LiteLLM API."""

    def __init__(self):
        self.api_base = settings.LITELLM_API_BASE
        self.api_key = settings.LITELLM_API_KEY
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    async def chat(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
    ) -> str:
        """Send a chat completion request.

        ``enable_thinking`` defaults on (used for the final answer). Fact
        extraction passes ``False`` — per-chunk extraction is a mechanical
        task that doesn't benefit from chain-of-thought and shouldn't burn
        the thinking budget. Falls back to a no-thinking call if the upstream
        model rejects the thinking parameter."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if not enable_thinking:
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LiteLLM chat error: {e}")
                raise

        try:
            response = await self.client.chat.completions.create(
                **kwargs, extra_body=_thinking_extra_body(),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Thinking-enabled chat failed (%s); retrying without thinking", e)
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e2:
                logger.error(f"LiteLLM chat error: {e2}")
                raise

    async def stream_chat(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, str]]:
        """Stream a chat completion. Yields tagged events:

          ``{"type": "thinking", "delta": "..."}``  for reasoning tokens
          ``{"type": "content",  "delta": "..."}``  for answer tokens

        Tagged dicts (not raw strings) let downstream layers (chat service,
        SSE controller, UI) route the two streams independently — thinking
        rendered in a collapsible block, content as the final answer."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = await self.client.chat.completions.create(
                **kwargs, extra_body=_thinking_extra_body(),
            )
        except Exception as e:
            logger.warning(
                "Thinking-enabled stream failed (%s); falling back without thinking", e,
            )
            try:
                stream = await self.client.chat.completions.create(**kwargs)
            except Exception as e2:
                logger.error(f"LiteLLM stream chat error: {e2}")
                raise

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # LiteLLM exposes reasoning as either ``reasoning_content``
                # (OpenAI/Anthropic convention) or ``thinking_blocks`` (raw
                # Anthropic). Surface either as a thinking event.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "thinking", "delta": reasoning}
                if getattr(delta, "content", None):
                    yield {"type": "content", "delta": delta.content}
        except Exception as e:
            logger.error(f"LiteLLM stream iteration error: {e}")
            raise
