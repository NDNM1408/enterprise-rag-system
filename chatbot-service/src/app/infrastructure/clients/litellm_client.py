"""LiteLLM client for LLM interactions."""

import logging
from typing import List, AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

from app.configurations.settings import settings


logger = logging.getLogger(__name__)


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
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM chat error: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Chunks of generated response
        """
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LiteLLM stream chat error: {e}")
            raise
