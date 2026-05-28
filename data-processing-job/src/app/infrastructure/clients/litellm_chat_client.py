"""Async chat client for the LiteLLM proxy.

Used by the hier_v2 splitter to ask Gemini Flash for a per-table summary
({retrieval_text, generation_text}). Strict-JSON response, temperature 0,
no thinking — keeps the ingestion path deterministic and cheap.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from app.configurations.configurations import settings

log = logging.getLogger(__name__)


class LiteLLMChatClient:
    """Tiny OpenAI-compatible chat-completion wrapper.

    Kept dependency-light (httpx, no openai SDK) and async so the splitter
    can ``asyncio.gather`` table summaries across blocks if it ever wants
    to. Returns the raw assistant content string — caller parses JSON.
    """

    def __init__(self, timeout: float = 90.0):
        self.api_base = settings.LITELLM_API_BASE
        self.api_key = settings.LITELLM_API_KEY
        self.timeout = timeout

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float = 0.0,
        thinking_budget: int = 0,
    ) -> str:
        """Send a system+user pair, ask for JSON output. Returns content string.

        Extras passed via ``extra_body``-style fields the LiteLLM proxy
        forwards: ``response_format`` (OpenAI) for JSON mode, plus the
        Anthropic-style ``thinking`` knob (LiteLLM translates per provider)
        so we can disable thinking on Gemini.
        """
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if thinking_budget > 0 else "disabled",
                         "budget_tokens": thinking_budget or 0},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            log.warning("LiteLLM chat: unexpected response shape: %s", data)
            return ""
