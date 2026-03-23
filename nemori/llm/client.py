"""Async OpenAI-compatible LLM client implementing LLMProvider."""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from nemori.domain.exceptions import LLMError, LLMAuthError, LLMRateLimitError

logger = logging.getLogger("nemori")


class AsyncLLMClient:
    """Async LLM client wrapping the OpenAI API."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict], **kwargs: Any) -> str:
        model = kwargs.pop("model", "gpt-4o-mini")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 2000)

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "403" in error_str:
                raise LLMAuthError(f"Authentication failed: {e}") from e
            if "429" in error_str:
                raise LLMRateLimitError(f"Rate limited: {e}") from e
            raise LLMError(f"LLM call failed: {e}") from e
