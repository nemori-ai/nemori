"""Async OpenAI-compatible LLM client implementing LLMProvider."""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from nemori.domain.exceptions import LLMError, LLMAuthError, LLMRateLimitError

logger = logging.getLogger("nemori")

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
GPT5_MODEL_PREFIXES = ("gpt-5",)


def _base_model_name(model: str) -> str:
    """Return provider-free model name, e.g. openai/gpt-5-mini -> gpt-5-mini."""
    return model.rsplit("/", maxsplit=1)[-1].lower()


def is_gpt5_model(model: str) -> bool:
    return _base_model_name(model).startswith(GPT5_MODEL_PREFIXES)


def _chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if is_gpt5_model(model):
        params["max_completion_tokens"] = extra.pop(
            "max_completion_tokens", max_tokens
        )
        reasoning_effort = extra.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens

    params.update(extra)
    return params


class AsyncLLMClient:
    """Async LLM client wrapping the OpenAI API."""

    supports_usage_tracking: bool = True

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict], **kwargs: Any) -> str:
        content, _ = await self.complete_with_usage(messages, **kwargs)
        return content

    async def complete_with_usage(
        self, messages: list[dict], **kwargs: Any
    ) -> tuple[str, dict[str, int]]:
        """Return (content, {"prompt_tokens": ..., "completion_tokens": ...})."""
        model = kwargs.pop("model", DEFAULT_OPENAI_CHAT_MODEL)
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 2000)

        try:
            request_kwargs = _chat_completion_kwargs(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=kwargs,
            )
            response = await self._client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }
            return content, usage
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "403" in error_str:
                raise LLMAuthError(f"Authentication failed: {e}") from e
            if "429" in error_str:
                raise LLMRateLimitError(f"Rate limited: {e}") from e
            raise LLMError(f"LLM call failed: {e}") from e
