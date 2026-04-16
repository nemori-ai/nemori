"""Unified LLM call orchestration with retry, concurrency, and budget."""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from nemori.domain.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    TokenBudgetExceeded,
)
from nemori.domain.interfaces import LLMProvider

logger = logging.getLogger("nemori")


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[dict, ...]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    response_format: dict[str, str] | None = None
    timeout: float = 30.0
    retries: int = 3
    metadata: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    request_id: str


@dataclass
class OrchestratorStats:
    total_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    requests_by_phase: dict[str, int] = field(default_factory=dict)
    tokens_by_phase: dict[str, int] = field(default_factory=dict)


class LLMOrchestrator:
    """Unified LLM call orchestration."""

    def __init__(
        self,
        provider: LLMProvider,
        default_model: str,
        max_concurrent: int = 10,
        token_budget: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._token_budget = token_budget
        self._log = logger or logging.getLogger("nemori")
        self._total_requests = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        self._requests_by_phase: dict[str, int] = {}
        self._tokens_by_phase: dict[str, int] = {}
        # Enable usage tracking if provider explicitly declares support
        self._track_usage = getattr(provider, "supports_usage_tracking", False)

    async def execute(self, request: LLMRequest) -> LLMResponse:
        if self._token_budget and self._total_tokens >= self._token_budget:
            raise TokenBudgetExceeded(
                "Token budget exceeded",
                used=self._total_tokens,
                budget=self._token_budget,
            )

        model = request.model or self._default_model
        request_id = str(uuid.uuid4())[:8]
        last_error: Exception | None = None

        for attempt in range(request.retries):
            try:
                async with self._semaphore:
                    start = time.monotonic()
                    extra_kwargs: dict[str, Any] = {}
                    if request.response_format is not None:
                        extra_kwargs["response_format"] = request.response_format

                    call_kwargs = dict(
                        model=model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        **extra_kwargs,
                    )
                    # Use complete_with_usage if available to capture token counts
                    usage = TokenUsage()
                    if self._track_usage:
                        result = await asyncio.wait_for(
                            self._provider.complete_with_usage(
                                list(request.messages), **call_kwargs,
                            ),
                            timeout=request.timeout,
                        )
                        content, raw_usage = result
                        usage = TokenUsage(
                            prompt_tokens=raw_usage.get("prompt_tokens", 0),
                            completion_tokens=raw_usage.get("completion_tokens", 0),
                        )
                    else:
                        content = await asyncio.wait_for(
                            self._provider.complete(
                                list(request.messages), **call_kwargs,
                            ),
                            timeout=request.timeout,
                        )
                        usage = TokenUsage()
                    latency = (time.monotonic() - start) * 1000

                self._total_requests += 1
                self._total_tokens += usage.total
                self._total_prompt_tokens += usage.prompt_tokens
                self._total_completion_tokens += usage.completion_tokens
                self._total_latency_ms += latency

                # Track per-phase stats from metadata
                phase = request.metadata.get("generator", "unknown")
                self._requests_by_phase[phase] = self._requests_by_phase.get(phase, 0) + 1
                self._tokens_by_phase[phase] = self._tokens_by_phase.get(phase, 0) + usage.total

                response = LLMResponse(
                    content=content,
                    model=model,
                    usage=usage,
                    latency_ms=latency,
                    request_id=request_id,
                )
                self._log.debug(
                    "LLM request %s completed in %.0fms",
                    request_id, latency,
                )
                return response

            except (LLMAuthError, TokenBudgetExceeded):
                raise
            except Exception as e:
                last_error = e
                self._total_errors += 1
                if attempt < request.retries - 1:
                    delay = min(1.0 * (2 ** attempt) + random.uniform(0, 0.5), 30.0)
                    self._log.warning(
                        "LLM request %s attempt %d failed: %s. Retrying in %.1fs",
                        request_id, attempt + 1, e, delay,
                    )
                    await asyncio.sleep(delay)

        raise LLMError(f"All {request.retries} attempts failed: {last_error}") from last_error

    async def execute_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        tasks = [self.execute(req) for req in requests]
        return await asyncio.gather(*tasks)

    @property
    def stats(self) -> OrchestratorStats:
        avg = (
            self._total_latency_ms / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return OrchestratorStats(
            total_requests=self._total_requests,
            total_tokens=self._total_tokens,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            total_errors=self._total_errors,
            avg_latency_ms=avg,
            requests_by_phase=dict(self._requests_by_phase),
            tokens_by_phase=dict(self._tokens_by_phase),
        )
