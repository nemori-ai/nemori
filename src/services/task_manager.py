"""Utility classes for orchestrating background semantic generation tasks."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SemanticTaskManager:
    """Wraps the semantic generation executor with retry logic."""

    def __init__(self, executor: ThreadPoolExecutor, max_retries: int = 1) -> None:
        self._executor = executor
        self._max_retries = max_retries

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        task = partial(self._run_with_retry, func, args, kwargs)
        return self._executor.submit(task)

    def _run_with_retry(self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive retry path
                attempt += 1
                if attempt > self._max_retries:
                    logger.error("Semantic task failed after retries: %s", exc)
                    raise
                logger.warning("Retrying semantic task (attempt %s): %s", attempt, exc)


__all__ = ["SemanticTaskManager"]
