"""Async event bus for decoupling pipeline stages."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("nemori")


class EventBus:
    """Simple async publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}
        self._tasks: set[asyncio.Task] = set()

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def emit(self, event: str, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            task = asyncio.create_task(handler(**kwargs))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def drain(self, timeout: float = 30.0) -> list[Exception]:
        """Wait for all background tasks and collect errors."""
        errors: list[Exception] = []
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            done, not_done = await asyncio.wait(pending, timeout=timeout)
            for t in done:
                if t.exception():
                    errors.append(t.exception())
            for t in not_done:
                t.cancel()
        return errors
