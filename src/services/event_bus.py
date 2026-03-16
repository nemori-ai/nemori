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

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def emit(self, event: str, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            asyncio.create_task(handler(**kwargs))
