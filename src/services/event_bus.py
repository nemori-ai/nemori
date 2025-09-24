"""Simple in-process event bus used to decouple pipeline stages."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

EventHandler = Callable[[str, Dict[str, Any]], None]


class EventBus:
    """Thread-safe publish/subscribe event bus."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[EventHandler]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            self._listeners[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            if event_name in self._listeners:
                self._listeners[event_name] = [h for h in self._listeners[event_name] if h != handler]
                if not self._listeners[event_name]:
                    self._listeners.pop(event_name)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners.get(event_name, ()))
        for handler in listeners:
            handler(event_name, payload)


__all__ = ["EventBus"]
