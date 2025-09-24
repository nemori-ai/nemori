"""Cache services extracting MemorySystem's ad-hoc caching into dedicated classes."""

from __future__ import annotations

import threading
import time
from typing import Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class PerUserCache(Generic[T]):
    """Thread-safe per-user cache with TTL semantics."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, Dict[str, any]] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> Optional[T]:
        with self._lock:
            entry = self._store.get(user_id)
            if not entry:
                return None
            if time.time() - entry["timestamp"] > self._ttl:
                self._store.pop(user_id, None)
                return None
            return entry["value"]  # type: ignore[return-value]

    def put(self, user_id: str, value: T) -> None:
        with self._lock:
            self._store[user_id] = {"timestamp": time.time(), "value": value}

    def invalidate(self, user_id: str) -> None:
        with self._lock:
            self._store.pop(user_id, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class SemanticEmbeddingCache:
    """Stores embeddings per semantic memory to avoid repeated computations."""

    def __init__(self) -> None:
        self._embeddings: Dict[str, Dict[str, List[float]]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._manager_lock = threading.Lock()

    def get_user_lock(self, user_id: str) -> threading.Lock:
        with self._manager_lock:
            if user_id not in self._locks:
                self._locks[user_id] = threading.Lock()
            return self._locks[user_id]

    def get(self, user_id: str, memory_id: str) -> Optional[List[float]]:
        lock = self.get_user_lock(user_id)
        with lock:
            return self._embeddings.get(user_id, {}).get(memory_id)

    def set(self, user_id: str, memory_id: str, embedding: List[float]) -> None:
        lock = self.get_user_lock(user_id)
        with lock:
            self._embeddings.setdefault(user_id, {})[memory_id] = embedding

    def list_user_embeddings(self, user_id: str) -> Dict[str, List[float]]:
        lock = self.get_user_lock(user_id)
        with lock:
            return dict(self._embeddings.get(user_id, {}))

    def invalidate_user(self, user_id: str) -> None:
        lock = self.get_user_lock(user_id)
        with lock:
            self._embeddings.pop(user_id, None)


__all__ = ["PerUserCache", "SemanticEmbeddingCache"]
