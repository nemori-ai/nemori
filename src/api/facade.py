"""High-level facade providing a simplified memory interface."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from ..config import MemoryConfig
from ..core.memory_system import MemorySystem
from ..utils import LLMClient, EmbeddingClient


class NemoriMemory:
    """Minimal public API for working with the Nemori memory system.

    Designed for "one-line" usage while still allowing custom dependency
    injection for tests and advanced scenarios.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        memory_system: Optional[MemorySystem] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self._memory_system = memory_system or MemorySystem(
            config=self.config,
            language=self.config.language,
            llm_client=llm_client,
            embedding_client=embedding_client,
        )

    # ------------------------------------------------------------------
    # Basic lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release all resources held by the underlying memory system."""
        self._memory_system.__exit__(None, None, None)

    def __enter__(self) -> "NemoriMemory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------
    def add_messages(self, user_id: str, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Append messages to the memory buffer for a given user."""
        return self._memory_system.add_messages(user_id, list(messages))

    def flush(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Force creation of an episode from the current buffer."""
        return self._memory_system.force_episode_creation(user_id)

    def wait_for_semantic(self, user_id: str, timeout: float = 30.0) -> bool:
        """Block until all semantic memory tasks for the user complete."""
        return self._memory_system.wait_for_semantic_generation(user_id, timeout=timeout)

    def search(
        self,
        user_id: str,
        query: str,
        *,
        top_k_episodes: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search episodic + semantic memory in a single call."""
        return self._memory_system.search_all(
            user_id=user_id,
            query=query,
            top_k_episodes=top_k_episodes,
            top_k_semantic=top_k_semantic,
            search_method=search_method,
        )

    def stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Expose system statistics for diagnostics."""
        return self._memory_system.get_stats(user_id)

    async def asearch(
        self,
        user_id: str,
        query: str,
        *,
        top_k_episodes: Optional[int] = None,
        top_k_semantic: Optional[int] = None,
        search_method: str = "hybrid",
    ) -> Dict[str, List[Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                user_id,
                query,
                top_k_episodes=top_k_episodes,
                top_k_semantic=top_k_semantic,
                search_method=search_method,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, **kwargs: Any) -> "NemoriMemory":
        """Create an instance using configuration sourced from env vars."""
        return cls(MemoryConfig(), **kwargs)


__all__ = ["NemoriMemory"]
