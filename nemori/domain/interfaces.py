# src/domain/interfaces.py
"""Domain protocols for the Nemori memory system."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from nemori.domain.models import Episode, SemanticMemory, Message


@runtime_checkable
class EpisodeStore(Protocol):
    """Unified episode persistence + search."""

    async def save(self, episode: Episode) -> None: ...
    async def get(self, episode_id: str, user_id: str, agent_id: str) -> Episode | None: ...
    async def list_by_user(
        self, user_id: str, agent_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]: ...
    async def delete(self, episode_id: str, user_id: str, agent_id: str) -> None: ...
    async def delete_by_user(self, user_id: str, agent_id: str) -> None: ...
    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Episode]: ...
    async def get_batch(
        self, episode_ids: list[str], user_id: str, agent_id: str
    ) -> list[Episode]: ...


@runtime_checkable
class SemanticStore(Protocol):
    """Unified semantic memory persistence + search."""

    async def save(self, memory: SemanticMemory) -> None: ...
    async def save_batch(self, memories: list[SemanticMemory]) -> None: ...
    async def get(self, memory_id: str, user_id: str, agent_id: str) -> SemanticMemory | None: ...
    async def list_by_user(
        self, user_id: str, agent_id: str, memory_type: str | None = None
    ) -> list[SemanticMemory]: ...
    async def delete(self, memory_id: str, user_id: str, agent_id: str) -> None: ...
    async def delete_by_user(self, user_id: str, agent_id: str) -> None: ...
    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]: ...
    async def get_batch(
        self, memory_ids: list[str], user_id: str, agent_id: str
    ) -> list[SemanticMemory]: ...


@runtime_checkable
class MessageBufferStore(Protocol):
    """Persistent message buffer."""

    async def push(self, user_id: str, agent_id: str, messages: list[Message]) -> None: ...
    async def get_unprocessed(self, user_id: str, agent_id: str) -> list[Message]: ...
    async def mark_processed(self, user_id: str, agent_id: str, message_ids: list[int]) -> None: ...
    async def count_unprocessed(self, user_id: str, agent_id: str) -> int: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding generation protocol."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """LLM call protocol."""

    async def complete(self, messages: list[dict], **kwargs: object) -> str: ...

    async def complete_with_usage(
        self, messages: list[dict], **kwargs: object
    ) -> tuple[str, dict[str, int]]:
        """Return (content, {"prompt_tokens": ..., "completion_tokens": ...})."""
        ...
