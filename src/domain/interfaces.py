"""Domain-level interface definitions for the Nemori memory system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Protocol

from ..models import Episode, SemanticMemory, MessageBuffer, Message


class EpisodeRepository(ABC):
    """Storage abstraction for episodic memories."""

    @abstractmethod
    def save(self, episode: Episode) -> str: ...

    @abstractmethod
    def list_by_user(self, user_id: str) -> List[Episode]: ...

    @abstractmethod
    def delete_user_data(self, user_id: str) -> bool: ...


class SemanticRepository(ABC):
    """Storage abstraction for semantic memories."""

    @abstractmethod
    def save(self, memory: SemanticMemory) -> str: ...

    @abstractmethod
    def list_by_user(self, user_id: str) -> List[SemanticMemory]: ...

    @abstractmethod
    def delete_user_data(self, user_id: str) -> bool: ...


class VectorIndex(ABC):
    """Vector index abstraction for episodic/semantic retrieval."""

    @abstractmethod
    def add_episode(self, user_id: str, episode: Episode, embedding: Optional[List[float]] = None) -> None: ...

    @abstractmethod
    def add_semantic(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None) -> None: ...

    @abstractmethod
    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...

    @abstractmethod
    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...

    @abstractmethod
    def clear(self, user_id: str) -> bool: ...


class LexicalIndex(ABC):
    """Traditional lexical index (e.g., BM25)."""

    @abstractmethod
    def add_episode(self, user_id: str, episode: Episode) -> None: ...

    @abstractmethod
    def add_semantic(self, user_id: str, memory: SemanticMemory) -> None: ...

    @abstractmethod
    def search_episodes(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...

    @abstractmethod
    def search_semantics(self, user_id: str, query: str, top_k: int) -> List[Dict]: ...

    @abstractmethod
    def clear(self, user_id: str) -> bool: ...


class BoundaryDetector(ABC):
    @abstractmethod
    def detect(self, buffer: MessageBuffer, new_messages: Iterable[Message]) -> Dict[str, any]: ...


class EpisodeGenerator(ABC):
    @abstractmethod
    def generate(self, user_id: str, messages: List[Message], boundary_reason: str) -> Episode: ...


class SemanticGenerator(ABC):
    @abstractmethod
    def generate(self, user_id: str, episode: Episode, existing_episodes: List[Episode], existing_semantics: List[SemanticMemory]) -> List[SemanticMemory]: ...


class DedupStrategy(Protocol):
    def is_duplicate(self, candidate: SemanticMemory, existing_semantics: List[SemanticMemory]) -> bool: ...
