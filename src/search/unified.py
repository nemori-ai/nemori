"""Unified search across episode and semantic stores."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.domain.models import Episode, SemanticMemory
from src.domain.interfaces import EpisodeStore, SemanticStore, EmbeddingProvider

logger = logging.getLogger("nemori")


class SearchMethod(Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    episodes: list[Episode] = field(default_factory=list)
    semantic_memories: list[SemanticMemory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "semantic_memories": [s.to_dict() for s in self.semantic_memories],
        }


class UnifiedSearch:
    """Delegates search to the appropriate store methods."""

    def __init__(
        self,
        episode_store: EpisodeStore,
        semantic_store: SemanticStore,
        embedding: EmbeddingProvider,
    ) -> None:
        self._episodes = episode_store
        self._semantics = semantic_store
        self._embedding = embedding

    async def search(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int = 10,
        top_k_semantic: int = 10,
        method: SearchMethod = SearchMethod.HYBRID,
    ) -> SearchResult:
        embedding = await self._embedding.embed(query)

        if method == SearchMethod.VECTOR:
            ep_task = self._episodes.search_by_vector(user_id, embedding, top_k_episodes)
            sm_task = self._semantics.search_by_vector(user_id, embedding, top_k_semantic)
        elif method == SearchMethod.TEXT:
            ep_task = self._episodes.search_by_text(user_id, query, top_k_episodes)
            sm_task = self._semantics.search_by_text(user_id, query, top_k_semantic)
        else:
            ep_task = self._episodes.search_hybrid(user_id, query, embedding, top_k_episodes)
            sm_task = self._semantics.search_hybrid(user_id, query, embedding, top_k_semantic)

        episodes, semantics = await asyncio.gather(ep_task, sm_task)
        return SearchResult(episodes=episodes, semantic_memories=semantics)
