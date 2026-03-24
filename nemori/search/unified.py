"""Unified search across episode and semantic stores."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nemori.domain.models import Episode, SemanticMemory
from nemori.domain.interfaces import EmbeddingProvider
from nemori.db.qdrant_store import QdrantVectorStore

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
    """Delegates search: Qdrant for vector, PostgreSQL for text, RRF for hybrid."""

    def __init__(
        self,
        episode_store: Any,
        semantic_store: Any,
        embedding: EmbeddingProvider,
        qdrant: QdrantVectorStore,
    ) -> None:
        self._episodes = episode_store
        self._semantics = semantic_store
        self._embedding = embedding
        self._qdrant = qdrant

    async def _vector_search_episodes(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        """Vector search via Qdrant, then fetch full records from PostgreSQL."""
        results = self._qdrant.search_episodes(user_id, agent_id, embedding, top_k)
        if not results:
            return []
        ids = [r["id"] for r in results]
        return await self._episodes.get_batch(ids, user_id, agent_id)

    async def _vector_search_semantic(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        """Vector search via Qdrant, then fetch full records from PostgreSQL."""
        results = self._qdrant.search_semantic(user_id, agent_id, embedding, top_k)
        if not results:
            return []
        ids = [r["id"] for r in results]
        return await self._semantics.get_batch(ids, user_id, agent_id)

    async def _hybrid_search_episodes(
        self, user_id: str, agent_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        """RRF fusion of Qdrant vector results and PostgreSQL text results."""
        qdrant_results = self._qdrant.search_episodes(user_id, agent_id, embedding, top_k * 2)
        text_results = await self._episodes.search_by_text(user_id, agent_id, query, top_k * 2)

        # Build RRF scores
        rrf: dict[str, float] = {}
        for rank, r in enumerate(qdrant_results, 1):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0 / (60 + rank)
        for rank, ep in enumerate(text_results, 1):
            rrf[ep.id] = rrf.get(ep.id, 0) + 1.0 / (60 + rank)

        # Sort by RRF score, take top_k
        sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
        if not sorted_ids:
            return []

        # Fetch full records
        episodes = await self._episodes.get_batch(sorted_ids, user_id, agent_id)
        # Re-sort by RRF score
        id_to_ep = {ep.id: ep for ep in episodes}
        return [id_to_ep[eid] for eid in sorted_ids if eid in id_to_ep]

    async def _hybrid_search_semantic(
        self, user_id: str, agent_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        """RRF fusion of Qdrant vector results and PostgreSQL text results."""
        qdrant_results = self._qdrant.search_semantic(user_id, agent_id, embedding, top_k * 2)
        text_results = await self._semantics.search_by_text(user_id, agent_id, query, top_k * 2)

        rrf: dict[str, float] = {}
        for rank, r in enumerate(qdrant_results, 1):
            rrf[r["id"]] = rrf.get(r["id"], 0) + 1.0 / (60 + rank)
        for rank, sm in enumerate(text_results, 1):
            rrf[sm.id] = rrf.get(sm.id, 0) + 1.0 / (60 + rank)

        sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
        if not sorted_ids:
            return []

        memories = await self._semantics.get_batch(sorted_ids, user_id, agent_id)
        id_to_mem = {m.id: m for m in memories}
        return [id_to_mem[mid] for mid in sorted_ids if mid in id_to_mem]

    async def search(
        self,
        user_id: str,
        agent_id: str,
        query: str,
        top_k_episodes: int = 10,
        top_k_semantic: int = 10,
        method: SearchMethod = SearchMethod.HYBRID,
    ) -> SearchResult:
        embedding = await self._embedding.embed(query)

        if method == SearchMethod.VECTOR:
            ep_task = self._vector_search_episodes(user_id, agent_id, embedding, top_k_episodes)
            sm_task = self._vector_search_semantic(user_id, agent_id, embedding, top_k_semantic)
        elif method == SearchMethod.TEXT:
            ep_task = self._episodes.search_by_text(user_id, agent_id, query, top_k_episodes)
            sm_task = self._semantics.search_by_text(user_id, agent_id, query, top_k_semantic)
        else:
            ep_task = self._hybrid_search_episodes(user_id, agent_id, query, embedding, top_k_episodes)
            sm_task = self._hybrid_search_semantic(user_id, agent_id, query, embedding, top_k_semantic)

        episodes, semantics = await asyncio.gather(ep_task, sm_task)
        return SearchResult(episodes=episodes, semantic_memories=semantics)
