"""Qdrant vector store for episode and semantic memory embeddings."""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger("nemori")


class QdrantVectorStore:
    """Manages vector storage and search in Qdrant."""

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        collection_prefix: str = "nemori",
    ) -> None:
        self._client = QdrantClient(url=url, port=port, api_key=api_key, check_compatibility=False)
        self._prefix = collection_prefix
        self._episodes_collection = f"{collection_prefix}_episodes"
        self._semantic_collection = f"{collection_prefix}_semantic"

    def _ensure_collection(self, name: str, dimension: int) -> None:
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if name not in collections:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s (dim=%d)", name, dimension)

    def ensure_collections(self, dimension: int) -> None:
        """Ensure both episode and semantic collections exist."""
        self._ensure_collection(self._episodes_collection, dimension)
        self._ensure_collection(self._semantic_collection, dimension)

    # --- Episode vectors ---

    def upsert_episode(
        self, episode_id: str, user_id: str, agent_id: str, embedding: list[float]
    ) -> None:
        self._client.upsert(
            collection_name=self._episodes_collection,
            points=[
                PointStruct(
                    id=episode_id,
                    vector=embedding,
                    payload={"user_id": user_id, "agent_id": agent_id},
                )
            ],
        )

    def search_episodes(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Search similar episodes. Returns list of {id, score}."""
        results = self._client.query_points(
            collection_name=self._episodes_collection,
            query=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]
            ),
            limit=top_k,
        )
        return [{"id": str(r.id), "score": r.score} for r in results.points]

    def delete_episode(self, episode_id: str) -> None:
        self._client.delete(
            collection_name=self._episodes_collection,
            points_selector=[episode_id],
        )

    def delete_episodes_by_user(self, user_id: str, agent_id: str) -> None:
        self._client.delete(
            collection_name=self._episodes_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]
            ),
        )

    # --- Semantic vectors ---

    def upsert_semantic(
        self, memory_id: str, user_id: str, agent_id: str, embedding: list[float]
    ) -> None:
        self._client.upsert(
            collection_name=self._semantic_collection,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={"user_id": user_id, "agent_id": agent_id},
                )
            ],
        )

    def search_semantic(
        self, user_id: str, agent_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        results = self._client.query_points(
            collection_name=self._semantic_collection,
            query=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]
            ),
            limit=top_k,
        )
        return [{"id": str(r.id), "score": r.score} for r in results.points]

    def delete_semantic(self, memory_id: str) -> None:
        self._client.delete(
            collection_name=self._semantic_collection,
            points_selector=[memory_id],
        )

    def delete_semantic_by_user(self, user_id: str, agent_id: str) -> None:
        self._client.delete(
            collection_name=self._semantic_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]
            ),
        )

    def close(self) -> None:
        self._client.close()
