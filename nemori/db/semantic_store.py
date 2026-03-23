"""PostgreSQL implementation of SemanticStore."""
from __future__ import annotations

import json
import logging
from typing import Any

from nemori.db.connection import DatabaseManager
from nemori.domain.models import SemanticMemory

logger = logging.getLogger("nemori")


class PgSemanticStore:
    """Semantic memory persistence + search backed by PostgreSQL + pgvector."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, memory: SemanticMemory) -> None:
        await self._db.execute(
            """
            INSERT INTO semantic_memories
                (id, user_id, content, memory_type, embedding,
                 source_episode_id, confidence, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                memory_type = EXCLUDED.memory_type,
                embedding = EXCLUDED.embedding,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            memory.id, memory.user_id, memory.content, memory.memory_type,
            memory.embedding, memory.source_episode_id, memory.confidence,
            json.dumps(memory.metadata), memory.created_at, memory.updated_at,
        )

    async def save_batch(self, memories: list[SemanticMemory]) -> None:
        for memory in memories:
            await self.save(memory)

    async def get(self, memory_id: str) -> SemanticMemory | None:
        row = await self._db.fetchrow(
            "SELECT * FROM semantic_memories WHERE id = $1", memory_id
        )
        return self._row_to_memory(row) if row else None

    async def list_by_user(
        self, user_id: str, memory_type: str | None = None
    ) -> list[SemanticMemory]:
        if memory_type:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 AND memory_type = $2
                   ORDER BY created_at DESC""",
                user_id, memory_type,
            )
        else:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 ORDER BY created_at DESC""",
                user_id,
            )
        return [self._row_to_memory(r) for r in rows]

    async def delete(self, memory_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE id = $1", memory_id
        )

    async def delete_by_user(self, user_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE user_id = $1", user_id
        )

    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """SELECT *, embedding <=> $2::vector AS distance
               FROM semantic_memories
               WHERE user_id = $1 AND embedding IS NOT NULL
               ORDER BY distance ASC
               LIMIT $3""",
            user_id, str(embedding), top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $2)) AS rank
               FROM semantic_memories
               WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
               ORDER BY rank DESC
               LIMIT $3""",
            user_id, query, top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """WITH vector_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $3::vector) AS vrank
                FROM semantic_memories
                WHERE user_id = $1 AND embedding IS NOT NULL
                LIMIT $4 * 2
            ),
            text_results AS (
                SELECT id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank(tsv, plainto_tsquery('simple', $2)) DESC
                ) AS trank
                FROM semantic_memories
                WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
                LIMIT $4 * 2
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(1.0 / (60 + v.vrank), 0) +
                       COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
                FROM vector_results v
                FULL OUTER JOIN text_results t ON v.id = t.id
                ORDER BY rrf_score DESC
                LIMIT $4
            )
            SELECT sm.* FROM fused f
            JOIN semantic_memories sm ON f.id = sm.id
            ORDER BY f.rrf_score DESC""",
            user_id, query, str(embedding), top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    @staticmethod
    def _row_to_memory(row: Any) -> SemanticMemory:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return SemanticMemory(
            id=str(row["id"]),
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            embedding=list(row["embedding"]) if row.get("embedding") else None,
            source_episode_id=str(row["source_episode_id"]) if row.get("source_episode_id") else None,
            confidence=row["confidence"],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
