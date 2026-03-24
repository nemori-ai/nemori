"""PostgreSQL implementation of SemanticStore."""
from __future__ import annotations

import json
import logging
from typing import Any

from nemori.db.connection import DatabaseManager
from nemori.domain.models import SemanticMemory

logger = logging.getLogger("nemori")


class PgSemanticStore:
    """Semantic memory persistence + text search backed by PostgreSQL."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, memory: SemanticMemory) -> None:
        await self._db.execute(
            """
            INSERT INTO semantic_memories
                (id, user_id, agent_id, content, memory_type,
                 source_episode_id, confidence, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                memory_type = EXCLUDED.memory_type,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            memory.id, memory.user_id, memory.agent_id, memory.content,
            memory.memory_type,
            memory.source_episode_id,
            memory.confidence, json.dumps(memory.metadata),
            memory.created_at, memory.updated_at,
        )

    async def save_batch(self, memories: list[SemanticMemory]) -> None:
        for memory in memories:
            await self.save(memory)

    async def get(self, memory_id: str, user_id: str, agent_id: str) -> SemanticMemory | None:
        row = await self._db.fetchrow(
            "SELECT * FROM semantic_memories WHERE id = $1 AND user_id = $2 AND agent_id = $3",
            memory_id, user_id, agent_id,
        )
        return self._row_to_memory(row) if row else None

    async def list_by_user(
        self, user_id: str, agent_id: str, memory_type: str | None = None
    ) -> list[SemanticMemory]:
        if memory_type:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 AND agent_id = $2 AND memory_type = $3
                   ORDER BY created_at DESC""",
                user_id, agent_id, memory_type,
            )
        else:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 AND agent_id = $2 ORDER BY created_at DESC""",
                user_id, agent_id,
            )
        return [self._row_to_memory(r) for r in rows]

    async def delete(self, memory_id: str, user_id: str, agent_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE id = $1 AND user_id = $2 AND agent_id = $3",
            memory_id, user_id, agent_id,
        )

    async def delete_by_user(self, user_id: str, agent_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE user_id = $1 AND agent_id = $2",
            user_id, agent_id,
        )

    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $3)) AS rank
               FROM semantic_memories
               WHERE user_id = $1 AND agent_id = $2 AND tsv @@ plainto_tsquery('simple', $3)
               ORDER BY rank DESC
               LIMIT $4""",
            user_id, agent_id, query, top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    async def get_batch(self, memory_ids: list[str], user_id: str, agent_id: str) -> list[SemanticMemory]:
        """Fetch multiple semantic memories by IDs."""
        if not memory_ids:
            return []
        rows = await self._db.fetch(
            """SELECT * FROM semantic_memories
               WHERE id = ANY($1::uuid[]) AND user_id = $2 AND agent_id = $3""",
            memory_ids, user_id, agent_id,
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
            agent_id=row.get("agent_id", "default"),
            content=row["content"],
            memory_type=row["memory_type"],
            embedding=None,
            source_episode_id=str(row["source_episode_id"]) if row.get("source_episode_id") else None,
            confidence=row["confidence"],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
