"""PostgreSQL implementation of EpisodeStore."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.db.connection import DatabaseManager
from src.domain.models import Episode

logger = logging.getLogger("nemori")


class PgEpisodeStore:
    """Episode persistence + search backed by PostgreSQL + pgvector."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, episode: Episode) -> None:
        await self._db.execute(
            """
            INSERT INTO episodes (id, user_id, title, content, embedding,
                                  source_messages, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                source_messages = EXCLUDED.source_messages,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            episode.id, episode.user_id, episode.title, episode.content,
            episode.embedding, json.dumps(episode.source_messages),
            json.dumps(episode.metadata), episode.created_at, episode.updated_at,
        )

    async def get(self, episode_id: str) -> Episode | None:
        row = await self._db.fetchrow(
            "SELECT * FROM episodes WHERE id = $1", episode_id
        )
        return self._row_to_episode(row) if row else None

    async def list_by_user(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT * FROM episodes WHERE user_id = $1
               ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
            user_id, limit, offset,
        )
        return [self._row_to_episode(r) for r in rows]

    async def delete(self, episode_id: str) -> None:
        await self._db.execute("DELETE FROM episodes WHERE id = $1", episode_id)

    async def delete_by_user(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM episodes WHERE user_id = $1", user_id)

    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT *, embedding <=> $2::vector AS distance
               FROM episodes
               WHERE user_id = $1 AND embedding IS NOT NULL
               ORDER BY distance ASC
               LIMIT $3""",
            user_id, str(embedding), top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $2)) AS rank
               FROM episodes
               WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
               ORDER BY rank DESC
               LIMIT $3""",
            user_id, query, top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """WITH vector_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $3::vector) AS vrank
                FROM episodes
                WHERE user_id = $1 AND embedding IS NOT NULL
                LIMIT $4 * 2
            ),
            text_results AS (
                SELECT id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank(tsv, plainto_tsquery('simple', $2)) DESC
                ) AS trank
                FROM episodes
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
            SELECT e.* FROM fused f JOIN episodes e ON f.id = e.id
            ORDER BY f.rrf_score DESC""",
            user_id, query, str(embedding), top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    @staticmethod
    def _row_to_episode(row: Any) -> Episode:
        source_msgs = row["source_messages"]
        if isinstance(source_msgs, str):
            source_msgs = json.loads(source_msgs)
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Episode(
            id=str(row["id"]),
            user_id=row["user_id"],
            title=row["title"],
            content=row["content"],
            embedding=list(row["embedding"]) if row.get("embedding") else None,
            source_messages=source_msgs or [],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
