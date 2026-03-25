"""PostgreSQL implementation of EpisodeStore."""
from __future__ import annotations

import json
import logging
from typing import Any

from nemori.db.connection import DatabaseManager
from nemori.domain.models import Episode

logger = logging.getLogger("nemori")


class PgEpisodeStore:
    """Episode persistence + text search backed by PostgreSQL."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, episode: Episode) -> None:
        await self._db.execute(
            """
            INSERT INTO episodes (id, user_id, agent_id, title, content,
                                  source_messages, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                source_messages = EXCLUDED.source_messages,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            episode.id, episode.user_id, episode.agent_id, episode.title,
            episode.content,
            json.dumps(episode.source_messages),
            json.dumps(episode.metadata), episode.created_at, episode.updated_at,
        )

    async def get(self, episode_id: str, user_id: str, agent_id: str) -> Episode | None:
        row = await self._db.fetchrow(
            "SELECT * FROM episodes WHERE id = $1 AND user_id = $2 AND agent_id = $3",
            episode_id, user_id, agent_id,
        )
        return self._row_to_episode(row) if row else None

    async def list_by_user(
        self, user_id: str, agent_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT * FROM episodes WHERE user_id = $1 AND agent_id = $2
               ORDER BY created_at DESC LIMIT $3 OFFSET $4""",
            user_id, agent_id, limit, offset,
        )
        return [self._row_to_episode(r) for r in rows]

    async def delete(self, episode_id: str, user_id: str, agent_id: str) -> None:
        await self._db.execute(
            "DELETE FROM episodes WHERE id = $1 AND user_id = $2 AND agent_id = $3",
            episode_id, user_id, agent_id,
        )

    async def delete_by_user(self, user_id: str, agent_id: str) -> None:
        await self._db.execute(
            "DELETE FROM episodes WHERE user_id = $1 AND agent_id = $2",
            user_id, agent_id,
        )

    async def search_by_text(
        self, user_id: str, agent_id: str, query: str, top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $3)) AS rank
               FROM episodes
               WHERE user_id = $1 AND agent_id = $2 AND tsv @@ plainto_tsquery('simple', $3)
               ORDER BY rank DESC
               LIMIT $4""",
            user_id, agent_id, query, top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    async def get_batch(self, episode_ids: list[str], user_id: str, agent_id: str) -> list[Episode]:
        """Fetch multiple episodes by IDs."""
        if not episode_ids:
            return []
        rows = await self._db.fetch(
            """SELECT * FROM episodes
               WHERE id = ANY($1::uuid[]) AND user_id = $2 AND agent_id = $3""",
            episode_ids, user_id, agent_id,
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
            agent_id=row.get("agent_id", "default"),
            title=row["title"],
            content=row["content"],
            embedding=None,
            source_messages=source_msgs or [],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
