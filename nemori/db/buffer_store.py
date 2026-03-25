"""PostgreSQL implementation of MessageBufferStore."""
from __future__ import annotations

import json
import logging

from nemori.db.connection import DatabaseManager
from nemori.domain.models import Message

logger = logging.getLogger("nemori")


class PgMessageBufferStore:
    """Persistent message buffer backed by PostgreSQL."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def push(self, user_id: str, agent_id: str, messages: list[Message]) -> None:
        for msg in messages:
            await self._db.execute(
                """INSERT INTO message_buffer (user_id, agent_id, role, content, timestamp)
                   VALUES ($1, $2, $3, $4::jsonb, $5)""",
                user_id, agent_id, msg.role, json.dumps(msg.content), msg.timestamp,
            )

    async def get_unprocessed(self, user_id: str, agent_id: str) -> list[Message]:
        rows = await self._db.fetch(
            """SELECT id, role, content, timestamp
               FROM message_buffer
               WHERE user_id = $1 AND agent_id = $2 AND NOT processed
               ORDER BY timestamp ASC""",
            user_id, agent_id,
        )
        result = []
        for row in rows:
            content = row["content"]
            # JSONB returns Python objects directly via asyncpg
            # If it's a plain string (old data), use as-is
            # If it's a list (multimodal), use as-is
            msg = Message(
                role=row["role"],
                content=content,
                timestamp=row["timestamp"],
                metadata={"buffer_id": row["id"]},
            )
            result.append(msg)
        return result

    async def mark_processed(self, user_id: str, agent_id: str, message_ids: list[int]) -> None:
        if not message_ids:
            return
        await self._db.execute(
            """DELETE FROM message_buffer
               WHERE user_id = $1 AND agent_id = $2 AND id = ANY($3)""",
            user_id, agent_id, message_ids,
        )

    async def count_unprocessed(self, user_id: str, agent_id: str) -> int:
        count = await self._db.fetchval(
            """SELECT COUNT(*) FROM message_buffer
               WHERE user_id = $1 AND agent_id = $2 AND NOT processed""",
            user_id, agent_id,
        )
        return count or 0
