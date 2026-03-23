"""asyncpg connection pool lifecycle management."""
from __future__ import annotations

import logging
from typing import Any

import asyncpg

from nemori.domain.exceptions import DatabaseError

logger = logging.getLogger("nemori")


class DatabaseManager:
    """Manages asyncpg connection pool lifecycle."""

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def init(
        self,
        dsn: str,
        min_size: int = 5,
        max_size: int = 20,
    ) -> None:
        try:
            self.pool = await asyncpg.create_pool(
                dsn, min_size=min_size, max_size=max_size
            )
            logger.info("Database pool created (min=%d, max=%d)", min_size, max_size)
        except Exception as e:
            raise DatabaseError(f"Failed to create connection pool: {e}") from e

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
            self.pool = None

    def _ensure_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise DatabaseError("DatabaseManager not initialized. Call init() first.")
        return self.pool

    async def execute(self, query: str, *args: Any) -> str:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def executemany(self, query: str, args: list[tuple]) -> None:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(query, args)

    async def ping(self) -> bool:
        try:
            pool = self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except DatabaseError:
            raise
        except Exception:
            return False

    async def ensure_schema(self, migrations_sql: list[tuple[int, str, str]]) -> None:
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
            applied = {r["version"] for r in rows}

            for version, name, sql in sorted(migrations_sql):
                if version not in applied:
                    logger.info("Applying migration %d: %s", version, name)
                    async with conn.transaction():
                        await conn.execute(sql)
                        await conn.execute(
                            "INSERT INTO schema_migrations (version) VALUES ($1)",
                            version,
                        )
