"""Async NemoriMemory public interface."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from nemori.config import MemoryConfig
from nemori.db.connection import DatabaseManager
from nemori.db.migrations import get_migrations
from nemori.domain.models import Message, Episode, HealthResult
from nemori.services.embedding import AsyncEmbeddingClient
from nemori.search.unified import SearchMethod, SearchResult
from nemori.core.memory_system import MemorySystem

logger = logging.getLogger("nemori")


class NemoriMemory:
    """High-level async interface for the Nemori memory system."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        self._db: DatabaseManager | None = None
        self._system: MemorySystem | None = None

    async def __aenter__(self) -> NemoriMemory:
        self._db = DatabaseManager()
        await self._db.init(
            self._config.dsn,
            min_size=self._config.db_pool_min,
            max_size=self._config.db_pool_max,
        )

        # Probe embedding dimension before schema setup
        try:
            embedding = AsyncEmbeddingClient(
                api_key=self._config.embedding_api_key,
                model=self._config.embedding_model,
                base_url=self._config.embedding_base_url,
                dimensions=self._config.embedding_dimension,
            )
            actual_dim = await embedding.probe_dimension()
            if actual_dim != self._config.embedding_dimension:
                logger.info(
                    "Embedding dimension probe: %d (config was %d), adjusting",
                    actual_dim, self._config.embedding_dimension,
                )
                self._config.embedding_dimension = actual_dim
        except Exception as e:
            logger.warning("Embedding dimension probe failed: %s. Using config value %d", e, self._config.embedding_dimension)

        # Run migrations with correct dimension
        migrations = get_migrations(self._config.embedding_dimension)
        await self._db.ensure_schema(migrations)

        # Check if dimension adaptation needed on existing DB
        await self._check_dimension_adaptation()

        self._system = await self._build_system()
        return self

    async def _check_dimension_adaptation(self) -> None:
        """Check if existing vector columns need dimension adaptation."""
        from nemori.db.migrations import get_dimension_adaptation_sql
        try:
            # Check current column dimension from pg_attribute
            row = await self._db.fetchrow("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'episodes'::regclass
                AND attname = 'embedding'
            """)
            if row and row['atttypmod'] > 0:
                current_dim = row['atttypmod']
                if current_dim != self._config.embedding_dimension:
                    logger.warning(
                        "Vector dimension mismatch: DB has %d, config needs %d. "
                        "Adapting schema and clearing stale embeddings.",
                        current_dim, self._config.embedding_dimension,
                    )
                    sql = get_dimension_adaptation_sql(self._config.embedding_dimension)
                    await self._db.execute(sql)
        except Exception as e:
            logger.debug("Dimension check skipped: %s", e)

    async def __aexit__(self, *exc: Any) -> None:
        if self._system:
            await self._system.drain(timeout=30.0)
        if self._db:
            await self._db.close()

    async def _build_system(self) -> MemorySystem:
        from nemori.factory import create_memory_system
        assert self._db is not None
        return await create_memory_system(self._config, self._db)

    def _ensure_system(self) -> MemorySystem:
        if self._system is None:
            raise RuntimeError("NemoriMemory not initialized. Use 'async with' context manager.")
        return self._system

    async def add_messages(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        system = self._ensure_system()
        msg_objects = []
        for m in messages:
            kwargs: dict[str, Any] = {"role": m["role"], "content": m["content"]}
            if "timestamp" in m:
                ts = m["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError as e:
                        raise ValueError(f"Invalid timestamp format in message: {ts!r}") from e
                elif not isinstance(ts, datetime):
                    raise TypeError(f"timestamp must be str or datetime, got {type(ts).__name__}")
                kwargs["timestamp"] = ts
            if "metadata" in m:
                kwargs["metadata"] = m["metadata"]
            msg_objects.append(Message(**kwargs))
        await system.add_messages(user_id, msg_objects)

    async def flush(self, user_id: str) -> list[dict[str, Any]]:
        system = self._ensure_system()
        episodes = await system.flush(user_id)
        return [e.to_dict() for e in episodes]

    async def search(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int | None = None,
        top_k_semantic: int | None = None,
        search_method: str = "hybrid",
    ) -> dict[str, Any]:
        system = self._ensure_system()
        method_map = {
            "vector": SearchMethod.VECTOR,
            "text": SearchMethod.TEXT,
            "bm25": SearchMethod.TEXT,  # backward compat
            "hybrid": SearchMethod.HYBRID,
        }
        method = method_map.get(search_method, SearchMethod.HYBRID)
        result = await system.search(
            user_id, query,
            top_k_episodes=top_k_episodes,
            top_k_semantic=top_k_semantic,
            method=method,
        )
        return result.to_dict()

    async def delete_episode(self, user_id: str, episode_id: str) -> None:
        system = self._ensure_system()
        await system.delete_episode(user_id, episode_id)

    async def delete_semantic(self, user_id: str, memory_id: str) -> None:
        system = self._ensure_system()
        await system.delete_semantic(user_id, memory_id)

    async def delete_user(self, user_id: str) -> None:
        system = self._ensure_system()
        await system.delete_user(user_id)

    async def stats(self, user_id: str) -> dict[str, Any]:
        system = self._ensure_system()
        return await system.stats(user_id)

    async def health(self) -> HealthResult:
        db_ok = False
        llm_ok = False
        embed_ok = False
        diagnostics: dict[str, Any] = {}

        if self._db:
            try:
                db_ok = await self._db.ping()
                if self._db.pool:
                    diagnostics["pool_size"] = self._db.pool.get_size()
                    diagnostics["pool_free"] = self._db.pool.get_idle_size()
            except Exception:
                pass

        return HealthResult(
            db=db_ok,
            llm=llm_ok,
            embedding=embed_ok,
            diagnostics=diagnostics,
        )
