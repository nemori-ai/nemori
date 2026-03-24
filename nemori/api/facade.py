"""Async NemoriMemory public interface."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from nemori.config import MemoryConfig
from nemori.db.connection import DatabaseManager
from nemori.db.migrations import get_migrations
from nemori.db.qdrant_store import QdrantVectorStore
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
        self._qdrant: QdrantVectorStore | None = None

    async def __aenter__(self) -> NemoriMemory:
        self._db = DatabaseManager()
        await self._db.init(
            self._config.dsn,
            min_size=self._config.db_pool_min,
            max_size=self._config.db_pool_max,
        )

        # Probe embedding dimension (let it detect native dimension, no truncation)
        try:
            embedding = AsyncEmbeddingClient(
                api_key=self._config.embedding_api_key,
                model=self._config.embedding_model,
                base_url=self._config.embedding_base_url,
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

        # Run PostgreSQL migrations
        migrations = get_migrations(self._config.embedding_dimension)
        await self._db.ensure_schema(migrations)

        # Initialize Qdrant
        self._qdrant = QdrantVectorStore(
            url=self._config.qdrant_url,
            port=self._config.qdrant_port,
            api_key=self._config.qdrant_api_key,
            collection_prefix=self._config.qdrant_collection_prefix,
        )
        self._qdrant.ensure_collections(self._config.embedding_dimension)

        self._system = await self._build_system()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._system:
            await self._system.drain(timeout=30.0)
        if self._qdrant:
            self._qdrant.close()
        if self._db:
            await self._db.close()

    async def _build_system(self) -> MemorySystem:
        from nemori.factory import create_memory_system
        assert self._db is not None
        assert self._qdrant is not None
        return await create_memory_system(self._config, self._db, self._qdrant)

    def _ensure_system(self) -> MemorySystem:
        if self._system is None:
            raise RuntimeError("NemoriMemory not initialized. Use 'async with' context manager.")
        return self._system

    async def add_multimodal_message(
        self,
        user_id: str,
        text: str,
        image_urls: list[str] | None = None,
        role: str = "user",
        timestamp: str | None = None,
        compress_images: bool = True,
    ) -> None:
        """Add a message with optional image attachments.

        Args:
            user_id: User identifier.
            text: Text content of the message.
            image_urls: Optional list of image URLs (data URLs or http URLs).
            role: Message role (default: "user").
            timestamp: Optional ISO timestamp string.
            compress_images: Whether to compress images for LLM (default: True).
        """
        if image_urls:
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for url in image_urls:
                if compress_images:
                    from nemori.utils.image import compress_image_for_llm
                    url = compress_image_for_llm(url)
                content.append({"type": "image_url", "image_url": {"url": url}})
            msg_dict: dict[str, Any] = {"role": role, "content": content}
        else:
            msg_dict = {"role": role, "content": text}

        if timestamp:
            msg_dict["timestamp"] = timestamp

        await self.add_messages(user_id, [msg_dict])

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
