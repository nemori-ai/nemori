"""Async NemoriMemory public interface."""
from __future__ import annotations

import logging
from typing import Any

from src.config import MemoryConfig
from src.db.connection import DatabaseManager
from src.db.migrations import get_migrations
from src.db.episode_store import PgEpisodeStore
from src.db.semantic_store import PgSemanticStore
from src.db.buffer_store import PgMessageBufferStore
from src.domain.models import Message, Episode, HealthResult
from src.llm.orchestrator import LLMOrchestrator
from src.llm.client import AsyncLLMClient
from src.llm.generators.episode import EpisodeGenerator
from src.llm.generators.semantic import SemanticGenerator
from src.services.embedding import AsyncEmbeddingClient
from src.services.event_bus import EventBus
from src.search.unified import UnifiedSearch, SearchMethod, SearchResult
from src.core.memory_system import MemorySystem

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
        migrations = get_migrations(self._config.embedding_dimension)
        await self._db.ensure_schema(migrations)
        self._system = await self._build_system()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._system:
            await self._system.drain(timeout=30.0)
        if self._db:
            await self._db.close()

    async def _build_system(self) -> MemorySystem:
        assert self._db is not None

        episode_store = PgEpisodeStore(self._db)
        semantic_store = PgSemanticStore(self._db)
        buffer_store = PgMessageBufferStore(self._db)

        llm_client = AsyncLLMClient(
            api_key=self._config.llm_api_key,
            base_url=self._config.llm_base_url,
        )
        orchestrator = LLMOrchestrator(
            provider=llm_client,
            default_model=self._config.llm_model,
            max_concurrent=self._config.llm_max_concurrent,
            token_budget=self._config.llm_token_budget,
        )
        embedding = AsyncEmbeddingClient(
            api_key=self._config.embedding_api_key,
            model=self._config.embedding_model,
            base_url=self._config.embedding_base_url,
        )
        episode_gen = EpisodeGenerator(orchestrator=orchestrator, embedding=embedding)
        semantic_gen = SemanticGenerator(
            orchestrator=orchestrator,
            embedding=embedding,
            enable_prediction_correction=self._config.enable_prediction_correction,
        )
        event_bus = EventBus()
        search = UnifiedSearch(episode_store, semantic_store, embedding)

        return MemorySystem(
            config=self._config,
            db=self._db,
            episode_store=episode_store,
            semantic_store=semantic_store,
            buffer_store=buffer_store,
            orchestrator=orchestrator,
            embedding=embedding,
            episode_generator=episode_gen,
            semantic_generator=semantic_gen,
            event_bus=event_bus,
            search=search,
        )

    def _ensure_system(self) -> MemorySystem:
        if self._system is None:
            raise RuntimeError("NemoriMemory not initialized. Use 'async with' context manager.")
        return self._system

    async def add_messages(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        system = self._ensure_system()
        msg_objects = [
            Message(role=m["role"], content=m["content"]) for m in messages
        ]
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
