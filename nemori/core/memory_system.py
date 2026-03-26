"""Core memory system orchestrator (async)."""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any

from nemori.config import MemoryConfig
from nemori.db.connection import DatabaseManager
from nemori.db.qdrant_store import QdrantVectorStore
from nemori.domain.interfaces import EpisodeStore, SemanticStore, MessageBufferStore, EmbeddingProvider
from nemori.domain.models import Message, Episode, SemanticMemory
from nemori.llm.orchestrator import LLMOrchestrator
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.semantic import SemanticGenerator
from nemori.llm.generators.segmenter import BatchSegmenter
from nemori.llm.generators.merger import EpisodeMerger
from nemori.search.unified import UnifiedSearch, SearchMethod, SearchResult
from nemori.services.event_bus import EventBus

logger = logging.getLogger("nemori")

_MAX_LOCKS = 10_000


class MemorySystem:
    """Core async orchestrator for the Nemori memory system."""

    def __init__(
        self,
        config: MemoryConfig,
        agent_id: str,
        db: DatabaseManager,
        episode_store: EpisodeStore,
        semantic_store: SemanticStore,
        buffer_store: MessageBufferStore,
        orchestrator: LLMOrchestrator,
        embedding: EmbeddingProvider,
        episode_generator: EpisodeGenerator,
        semantic_generator: SemanticGenerator,
        event_bus: EventBus,
        search: UnifiedSearch,
        merger: EpisodeMerger | None = None,
        qdrant: QdrantVectorStore | None = None,
    ) -> None:
        self._config = config
        self._agent_id = agent_id
        self._db = db
        self._episode_store = episode_store
        self._semantic_store = semantic_store
        self._buffer_store = buffer_store
        self._orchestrator = orchestrator
        self._embedding = embedding
        self._episode_gen = episode_generator
        self._semantic_gen = semantic_generator
        self._event_bus = event_bus
        self._search = search
        self._merger = merger
        self._qdrant = qdrant

        self._user_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()

        self._enable_semantic = config.enable_semantic_memory

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        key = f"{self._agent_id}:{user_id}"
        if key not in self._user_locks:
            if len(self._user_locks) >= _MAX_LOCKS:
                self._user_locks.popitem(last=False)
            self._user_locks[key] = asyncio.Lock()
        self._user_locks.move_to_end(key)
        return self._user_locks[key]

    async def add_messages(self, user_id: str, messages: list[Message]) -> None:
        """Add messages to the buffer. Triggers processing if buffer is ready."""
        await self._buffer_store.push(user_id, self._agent_id, messages)
        count = await self._buffer_store.count_unprocessed(user_id, self._agent_id)
        if count >= self._config.buffer_size_min:
            task = asyncio.create_task(self._process(user_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def flush(self, user_id: str) -> list[Episode]:
        """Force processing of all buffered messages."""
        return await self._process(user_id)

    async def _process(self, user_id: str) -> list[Episode]:
        """Process buffered messages into episodes."""
        async with self._get_lock(user_id):
            messages = await self._buffer_store.get_unprocessed(user_id, self._agent_id)
            if not messages:
                return []

            # Extract buffer IDs for cleanup
            buffer_ids = [
                m.metadata.get("buffer_id")
                for m in messages
                if m.metadata.get("buffer_id") is not None
            ]

            episodes = []

            # Segment if batch is large enough
            if (
                self._config.enable_batch_segmentation
                and len(messages) >= self._config.batch_threshold
            ):
                segmenter = BatchSegmenter(orchestrator=self._orchestrator)
                groups = await segmenter.segment(messages)
            else:
                groups = [{"messages": messages, "topic": "conversation"}]

            # Generate episodes for each group
            for group in groups:
                group_msgs = group["messages"]
                if len(group_msgs) < self._config.episode_min_messages:
                    continue

                episode = await self._episode_gen.generate(
                    user_id, self._agent_id, group_msgs, group.get("topic", "conversation")
                )
                await self._episode_store.save(episode)

                # Upsert episode vector to Qdrant
                if self._qdrant and episode.embedding:
                    self._qdrant.upsert_episode(
                        episode.id, user_id, self._agent_id, episode.embedding
                    )

                # Check for merge
                if self._merger:
                    merged, merged_ep, old_id = await self._merger.check_and_merge(episode, self._agent_id)
                    if merged and merged_ep and old_id:
                        # Delete old target episode from PG and Qdrant
                        await self._episode_store.delete(old_id, user_id, self._agent_id)
                        if self._qdrant:
                            self._qdrant.delete_episode(old_id)
                        # Delete original episode (replaced by merged)
                        await self._episode_store.delete(episode.id, user_id, self._agent_id)
                        if self._qdrant:
                            self._qdrant.delete_episode(episode.id)
                        await self._episode_store.save(merged_ep)
                        # Upsert merged episode vector
                        if self._qdrant and merged_ep.embedding:
                            self._qdrant.upsert_episode(
                                merged_ep.id, user_id, self._agent_id, merged_ep.embedding
                            )
                        episode = merged_ep  # Use merged episode for downstream

                episodes.append(episode)

                # Generate semantic memories synchronously (avoids race with merge)
                if self._enable_semantic:
                    await self._on_episode_created(user_id=user_id, episode=episode)

            # Mark processed messages as done (deletes them)
            if buffer_ids:
                await self._buffer_store.mark_processed(user_id, self._agent_id, buffer_ids)

            return episodes

    async def _on_episode_created(self, user_id: str, episode: Episode) -> None:
        """Handle episode_created event by generating semantic memories."""
        try:
            # Retrieve semantically relevant existing memories via Qdrant
            existing_sem: list[SemanticMemory] = []
            if self._qdrant and episode.embedding:
                hits = self._qdrant.search_semantic(
                    user_id, self._agent_id, episode.embedding,
                    top_k=self._config.search_top_k_semantic,
                )
                ids = [h["id"] for h in hits]
                if ids:
                    existing_sem = await self._semantic_store.get_batch(ids, user_id, self._agent_id)

            memories = await self._semantic_gen.generate(
                user_id, self._agent_id, episode, existing_sem
            )
            if memories:
                await self._semantic_store.save_batch(memories)
                # Upsert semantic vectors to Qdrant
                if self._qdrant:
                    for mem in memories:
                        if mem.embedding:
                            self._qdrant.upsert_semantic(
                                mem.id, user_id, self._agent_id, mem.embedding
                            )
                logger.info(
                    "Generated %d semantic memories for user %s",
                    len(memories), user_id,
                )
        except Exception as e:
            logger.error("Semantic generation failed for user %s: %s", user_id, e)

    async def search(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int | None = None,
        top_k_semantic: int | None = None,
        method: SearchMethod = SearchMethod.HYBRID,
    ) -> SearchResult:
        return await self._search.search(
            user_id=user_id,
            agent_id=self._agent_id,
            query=query,
            top_k_episodes=top_k_episodes or self._config.search_top_k_episodes,
            top_k_semantic=top_k_semantic or self._config.search_top_k_semantic,
            method=method,
        )

    async def delete_episode(self, user_id: str, episode_id: str) -> None:
        await self._episode_store.delete(episode_id, user_id, self._agent_id)
        if self._qdrant:
            self._qdrant.delete_episode(episode_id)

    async def delete_semantic(self, user_id: str, memory_id: str) -> None:
        await self._semantic_store.delete(memory_id, user_id, self._agent_id)
        if self._qdrant:
            self._qdrant.delete_semantic(memory_id)

    async def delete_user(self, user_id: str) -> None:
        await self._semantic_store.delete_by_user(user_id, self._agent_id)
        await self._episode_store.delete_by_user(user_id, self._agent_id)
        if self._qdrant:
            self._qdrant.delete_semantic_by_user(user_id, self._agent_id)
            self._qdrant.delete_episodes_by_user(user_id, self._agent_id)

    async def stats(self, user_id: str) -> dict[str, Any]:
        episodes = await self._episode_store.list_by_user(user_id, self._agent_id)
        semantics = await self._semantic_store.list_by_user(user_id, self._agent_id)
        buffer_count = await self._buffer_store.count_unprocessed(user_id, self._agent_id)
        return {
            "user_id": user_id,
            "agent_id": self._agent_id,
            "episode_count": len(episodes),
            "semantic_memory_count": len(semantics),
            "pending_messages": buffer_count,
            "orchestrator_stats": {
                "total_requests": self._orchestrator.stats.total_requests,
                "total_errors": self._orchestrator.stats.total_errors,
            },
        }

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait for all background tasks to complete."""
        # Drain our own tasks
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            done, not_done = await asyncio.wait(pending, timeout=timeout)
            for t in not_done:
                t.cancel()
        # Drain event bus tasks and log errors
        errors = await self._event_bus.drain(timeout=max(1.0, timeout / 2))
        for err in errors:
            logger.error("Background task failed: %s", err)
