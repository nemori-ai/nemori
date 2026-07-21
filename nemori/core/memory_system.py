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
from nemori.llm.generators.consolidator import SemanticConsolidator
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
        semantic_consolidator: SemanticConsolidator | None = None,
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
        self._semantic_consolidator = semantic_consolidator or SemanticConsolidator(
            orchestrator
        )

        self._user_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()
        self._observation_tasks: dict[str, asyncio.Task] = {}

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
        # In the paper's batch path, the observation step starts when buffer B
        # reaches w messages.  ``flush`` remains the explicit way to process a
        # final partial window.  The legacy minimum threshold still applies
        # when batch segmentation is disabled.
        trigger_threshold = self._observation_trigger_threshold()
        if count >= trigger_threshold:
            self._ensure_observation_worker(user_id)

    def _observation_trigger_threshold(self) -> int:
        return (
            self._config.observation_window
            if self._config.enable_batch_segmentation
            else self._config.buffer_size_min
        )

    def _ensure_observation_worker(self, user_id: str) -> None:
        """Start at most one background observation worker per user."""
        key = f"{self._agent_id}:{user_id}"
        existing = self._observation_tasks.get(key)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(self._observation_worker(user_id, key))
        self._observation_tasks[key] = task
        self._tasks.add(task)
        task.add_done_callback(lambda done, worker_key=key: self._observation_done(worker_key, done))

    def _observation_done(self, key: str, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if self._observation_tasks.get(key) is task:
            self._observation_tasks.pop(key, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                detail = str(error).strip()
                label = f"{type(error).__name__}: {detail}" if detail else type(error).__name__
                logger.error("Observation worker %s failed: %s", key, label)

    async def _observation_worker(self, user_id: str, key: str) -> None:
        """Observe full windows until the user's buffer falls below threshold."""
        threshold = self._observation_trigger_threshold()
        current = asyncio.current_task()

        while True:
            await self._process(user_id)
            count = await self._buffer_store.count_unprocessed(user_id, self._agent_id)
            if count >= threshold:
                continue

            # Unregister before the final recheck.  Messages arriving before
            # this point were unable to schedule a second worker and are seen
            # by the recheck; messages arriving afterwards can schedule one.
            if current is not None and self._observation_tasks.get(key) is current:
                self._observation_tasks.pop(key, None)

            count = await self._buffer_store.count_unprocessed(user_id, self._agent_id)
            if count < threshold:
                return

            replacement = self._observation_tasks.get(key)
            if replacement is not None and replacement is not current and not replacement.done():
                return
            if current is not None:
                self._observation_tasks[key] = current

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
                and len(messages) >= self._config.observation_window
            ):
                segmenter = BatchSegmenter(
                    orchestrator=self._orchestrator,
                    window_size=self._config.observation_window,
                    timeout=self._config.llm_timeout,
                    retries=self._config.llm_retries,
                )
                groups = await segmenter.segment(messages)
            else:
                groups = [{"messages": messages, "topic": "conversation"}]

            # Generate episodes for each group
            total_groups = len(groups)
            for group_number, group in enumerate(groups, 1):
                group_msgs = group["messages"]
                if len(group_msgs) < self._config.episode_min_messages:
                    continue

                logger.info(
                    "Memory construction progress: episode %d/%d, %d messages, "
                    "topic=%s",
                    group_number,
                    total_groups,
                    len(group_msgs),
                    str(group.get("topic", "conversation"))[:120],
                )

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

                logger.info(
                    "Memory construction progress: completed episode %d/%d",
                    group_number,
                    total_groups,
                )

            # Mark processed messages as done (deletes them)
            if buffer_ids:
                await self._buffer_store.mark_processed(user_id, self._agent_id, buffer_ids)

            return episodes

    async def _on_episode_created(self, user_id: str, episode: Episode) -> None:
        """Handle episode_created event by generating semantic memories."""
        try:
            # Native context evocation: threshold-filtered top-Ks semantic search.
            existing_sem: list[SemanticMemory] = []
            if self._qdrant and episode.embedding:
                hits = self._qdrant.search_semantic(
                    user_id, self._agent_id, episode.embedding,
                    top_k=self._config.semantic_context_top_k,
                )
                ids = [
                    hit["id"]
                    for hit in hits
                    if float(hit.get("score", 0))
                    > self._config.semantic_context_similarity_threshold
                ]
                if ids:
                    existing_sem = await self._semantic_store.get_batch(ids, user_id, self._agent_id)

            memories = await self._semantic_gen.generate(
                user_id, self._agent_id, episode, existing_sem
            )
            if memories:
                decisions: dict[str, int] = {
                    "NEW": 0,
                    "MERGE": 0,
                    "CONFLICT_DELETE": 0,
                }
                for memory in memories:
                    candidates: list[SemanticMemory] = []
                    if (
                        self._config.enable_semantic_consolidation
                        and self._qdrant
                        and memory.embedding
                    ):
                        candidate_hits = self._qdrant.search_semantic(
                            user_id,
                            self._agent_id,
                            memory.embedding,
                            top_k=self._config.semantic_consolidation_top_k,
                        )
                        candidate_ids = [hit["id"] for hit in candidate_hits]
                        if candidate_ids:
                            candidates = await self._semantic_store.get_batch(
                                candidate_ids, user_id, self._agent_id
                            )

                    decision = await self._semantic_consolidator.decide(
                        memory, candidates
                    )
                    decisions[decision.decision] += 1
                    if decision.decision == "MERGE" and decision.new_content:
                        memory.content = decision.new_content
                        memory.embedding = await self._embedding.embed(memory.content)

                    memory.metadata = {
                        **memory.metadata,
                        "consolidation": {
                            "decision": decision.decision,
                            "target_ids": list(decision.target_ids),
                            "reason": decision.reason,
                        },
                    }

                    # Insert the replacement first so a later storage failure
                    # cannot erase the only durable version of the knowledge.
                    await self._semantic_store.save(memory)
                    if self._qdrant and memory.embedding:
                        self._qdrant.upsert_semantic(
                            memory.id, user_id, self._agent_id, memory.embedding
                        )
                    for target_id in decision.target_ids:
                        await self._semantic_store.delete(
                            target_id, user_id, self._agent_id
                        )
                        if self._qdrant:
                            self._qdrant.delete_semantic(target_id)
                logger.info(
                    "Generated and consolidated %d semantic memories for user %s "
                    "(NEW=%d, MERGE=%d, CONFLICT_DELETE=%d)",
                    len(memories),
                    user_id,
                    decisions["NEW"],
                    decisions["MERGE"],
                    decisions["CONFLICT_DELETE"],
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
                "total_tokens": self._orchestrator.stats.total_tokens,
                "prompt_tokens": self._orchestrator.stats.total_prompt_tokens,
                "completion_tokens": self._orchestrator.stats.total_completion_tokens,
                "total_errors": self._orchestrator.stats.total_errors,
                "avg_latency_ms": round(self._orchestrator.stats.avg_latency_ms, 1),
                "requests_by_phase": self._orchestrator.stats.requests_by_phase,
                "tokens_by_phase": self._orchestrator.stats.tokens_by_phase,
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
