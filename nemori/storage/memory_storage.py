"""
In-memory storage implementation for Nemori.

This module provides a simple in-memory implementation of the storage repositories,
suitable for development, testing, and small-scale deployments.
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta

from ..core.data_types import DataType, RawEventData
from ..core.episode import Episode, EpisodeLevel, EpisodeType
from .repository import EpisodicMemoryRepository, RawDataRepository
from .storage_types import (
    EpisodeQuery,
    EpisodeSearchResult,
    RawDataQuery,
    RawDataSearchResult,
    SortBy,
    SortOrder,
    StorageConfig,
    StorageStats,
)


class MemoryRawDataRepository(RawDataRepository):
    """In-memory implementation of raw data repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._data: dict[str, RawEventData] = {}
        self._data_by_type: dict[DataType, set[str]] = defaultdict(set)
        self._data_by_source: dict[str, set[str]] = defaultdict(set)
        self._processed_data: set[str] = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the in-memory storage."""
        self._initialized = True

    async def close(self) -> None:
        """Close the storage (no-op for memory)."""
        pass

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        return self._initialized

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()
        stats.total_raw_data = len(self._data)
        stats.processed_raw_data = len(self._processed_data)

        # Count by type
        for data_type, data_ids in self._data_by_type.items():
            stats.raw_data_by_type[data_type] = len(data_ids)

        # Calculate approximate storage size
        total_size = 0
        for data in self._data.values():
            total_size += len(json.dumps(data.to_dict()).encode("utf-8"))
        stats.storage_size_mb = total_size / (1024 * 1024)

        # Temporal stats
        if self._data:
            timestamps = [data.temporal_info.timestamp for data in self._data.values()]
            stats.oldest_data = min(timestamps)
            stats.newest_data = max(timestamps)

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup (serialize to JSON file)."""
        try:
            backup_data = {
                "data": {data_id: data.to_dict() for data_id, data in self._data.items()},
                "processed": list(self._processed_data),
                "timestamp": datetime.now().isoformat(),
            }
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup."""
        try:
            with open(source, encoding="utf-8") as f:
                backup_data = json.load(f)

            # Clear existing data
            self._data.clear()
            self._data_by_type.clear()
            self._data_by_source.clear()
            self._processed_data.clear()

            # Restore data
            for data_id, data_dict in backup_data["data"].items():
                raw_data = RawEventData.from_dict(data_dict)
                self._data[data_id] = raw_data
                self._data_by_type[raw_data.data_type].add(data_id)
                if raw_data.source:
                    self._data_by_source[raw_data.source].add(data_id)

            self._processed_data = set(backup_data.get("processed", []))
            return True
        except Exception:
            return False

    async def store_raw_data(self, data: RawEventData) -> str:
        """Store raw event data."""
        data_id = data.data_id
        self._data[data_id] = data
        self._data_by_type[data.data_type].add(data_id)
        if data.source:
            self._data_by_source[data.source].add(data_id)

        if data.processed:
            self._processed_data.add(data_id)

        return data_id

    async def store_raw_data_batch(self, data_list: list[RawEventData]) -> list[str]:
        """Store multiple raw event data in batch."""
        data_ids = []
        for data in data_list:
            data_id = await self.store_raw_data(data)
            data_ids.append(data_id)
        return data_ids

    async def get_raw_data(self, data_id: str) -> RawEventData | None:
        """Retrieve raw event data by ID."""
        return self._data.get(data_id)

    async def get_raw_data_batch(self, data_ids: list[str]) -> list[RawEventData | None]:
        """Retrieve multiple raw event data by IDs."""
        return [self._data.get(data_id) for data_id in data_ids]

    async def search_raw_data(self, query: RawDataQuery) -> RawDataSearchResult:
        """Search raw event data based on query parameters."""
        start_time = time.time()

        # Get candidate data IDs
        candidate_ids = set(self._data.keys())

        # Apply filters
        if query.data_ids:
            candidate_ids &= set(query.data_ids)

        if query.data_types:
            type_ids = set()
            for data_type in query.data_types:
                type_ids.update(self._data_by_type.get(data_type, set()))
            candidate_ids &= type_ids

        if query.sources:
            source_ids = set()
            for source in query.sources:
                source_ids.update(self._data_by_source.get(source, set()))
            candidate_ids &= source_ids

        if query.processed_only is not None:
            if query.processed_only:
                candidate_ids &= self._processed_data
            else:
                candidate_ids -= self._processed_data

        # Filter by time range and content
        matching_data = []
        for data_id in candidate_ids:
            data = self._data[data_id]

            # Time range filter
            if query.time_range and not query.time_range.contains(data.temporal_info.timestamp):
                continue

            # Content search
            if query.content_contains:
                content_str = str(data.content).lower()
                if query.content_contains.lower() not in content_str:
                    continue

            # Metadata filters
            if query.metadata_filters:
                matches_metadata = True
                for key, value in query.metadata_filters.items():
                    if key not in data.metadata or data.metadata[key] != value:
                        matches_metadata = False
                        break
                if not matches_metadata:
                    continue

            matching_data.append(data)

        # Sort results
        if query.sort_by == SortBy.TIMESTAMP:
            matching_data.sort(key=lambda x: x.temporal_info.timestamp, reverse=(query.sort_order == SortOrder.DESC))

        # Apply pagination
        total_count = len(matching_data)
        offset = query.offset or 0
        limit = query.limit

        if limit:
            end_idx = offset + limit
            paginated_data = matching_data[offset:end_idx]
            has_more = end_idx < total_count
        else:
            paginated_data = matching_data[offset:]
            has_more = False

        query_time_ms = (time.time() - start_time) * 1000

        return RawDataSearchResult(
            data=paginated_data, total_count=total_count, has_more=has_more, query_time_ms=query_time_ms
        )

    async def update_raw_data(self, data_id: str, data: RawEventData) -> bool:
        """Update existing raw event data."""
        if data_id not in self._data:
            return False

        old_data = self._data[data_id]

        # Update indexes
        self._data_by_type[old_data.data_type].discard(data_id)
        if old_data.source:
            self._data_by_source[old_data.source].discard(data_id)

        self._data[data_id] = data
        self._data_by_type[data.data_type].add(data_id)
        if data.source:
            self._data_by_source[data.source].add(data_id)

        if data.processed:
            self._processed_data.add(data_id)
        else:
            self._processed_data.discard(data_id)

        return True

    async def mark_as_processed(self, data_id: str, processing_version: str) -> bool:
        """Mark raw data as processed."""
        if data_id not in self._data:
            return False

        data = self._data[data_id]
        # Create updated data with processed flag
        updated_data = RawEventData(
            data_id=data.data_id,
            data_type=data.data_type,
            content=data.content,
            source=data.source,
            temporal_info=data.temporal_info,
            metadata=data.metadata,
            processed=True,
            processing_version=processing_version,
        )

        self._data[data_id] = updated_data
        self._processed_data.add(data_id)
        return True

    async def mark_batch_as_processed(self, data_ids: list[str], processing_version: str) -> list[bool]:
        """Mark multiple raw data as processed."""
        results = []
        for data_id in data_ids:
            success = await self.mark_as_processed(data_id, processing_version)
            results.append(success)
        return results

    async def delete_raw_data(self, data_id: str) -> bool:
        """Delete raw event data."""
        if data_id not in self._data:
            return False

        data = self._data[data_id]

        # Remove from indexes
        self._data_by_type[data.data_type].discard(data_id)
        if data.source:
            self._data_by_source[data.source].discard(data_id)
        self._processed_data.discard(data_id)

        # Remove data
        del self._data[data_id]
        return True

    async def get_unprocessed_data(
        self, data_type: DataType | None = None, limit: int | None = None
    ) -> list[RawEventData]:
        """Get unprocessed raw event data."""
        unprocessed_ids = set(self._data.keys()) - self._processed_data

        if data_type:
            type_ids = self._data_by_type.get(data_type, set())
            unprocessed_ids &= type_ids

        unprocessed_data = [self._data[data_id] for data_id in unprocessed_ids]

        # Sort by timestamp (oldest first for processing)
        unprocessed_data.sort(key=lambda x: x.temporal_info.timestamp)

        if limit:
            unprocessed_data = unprocessed_data[:limit]

        return unprocessed_data


class MemoryEpisodicMemoryRepository(EpisodicMemoryRepository):
    """In-memory implementation of episodic memory repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._episodes: dict[str, Episode] = {}
        self._episodes_by_owner: dict[str, set[str]] = defaultdict(set)
        self._episodes_by_type: dict[EpisodeType, set[str]] = defaultdict(set)
        self._episodes_by_level: dict[EpisodeLevel, set[str]] = defaultdict(set)
        self._episode_to_raw_data: dict[str, set[str]] = defaultdict(set)
        self._raw_data_to_episodes: dict[str, set[str]] = defaultdict(set)
        self._episode_relationships: dict[str, set[str]] = defaultdict(set)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the in-memory storage."""
        self._initialized = True

    async def close(self) -> None:
        """Close the storage (no-op for memory)."""
        pass

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        return self._initialized

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()
        stats.total_episodes = len(self._episodes)

        # Count by type
        for episode_type, episode_ids in self._episodes_by_type.items():
            stats.episodes_by_type[episode_type] = len(episode_ids)

        # Count by level
        for episode_level, episode_ids in self._episodes_by_level.items():
            stats.episodes_by_level[episode_level] = len(episode_ids)

        # Calculate approximate storage size
        total_size = 0
        for episode in self._episodes.values():
            total_size += len(json.dumps(episode.to_dict()).encode("utf-8"))
        stats.storage_size_mb = total_size / (1024 * 1024)

        # Temporal stats
        if self._episodes:
            timestamps = [episode.temporal_info.timestamp for episode in self._episodes.values()]
            stats.oldest_data = min(timestamps)
            stats.newest_data = max(timestamps)

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup."""
        try:
            backup_data = {
                "episodes": {ep_id: ep.to_dict() for ep_id, ep in self._episodes.items()},
                "episode_to_raw_data": {ep_id: list(raw_ids) for ep_id, raw_ids in self._episode_to_raw_data.items()},
                "episode_relationships": {
                    ep_id: list(rel_ids) for ep_id, rel_ids in self._episode_relationships.items()
                },
                "timestamp": datetime.now().isoformat(),
            }
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup."""
        try:
            with open(source, encoding="utf-8") as f:
                backup_data = json.load(f)

            # Clear existing data
            self._episodes.clear()
            self._episodes_by_owner.clear()
            self._episodes_by_type.clear()
            self._episodes_by_level.clear()
            self._episode_to_raw_data.clear()
            self._raw_data_to_episodes.clear()
            self._episode_relationships.clear()

            # Restore episodes
            for _ep_id, ep_dict in backup_data["episodes"].items():
                episode = Episode.from_dict(ep_dict)
                await self.store_episode(episode)

            # Restore relationships
            for ep_id, raw_ids in backup_data.get("episode_to_raw_data", {}).items():
                self._episode_to_raw_data[ep_id] = set(raw_ids)
                for raw_id in raw_ids:
                    self._raw_data_to_episodes[raw_id].add(ep_id)

            for ep_id, rel_ids in backup_data.get("episode_relationships", {}).items():
                self._episode_relationships[ep_id] = set(rel_ids)

            return True
        except Exception:
            return False

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode."""
        episode_id = episode.episode_id
        self._episodes[episode_id] = episode

        # Update indexes
        self._episodes_by_owner[episode.owner_id].add(episode_id)
        self._episodes_by_type[episode.episode_type].add(episode_id)
        self._episodes_by_level[episode.level].add(episode_id)

        return episode_id

    async def store_episode_batch(self, episodes: list[Episode]) -> list[str]:
        """Store multiple episodes in batch."""
        episode_ids = []
        for episode in episodes:
            episode_id = await self.store_episode(episode)
            episode_ids.append(episode_id)
        return episode_ids

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID."""
        return self._episodes.get(episode_id)

    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """Retrieve multiple episodes by IDs."""
        return [self._episodes.get(episode_id) for episode_id in episode_ids]

    def _calculate_text_relevance(self, text: str, query_terms: list[str]) -> float:
        """Calculate text relevance score based on term frequency."""
        if not query_terms:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for term in query_terms if term.lower() in text_lower)
        return matches / len(query_terms)

    def _calculate_embedding_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        # Simple dot product / magnitude calculation
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def search_episodes(self, query: EpisodeQuery) -> EpisodeSearchResult:
        """Search episodes based on query parameters."""
        start_time = time.time()

        # Get candidate episode IDs
        candidate_ids = set(self._episodes.keys())

        # Apply basic filters
        if query.episode_ids:
            candidate_ids &= set(query.episode_ids)

        if query.owner_ids:
            owner_ids = set()
            for owner_id in query.owner_ids:
                owner_ids.update(self._episodes_by_owner.get(owner_id, set()))
            candidate_ids &= owner_ids

        if query.episode_types:
            type_ids = set()
            for episode_type in query.episode_types:
                type_ids.update(self._episodes_by_type.get(episode_type, set()))
            candidate_ids &= type_ids

        if query.levels:
            level_ids = set()
            for level in query.levels:
                level_ids.update(self._episodes_by_level.get(level, set()))
            candidate_ids &= level_ids

        # Filter episodes and calculate relevance
        matching_episodes = []
        relevance_scores = []

        for episode_id in candidate_ids:
            episode = self._episodes[episode_id]

            # Time range filter
            if query.time_range and not query.time_range.contains(episode.temporal_info.timestamp):
                continue

            # Recent episodes filter
            if query.recent_hours:
                time_threshold = datetime.now() - timedelta(hours=query.recent_hours)
                if episode.temporal_info.timestamp < time_threshold:
                    continue

            # Importance filters
            if query.min_importance is not None and episode.importance_score < query.min_importance:
                continue
            if query.max_importance is not None and episode.importance_score > query.max_importance:
                continue

            # Recall count filters
            if query.min_recall_count is not None and episode.recall_count < query.min_recall_count:
                continue
            if query.max_recall_count is not None and episode.recall_count > query.max_recall_count:
                continue

            # Calculate relevance score
            relevance = 0.0

            # Text search
            if query.text_search:
                combined_text = f"{episode.title} {episode.content} {episode.summary}"
                relevance += self._calculate_text_relevance(combined_text, [query.text_search]) * 0.4

            # Keywords search
            if query.keywords:
                combined_text = f"{episode.title} {episode.content} {episode.summary}"
                relevance += self._calculate_text_relevance(combined_text, query.keywords) * 0.3

            # Entities search
            if query.entities:
                relevance += self._calculate_text_relevance(" ".join(episode.metadata.entities), query.entities) * 0.2

            # Topics search
            if query.topics:
                relevance += self._calculate_text_relevance(" ".join(episode.metadata.topics), query.topics) * 0.1

            # Embedding similarity
            if query.embedding_query and episode.embedding_vector:
                similarity = self._calculate_embedding_similarity(query.embedding_query, episode.embedding_vector)
                if query.similarity_threshold and similarity < query.similarity_threshold:
                    continue
                relevance += similarity * 0.5

            # If no specific relevance criteria, use importance and recency
            if relevance == 0.0:
                relevance = episode.importance_score * 0.7 + (1.0 if episode.is_recent() else 0.0) * 0.3

            matching_episodes.append(episode)
            relevance_scores.append(relevance)

        # Sort results
        if query.sort_by == SortBy.RELEVANCE and relevance_scores:
            # Sort by relevance score
            sorted_pairs = sorted(
                zip(matching_episodes, relevance_scores, strict=False),
                key=lambda x: x[1],
                reverse=(query.sort_order == SortOrder.DESC),
            )
            matching_episodes, relevance_scores = zip(*sorted_pairs, strict=False) if sorted_pairs else ([], [])
            matching_episodes = list(matching_episodes)
            relevance_scores = list(relevance_scores)
        elif query.sort_by == SortBy.TIMESTAMP:
            # Sort by timestamp
            sorted_data = sorted(
                zip(matching_episodes, relevance_scores, strict=False),
                key=lambda x: x[0].temporal_info.timestamp,
                reverse=(query.sort_order == SortOrder.DESC),
            )
            matching_episodes, relevance_scores = zip(*sorted_data, strict=False) if sorted_data else ([], [])
            matching_episodes = list(matching_episodes)
            relevance_scores = list(relevance_scores)
        elif query.sort_by == SortBy.IMPORTANCE:
            # Sort by importance
            sorted_data = sorted(
                zip(matching_episodes, relevance_scores, strict=False),
                key=lambda x: x[0].importance_score,
                reverse=(query.sort_order == SortOrder.DESC),
            )
            matching_episodes, relevance_scores = zip(*sorted_data, strict=False) if sorted_data else ([], [])
            matching_episodes = list(matching_episodes)
            relevance_scores = list(relevance_scores)
        elif query.sort_by == SortBy.RECALL_COUNT:
            # Sort by recall count
            sorted_data = sorted(
                zip(matching_episodes, relevance_scores, strict=False),
                key=lambda x: x[0].recall_count,
                reverse=(query.sort_order == SortOrder.DESC),
            )
            matching_episodes, relevance_scores = zip(*sorted_data, strict=False) if sorted_data else ([], [])
            matching_episodes = list(matching_episodes)
            relevance_scores = list(relevance_scores)

        # Apply pagination
        total_count = len(matching_episodes)
        offset = query.offset or 0
        limit = query.limit

        if limit:
            end_idx = offset + limit
            paginated_episodes = matching_episodes[offset:end_idx]
            paginated_scores = relevance_scores[offset:end_idx] if relevance_scores else None
            has_more = end_idx < total_count
        else:
            paginated_episodes = matching_episodes[offset:]
            paginated_scores = relevance_scores[offset:] if relevance_scores else None
            has_more = False

        query_time_ms = (time.time() - start_time) * 1000

        return EpisodeSearchResult(
            episodes=paginated_episodes,
            total_count=total_count,
            has_more=has_more,
            query_time_ms=query_time_ms,
            relevance_scores=paginated_scores,
        )

    async def search_episodes_by_text(
        self, text: str, owner_id: str | None = None, limit: int | None = None
    ) -> EpisodeSearchResult:
        """Search episodes by text content."""
        query = EpisodeQuery(
            text_search=text, owner_ids=[owner_id] if owner_id else None, limit=limit, sort_by=SortBy.RELEVANCE
        )
        return await self.search_episodes(query)

    async def search_episodes_by_keywords(
        self, keywords: list[str], owner_id: str | None = None, limit: int | None = None
    ) -> EpisodeSearchResult:
        """Search episodes by keywords."""
        query = EpisodeQuery(
            keywords=keywords, owner_ids=[owner_id] if owner_id else None, limit=limit, sort_by=SortBy.RELEVANCE
        )
        return await self.search_episodes(query)

    async def search_episodes_by_embedding(
        self,
        embedding: list[float],
        owner_id: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> EpisodeSearchResult:
        """Search episodes by semantic similarity using embeddings."""
        query = EpisodeQuery(
            embedding_query=embedding,
            similarity_threshold=threshold,
            owner_ids=[owner_id] if owner_id else None,
            limit=limit,
            sort_by=SortBy.RELEVANCE,
        )
        return await self.search_episodes(query)

    async def get_episodes_by_owner(
        self, owner_id: str, limit: int | None = None, offset: int | None = None
    ) -> EpisodeSearchResult:
        """Get episodes for a specific owner."""
        query = EpisodeQuery(owner_ids=[owner_id], limit=limit, offset=offset, sort_by=SortBy.TIMESTAMP)
        return await self.search_episodes(query)

    async def get_recent_episodes(
        self, owner_id: str | None = None, hours: int = 24, limit: int | None = None
    ) -> EpisodeSearchResult:
        """Get recent episodes within specified time window."""
        query = EpisodeQuery(
            owner_ids=[owner_id] if owner_id else None, recent_hours=hours, limit=limit, sort_by=SortBy.TIMESTAMP
        )
        return await self.search_episodes(query)

    async def update_episode(self, episode_id: str, episode: Episode) -> bool:
        """Update an existing episode."""
        if episode_id not in self._episodes:
            return False

        old_episode = self._episodes[episode_id]

        # Update indexes
        self._episodes_by_owner[old_episode.owner_id].discard(episode_id)
        self._episodes_by_type[old_episode.episode_type].discard(episode_id)
        self._episodes_by_level[old_episode.level].discard(episode_id)

        self._episodes[episode_id] = episode
        self._episodes_by_owner[episode.owner_id].add(episode_id)
        self._episodes_by_type[episode.episode_type].add(episode_id)
        self._episodes_by_level[episode.level].add(episode_id)

        return True

    async def update_episode_importance(self, episode_id: str, importance_score: float) -> bool:
        """Update episode importance score."""
        if episode_id not in self._episodes:
            return False

        episode = self._episodes[episode_id]
        episode.update_importance(importance_score)
        return True

    async def mark_episode_accessed(self, episode_id: str) -> bool:
        """Mark an episode as accessed."""
        if episode_id not in self._episodes:
            return False

        episode = self._episodes[episode_id]
        episode.mark_accessed()
        return True

    async def link_episode_to_raw_data(self, episode_id: str, raw_data_ids: list[str]) -> bool:
        """Create association between episode and its source raw data."""
        if episode_id not in self._episodes:
            return False

        for raw_data_id in raw_data_ids:
            self._episode_to_raw_data[episode_id].add(raw_data_id)
            self._raw_data_to_episodes[raw_data_id].add(episode_id)

        return True

    async def get_episodes_for_raw_data(self, raw_data_id: str) -> list[Episode]:
        """Get all episodes that were created from specific raw data."""
        episode_ids = self._raw_data_to_episodes.get(raw_data_id, set())
        return [self._episodes[ep_id] for ep_id in episode_ids if ep_id in self._episodes]

    async def get_raw_data_for_episode(self, episode_id: str) -> list[str]:
        """Get raw data IDs that contributed to an episode."""
        return list(self._episode_to_raw_data.get(episode_id, set()))

    async def link_related_episodes(self, episode_id1: str, episode_id2: str) -> bool:
        """Create bidirectional relationship between episodes."""
        if episode_id1 not in self._episodes or episode_id2 not in self._episodes:
            return False

        self._episode_relationships[episode_id1].add(episode_id2)
        self._episode_relationships[episode_id2].add(episode_id1)

        # Also update episode metadata
        episode1 = self._episodes[episode_id1]
        episode2 = self._episodes[episode_id2]
        episode1.add_related_episode(episode_id2)
        episode2.add_related_episode(episode_id1)

        return True

    async def get_related_episodes(self, episode_id: str) -> list[Episode]:
        """Get episodes related to a specific episode."""
        related_ids = self._episode_relationships.get(episode_id, set())
        return [self._episodes[ep_id] for ep_id in related_ids if ep_id in self._episodes]

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        if episode_id not in self._episodes:
            return False

        episode = self._episodes[episode_id]

        # Remove from indexes
        self._episodes_by_owner[episode.owner_id].discard(episode_id)
        self._episodes_by_type[episode.episode_type].discard(episode_id)
        self._episodes_by_level[episode.level].discard(episode_id)

        # Remove relationships
        for raw_data_id in self._episode_to_raw_data[episode_id]:
            self._raw_data_to_episodes[raw_data_id].discard(episode_id)
        del self._episode_to_raw_data[episode_id]

        for related_id in self._episode_relationships[episode_id]:
            self._episode_relationships[related_id].discard(episode_id)
        del self._episode_relationships[episode_id]

        # Remove episode
        del self._episodes[episode_id]
        return True

    async def cleanup_old_episodes(self, max_age_days: int) -> int:
        """Clean up episodes older than specified age."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        old_episode_ids = []

        for episode_id, episode in self._episodes.items():
            if episode.temporal_info.timestamp < cutoff_date:
                old_episode_ids.append(episode_id)

        deleted_count = 0
        for episode_id in old_episode_ids:
            if await self.delete_episode(episode_id):
                deleted_count += 1

        return deleted_count
