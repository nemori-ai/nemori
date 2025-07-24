"""
JSONL-based storage implementation for Nemori.

This module provides a simple file-based storage implementation using JSONL
format for both raw data and episodes. Perfect for development, testing,
and scenarios where database setup is inconvenient.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.data_types import DataType, RawEventData, TemporalInfo
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


def _serialize_datetime(obj: Any) -> Any:
    """Serialize datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj


def _deserialize_datetime(obj: Any) -> Any:
    """Deserialize ISO format strings back to datetime objects."""
    if isinstance(obj, str) and obj.endswith('Z') or 'T' in obj:
        try:
            return datetime.fromisoformat(obj.replace('Z', '+00:00'))
        except ValueError:
            return obj
    elif isinstance(obj, dict):
        return {k: _deserialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_datetime(item) for item in obj]
    return obj


class JSONLRawDataRepository(RawDataRepository):
    """JSONL-based implementation of raw data repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.data_dir = Path(config.connection_string or "nemori_data")
        self.raw_data_file = self.data_dir / "raw_data.jsonl"
        self._cache = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the JSONL storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create files if they don't exist
        if not self.raw_data_file.exists():
            self.raw_data_file.touch()

        # Load existing data into cache
        await self._load_cache()
        self._initialized = True

    async def close(self) -> None:
        """Close storage and cleanup resources."""
        await self._flush_cache()
        self._cache.clear()

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        return self._initialized and self.data_dir.exists() and self.raw_data_file.exists()

    async def backup(self, destination: str) -> bool:
        """Create a backup of the storage."""
        try:
            import shutil
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.data_dir, dest_path / "nemori_backup", dirs_exist_ok=True)
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore storage from a backup."""
        try:
            import shutil
            source_path = Path(source) / "nemori_backup"
            if source_path.exists():
                shutil.copytree(source_path, self.data_dir, dirs_exist_ok=True)
                await self._load_cache()
                return True
            return False
        except Exception:
            return False

    async def _load_cache(self) -> None:
        """Load existing data into memory cache."""
        self._cache = {}
        if not self.raw_data_file.exists():
            return

        try:
            with open(self.raw_data_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_dict = json.loads(line)
                        data_dict = _deserialize_datetime(data_dict)

                        # Reconstruct RawEventData object
                        temporal_info = TemporalInfo(
                            timestamp=data_dict['temporal_info']['timestamp'],
                            duration=data_dict['temporal_info']['duration'],
                            timezone=data_dict['temporal_info']['timezone'],
                            precision=data_dict['temporal_info'].get('precision', 'second')
                        )

                        raw_data = RawEventData(
                            data_id=data_dict['data_id'],
                            data_type=DataType(data_dict['data_type']),
                            content=data_dict['content'],
                            source=data_dict['source'],
                            temporal_info=temporal_info,
                            metadata=data_dict['metadata'],
                            processed=data_dict['processed'],
                            processing_version=data_dict['processing_version']
                        )

                        self._cache[raw_data.data_id] = raw_data
        except Exception as e:
            print(f"Warning: Failed to load cache from {self.raw_data_file}: {e}")

    async def _flush_cache(self) -> None:
        """Write cache to disk."""
        try:
            with open(self.raw_data_file, 'w', encoding='utf-8') as f:
                for raw_data in self._cache.values():
                    data_dict = {
                        'data_id': raw_data.data_id,
                        'data_type': raw_data.data_type.value,
                        'content': raw_data.content,
                        'source': raw_data.source,
                        'temporal_info': {
                            'timestamp': raw_data.temporal_info.timestamp,
                            'duration': raw_data.temporal_info.duration,
                            'timezone': raw_data.temporal_info.timezone,
                            'precision': raw_data.temporal_info.precision
                        },
                        'metadata': raw_data.metadata,
                        'processed': raw_data.processed,
                        'processing_version': raw_data.processing_version
                    }
                    data_dict = _serialize_datetime(data_dict)
                    json.dump(data_dict, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to flush cache to {self.raw_data_file}: {e}")

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        # Count total and processed data
        stats.total_raw_data = len(self._cache)
        stats.processed_raw_data = sum(1 for data in self._cache.values() if data.processed)

        # Count by type
        for data in self._cache.values():
            stats.raw_data_by_type[data.data_type] = stats.raw_data_by_type.get(data.data_type, 0) + 1

        # Get temporal info
        if self._cache:
            timestamps = [data.temporal_info.timestamp for data in self._cache.values()]
            stats.oldest_data = min(timestamps)
            stats.newest_data = max(timestamps)

        # Calculate storage size
        if self.raw_data_file.exists():
            stats.storage_size_mb = self.raw_data_file.stat().st_size / (1024 * 1024)

        return stats

    async def store_raw_data(self, data: RawEventData) -> str:
        """Store raw event data."""
        self._cache[data.data_id] = data
        await self._flush_cache()
        return data.data_id

    async def store_raw_data_batch(self, data_list: list[RawEventData]) -> list[str]:
        """Store multiple raw event data in batch."""
        data_ids = []
        for data in data_list:
            self._cache[data.data_id] = data
            data_ids.append(data.data_id)
        await self._flush_cache()
        return data_ids

    async def get_raw_data(self, data_id: str) -> RawEventData | None:
        """Retrieve raw event data by ID."""
        return self._cache.get(data_id)

    async def get_raw_data_batch(self, data_ids: list[str]) -> list[RawEventData | None]:
        """Retrieve multiple raw event data by IDs."""
        return [self._cache.get(data_id) for data_id in data_ids]

    async def search_raw_data(self, query: RawDataQuery) -> RawDataSearchResult:
        """Search raw event data based on query parameters."""
        start_time = time.time()
        results = []

        for data in self._cache.values():
            # Apply filters
            if query.data_ids and data.data_id not in query.data_ids:
                continue
            if query.data_types and data.data_type not in query.data_types:
                continue
            if query.sources and data.source not in query.sources:
                continue
            if query.processed_only is not None and data.processed != query.processed_only:
                continue

            # Time range filter
            if query.time_range:
                if not query.time_range.contains(data.temporal_info.timestamp):
                    continue

            # Content search
            if query.content_contains:
                content_str = json.dumps(data.content) if not isinstance(data.content, str) else data.content
                if query.content_contains.lower() not in content_str.lower():
                    continue

            # Metadata filters
            if query.metadata_filters:
                match = True
                for key, value in query.metadata_filters.items():
                    if key not in data.metadata or data.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(data)

        # Sort results
        if query.sort_by == SortBy.TIMESTAMP:
            results.sort(
                key=lambda x: x.temporal_info.timestamp,
                reverse=(query.sort_order == SortOrder.DESC)
            )

        # Apply pagination
        total_count = len(results)
        if query.offset:
            results = results[query.offset:]
        if query.limit:
            results = results[:query.limit]

        query_time_ms = (time.time() - start_time) * 1000
        has_more = query.limit is not None and len(results) == query.limit and total_count > (query.offset or 0) + len(results)

        return RawDataSearchResult(
            data=results,
            total_count=total_count,
            has_more=has_more,
            query_time_ms=query_time_ms
        )

    async def update_raw_data(self, data_id: str, data: RawEventData) -> bool:
        """Update existing raw event data."""
        if data_id in self._cache:
            self._cache[data_id] = data
            await self._flush_cache()
            return True
        return False

    async def mark_as_processed(self, data_id: str, processing_version: str) -> bool:
        """Mark raw data as processed."""
        if data_id in self._cache:
            data = self._cache[data_id]
            # Create a new instance with updated fields
            updated_data = RawEventData(
                data_id=data.data_id,
                data_type=data.data_type,
                content=data.content,
                source=data.source,
                temporal_info=data.temporal_info,
                metadata=data.metadata,
                processed=True,
                processing_version=processing_version
            )
            self._cache[data_id] = updated_data
            await self._flush_cache()
            return True
        return False

    async def mark_batch_as_processed(self, data_ids: list[str], processing_version: str) -> list[bool]:
        """Mark multiple raw data as processed."""
        results = []
        for data_id in data_ids:
            result = await self.mark_as_processed(data_id, processing_version)
            results.append(result)
        return results

    async def delete_raw_data(self, data_id: str) -> bool:
        """Delete raw event data."""
        if data_id in self._cache:
            del self._cache[data_id]
            await self._flush_cache()
            return True
        return False

    async def get_unprocessed_data(
        self, data_type: DataType | None = None, limit: int | None = None
    ) -> list[RawEventData]:
        """Get unprocessed raw event data."""
        results = []
        for data in self._cache.values():
            if data.processed:
                continue
            if data_type and data.data_type != data_type:
                continue
            results.append(data)
            if limit and len(results) >= limit:
                break
        return results


class JSONLEpisodicMemoryRepository(EpisodicMemoryRepository):
    """JSONL-based implementation of episodic memory repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.data_dir = Path(config.connection_string or "nemori_data")
        self.episodes_file = self.data_dir / "episodes.jsonl"
        self.links_file = self.data_dir / "episode_links.jsonl"
        self._episodes_cache = {}
        self._links_cache = {}  # episode_id -> list of raw_data_ids
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the JSONL storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create files if they don't exist
        if not self.episodes_file.exists():
            self.episodes_file.touch()
        if not self.links_file.exists():
            self.links_file.touch()

        # Load existing data into cache
        await self._load_episodes_cache()
        await self._load_links_cache()
        self._initialized = True

    async def close(self) -> None:
        """Close storage and cleanup resources."""
        await self._flush_episodes_cache()
        await self._flush_links_cache()
        self._episodes_cache.clear()
        self._links_cache.clear()

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        return (self._initialized and
                self.data_dir.exists() and
                self.episodes_file.exists() and
                self.links_file.exists())

    async def backup(self, destination: str) -> bool:
        """Create a backup of the storage."""
        try:
            import shutil
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.data_dir, dest_path / "nemori_backup", dirs_exist_ok=True)
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore storage from a backup."""
        try:
            import shutil
            source_path = Path(source) / "nemori_backup"
            if source_path.exists():
                shutil.copytree(source_path, self.data_dir, dirs_exist_ok=True)
                await self._load_episodes_cache()
                await self._load_links_cache()
                return True
            return False
        except Exception:
            return False

    async def _load_episodes_cache(self) -> None:
        """Load existing episodes into memory cache."""
        self._episodes_cache = {}
        if not self.episodes_file.exists():
            return

        try:
            with open(self.episodes_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode_dict = json.loads(line)
                        episode_dict = _deserialize_datetime(episode_dict)
                        episode = self._dict_to_episode(episode_dict)
                        self._episodes_cache[episode.episode_id] = episode
        except Exception as e:
            print(f"Warning: Failed to load episodes cache from {self.episodes_file}: {e}")

    async def _load_links_cache(self) -> None:
        """Load existing episode-raw data links into memory cache."""
        self._links_cache = {}
        if not self.links_file.exists():
            return

        try:
            with open(self.links_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        link_dict = json.loads(line)
                        episode_id = link_dict['episode_id']
                        raw_data_ids = link_dict['raw_data_ids']
                        self._links_cache[episode_id] = raw_data_ids
        except Exception as e:
            print(f"Warning: Failed to load links cache from {self.links_file}: {e}")

    async def _flush_episodes_cache(self) -> None:
        """Write episodes cache to disk."""
        try:
            with open(self.episodes_file, 'w', encoding='utf-8') as f:
                for episode in self._episodes_cache.values():
                    episode_dict = self._episode_to_dict(episode)
                    episode_dict = _serialize_datetime(episode_dict)
                    json.dump(episode_dict, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to flush episodes cache to {self.episodes_file}: {e}")

    async def _flush_links_cache(self) -> None:
        """Write links cache to disk."""
        try:
            with open(self.links_file, 'w', encoding='utf-8') as f:
                for episode_id, raw_data_ids in self._links_cache.items():
                    link_dict = {
                        'episode_id': episode_id,
                        'raw_data_ids': raw_data_ids
                    }
                    json.dump(link_dict, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to flush links cache to {self.links_file}: {e}")

    def _episode_to_dict(self, episode: Episode) -> dict:
        """Convert Episode object to dictionary."""
        return {
            'episode_id': episode.episode_id,
            'owner_id': episode.owner_id,
            'episode_type': episode.episode_type.value,
            'level': episode.level.value,
            'title': episode.title,
            'content': episode.content,
            'summary': episode.summary,
            'temporal_info': {
                'timestamp': episode.temporal_info.timestamp,
                'duration': episode.temporal_info.duration,
                'timezone': episode.temporal_info.timezone,
                'precision': episode.temporal_info.precision
            },
            'metadata': {
                'source_data_ids': episode.metadata.source_data_ids,
                'source_types': [dt.value for dt in episode.metadata.source_types],
                'entities': episode.metadata.entities,
                'topics': episode.metadata.topics,
                'key_points': episode.metadata.key_points
            },
            'search_keywords': episode.search_keywords,
            'importance_score': episode.importance_score,
            'recall_count': episode.recall_count,
            'last_accessed': episode.last_accessed
        }

    def _dict_to_episode(self, episode_dict: dict) -> Episode:
        """Convert dictionary to Episode object."""
        from ..core.episode import EpisodeMetadata

        temporal_info = TemporalInfo(
            timestamp=episode_dict['temporal_info']['timestamp'],
            duration=episode_dict['temporal_info']['duration'],
            timezone=episode_dict['temporal_info']['timezone'],
            precision=episode_dict['temporal_info'].get('precision', 'second')
        )

        metadata = EpisodeMetadata(
            source_data_ids=episode_dict['metadata']['source_data_ids'],
            source_types={DataType(dt) for dt in episode_dict['metadata']['source_types']},
            entities=episode_dict['metadata']['entities'],
            topics=episode_dict['metadata']['topics'],
            key_points=episode_dict['metadata']['key_points']
        )

        return Episode(
            episode_id=episode_dict['episode_id'],
            owner_id=episode_dict['owner_id'],
            episode_type=EpisodeType(episode_dict['episode_type']),
            level=EpisodeLevel(episode_dict['level']),
            title=episode_dict['title'],
            content=episode_dict['content'],
            summary=episode_dict['summary'],
            temporal_info=temporal_info,
            metadata=metadata,
            search_keywords=episode_dict['search_keywords'],
            importance_score=episode_dict['importance_score'],
            recall_count=episode_dict['recall_count'],
            last_accessed=episode_dict['last_accessed']
        )

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        # Count total episodes
        stats.total_episodes = len(self._episodes_cache)

        # Count by type and level
        for episode in self._episodes_cache.values():
            stats.episodes_by_type[episode.episode_type] = stats.episodes_by_type.get(episode.episode_type, 0) + 1
            stats.episodes_by_level[episode.level] = stats.episodes_by_level.get(episode.level, 0) + 1

        # Get temporal info
        if self._episodes_cache:
            timestamps = [episode.temporal_info.timestamp for episode in self._episodes_cache.values()]
            stats.oldest_data = min(timestamps)
            stats.newest_data = max(timestamps)

        # Calculate storage size
        if self.episodes_file.exists():
            stats.storage_size_mb += self.episodes_file.stat().st_size / (1024 * 1024)
        if self.links_file.exists():
            stats.storage_size_mb += self.links_file.stat().st_size / (1024 * 1024)

        return stats

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode."""
        self._episodes_cache[episode.episode_id] = episode
        await self._flush_episodes_cache()
        return episode.episode_id

    async def store_episode_batch(self, episodes: list[Episode]) -> list[str]:
        """Store multiple episodes in batch."""
        episode_ids = []
        for episode in episodes:
            self._episodes_cache[episode.episode_id] = episode
            episode_ids.append(episode.episode_id)
        await self._flush_episodes_cache()
        return episode_ids

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID."""
        return self._episodes_cache.get(episode_id)

    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """Retrieve multiple episodes by IDs."""
        return [self._episodes_cache.get(episode_id) for episode_id in episode_ids]

    async def search_episodes(self, query: EpisodeQuery) -> EpisodeSearchResult:
        """Search episodes based on query parameters."""
        start_time = time.time()
        results = []

        for episode in self._episodes_cache.values():
            # Apply filters
            if query.episode_ids and episode.episode_id not in query.episode_ids:
                continue
            if query.owner_ids and episode.owner_id not in query.owner_ids:
                continue
            if query.episode_types and episode.episode_type not in query.episode_types:
                continue
            if query.levels and episode.level not in query.levels:
                continue

            # Time range filter
            if query.time_range:
                if not query.time_range.contains(episode.temporal_info.timestamp):
                    continue

            # Text search (simple implementation)
            if query.text_search:
                search_text = query.text_search.lower()
                searchable_content = (
                    episode.title.lower() + " " +
                    episode.content.lower() + " " +
                    episode.summary.lower() + " " +
                    " ".join(episode.search_keywords).lower()
                )
                if search_text not in searchable_content:
                    continue

            # Keywords filter
            if query.keywords:
                episode_keywords = [kw.lower() for kw in episode.search_keywords]
                if not any(kw.lower() in episode_keywords for kw in query.keywords):
                    continue

            # Entities filter
            if query.entities:
                episode_entities = [ent.lower() for ent in episode.metadata.entities]
                if not any(ent.lower() in episode_entities for ent in query.entities):
                    continue

            # Topics filter
            if query.topics:
                episode_topics = [topic.lower() for topic in episode.metadata.topics]
                if not any(topic.lower() in episode_topics for topic in query.topics):
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

            # Recent episodes filter
            if query.recent_hours is not None:
                cutoff_time = datetime.now(UTC) - timedelta(hours=query.recent_hours)
                if episode.temporal_info.timestamp < cutoff_time:
                    continue

            results.append(episode)

        # Sort results
        if query.sort_by == SortBy.TIMESTAMP:
            results.sort(
                key=lambda x: x.temporal_info.timestamp,
                reverse=(query.sort_order == SortOrder.DESC)
            )
        elif query.sort_by == SortBy.IMPORTANCE:
            results.sort(
                key=lambda x: x.importance_score,
                reverse=(query.sort_order == SortOrder.DESC)
            )
        elif query.sort_by == SortBy.RECALL_COUNT:
            results.sort(
                key=lambda x: x.recall_count,
                reverse=(query.sort_order == SortOrder.DESC)
            )

        # Apply pagination
        total_count = len(results)
        if query.offset:
            results = results[query.offset:]
        if query.limit:
            results = results[:query.limit]

        query_time_ms = (time.time() - start_time) * 1000
        has_more = query.limit is not None and len(results) == query.limit and total_count > (query.offset or 0) + len(results)

        return EpisodeSearchResult(
            episodes=results,
            total_count=total_count,
            has_more=has_more,
            query_time_ms=query_time_ms
        )

    async def get_episodes_by_owner(
        self, owner_id: str, limit: int | None = None, offset: int | None = None
    ) -> EpisodeSearchResult:
        """Get episodes for a specific owner."""
        query = EpisodeQuery(
            owner_ids=[owner_id],
            limit=limit,
            offset=offset,
            sort_by=SortBy.TIMESTAMP,
            sort_order=SortOrder.DESC
        )
        return await self.search_episodes(query)

    async def get_recent_episodes(
        self, owner_id: str | None = None, hours: int = 24, limit: int | None = None
    ) -> EpisodeSearchResult:
        """Get recent episodes within specified time window."""
        query = EpisodeQuery(
            owner_ids=[owner_id] if owner_id else None,
            recent_hours=hours,
            limit=limit,
            sort_by=SortBy.TIMESTAMP,
            sort_order=SortOrder.DESC
        )
        return await self.search_episodes(query)

    async def update_episode(self, episode_id: str, episode: Episode) -> bool:
        """Update an existing episode."""
        if episode_id in self._episodes_cache:
            self._episodes_cache[episode_id] = episode
            await self._flush_episodes_cache()
            return True
        return False

    async def update_episode_importance(self, episode_id: str, importance_score: float) -> bool:
        """Update episode importance score."""
        if episode_id in self._episodes_cache:
            episode = self._episodes_cache[episode_id]
            # Create updated episode
            updated_episode = Episode(
                episode_id=episode.episode_id,
                owner_id=episode.owner_id,
                episode_type=episode.episode_type,
                level=episode.level,
                title=episode.title,
                content=episode.content,
                summary=episode.summary,
                temporal_info=episode.temporal_info,
                metadata=episode.metadata,
                search_keywords=episode.search_keywords,
                importance_score=importance_score,
                recall_count=episode.recall_count,
                last_accessed=episode.last_accessed
            )
            self._episodes_cache[episode_id] = updated_episode
            await self._flush_episodes_cache()
            return True
        return False

    async def mark_episode_accessed(self, episode_id: str) -> bool:
        """Mark an episode as accessed (increment recall count)."""
        if episode_id in self._episodes_cache:
            episode = self._episodes_cache[episode_id]
            # Create updated episode
            updated_episode = Episode(
                episode_id=episode.episode_id,
                owner_id=episode.owner_id,
                episode_type=episode.episode_type,
                level=episode.level,
                title=episode.title,
                content=episode.content,
                summary=episode.summary,
                temporal_info=episode.temporal_info,
                metadata=episode.metadata,
                search_keywords=episode.search_keywords,
                importance_score=episode.importance_score,
                recall_count=episode.recall_count + 1,
                last_accessed=datetime.now(UTC)
            )
            self._episodes_cache[episode_id] = updated_episode
            await self._flush_episodes_cache()
            return True
        return False

    async def link_episode_to_raw_data(self, episode_id: str, raw_data_ids: list[str]) -> bool:
        """Create association between episode and its source raw data."""
        self._links_cache[episode_id] = raw_data_ids
        await self._flush_links_cache()
        return True

    async def get_episodes_for_raw_data(self, raw_data_id: str) -> list[Episode]:
        """Get all episodes that were created from specific raw data."""
        episodes = []
        for episode_id, raw_data_ids in self._links_cache.items():
            if raw_data_id in raw_data_ids:
                episode = self._episodes_cache.get(episode_id)
                if episode:
                    episodes.append(episode)
        return episodes

    async def get_raw_data_for_episode(self, episode_id: str) -> list[str]:
        """Get raw data IDs that contributed to an episode."""
        return self._links_cache.get(episode_id, [])

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        deleted = False
        if episode_id in self._episodes_cache:
            del self._episodes_cache[episode_id]
            deleted = True
        if episode_id in self._links_cache:
            del self._links_cache[episode_id]
            deleted = True

        if deleted:
            await self._flush_episodes_cache()
            await self._flush_links_cache()

        return deleted

    async def cleanup_old_episodes(self, max_age_days: int) -> int:
        """Clean up episodes older than specified age."""
        cutoff_time = datetime.now(UTC) - timedelta(days=max_age_days)
        to_delete = []

        for episode_id, episode in self._episodes_cache.items():
            if episode.temporal_info.timestamp < cutoff_time:
                to_delete.append(episode_id)

        deleted_count = 0
        for episode_id in to_delete:
            if await self.delete_episode(episode_id):
                deleted_count += 1

        return deleted_count
