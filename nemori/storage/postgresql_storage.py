"""
PostgreSQL storage implementation for Nemori.

This module provides PostgreSQL-based storage implementation using SQLModel
for better type safety and async/await support with asyncpg.
"""

import json
import time
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlmodel import and_, delete, func, or_, select, update

from ..core.data_types import DataType, RawEventData
from ..core.episode import Episode, EpisodeLevel, EpisodeType
from .repository import EpisodicMemoryRepository, RawDataRepository
from .sql_models import (
    BaseSQLRepository,
    EpisodeRawDataTable,
    EpisodeTable,
    RawDataTable,
)
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


class PostgreSQLRawDataRepository(RawDataRepository, BaseSQLRepository):
    """PostgreSQL implementation of raw data repository using SQLModel."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.connection_string = config.connection_string or "postgresql+asyncpg://localhost/nemori"
        self.engine: AsyncEngine | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the PostgreSQL storage with SQLModel."""
        # Create async engine
        self.engine = create_async_engine(
            self.connection_string,
            echo=False,
            pool_size=10,
            max_overflow=0,
        )

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create tables
        from sqlmodel import SQLModel

        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            async with AsyncSession(self.engine) as session:
                await session.execute(select(1))
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        async with AsyncSession(self.engine) as session:
            # Total raw data count
            result = await session.execute(select(func.count(RawDataTable.data_id)))
            stats.total_raw_data = result.scalar()

            # Processed data count
            result = await session.execute(select(func.count(RawDataTable.data_id)).where(RawDataTable.processed))
            stats.processed_raw_data = result.scalar()

            # Count by type
            result = await session.execute(
                select(RawDataTable.data_type, func.count(RawDataTable.data_id)).group_by(RawDataTable.data_type)
            )
            type_results = result.fetchall()

            for data_type_str, count in type_results:
                try:
                    data_type = DataType(data_type_str)
                    stats.raw_data_by_type[data_type] = count
                except ValueError:
                    pass

            # Temporal stats
            result = await session.execute(select(func.min(RawDataTable.timestamp), func.max(RawDataTable.timestamp)))
            temporal_result = result.fetchone()

            if temporal_result and temporal_result[0] and temporal_result[1]:
                stats.oldest_data = temporal_result[0]
                stats.newest_data = temporal_result[1]

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup using pg_dump."""
        try:
            import subprocess
            from urllib.parse import urlparse

            # Parse connection string
            parsed = urlparse(self.connection_string.replace("+asyncpg", ""))

            # Build pg_dump command
            cmd = [
                "pg_dump",
                "-h",
                parsed.hostname or "localhost",
                "-p",
                str(parsed.port or 5432),
                "-U",
                parsed.username or "postgres",
                "-d",
                parsed.path.lstrip("/") if parsed.path else "nemori",
                "-f",
                destination,
                "--no-password",
            ]

            # Set PGPASSWORD environment variable if password exists
            env = {}
            if parsed.password:
                env["PGPASSWORD"] = parsed.password

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup using pg_restore or psql."""
        try:
            import subprocess
            from urllib.parse import urlparse

            # Parse connection string
            parsed = urlparse(self.connection_string.replace("+asyncpg", ""))

            # Build psql command for SQL file restore
            cmd = [
                "psql",
                "-h",
                parsed.hostname or "localhost",
                "-p",
                str(parsed.port or 5432),
                "-U",
                parsed.username or "postgres",
                "-d",
                parsed.path.lstrip("/") if parsed.path else "nemori",
                "-f",
                source,
                "--no-password",
            ]

            # Set PGPASSWORD environment variable if password exists
            env = {}
            if parsed.password:
                env["PGPASSWORD"] = parsed.password

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    async def store_raw_data(self, data: RawEventData) -> str:
        """Store raw event data."""
        # Validate data_id
        data_id = self.validate_id(data.data_id)

        raw_data_row = RawDataTable(
            data_id=data_id,
            data_type=data.data_type.value,
            content=json.dumps(data.content, ensure_ascii=False),
            source=data.source,
            timestamp=data.temporal_info.timestamp,
            duration=data.temporal_info.duration,
            timezone=data.temporal_info.timezone,
            precision=data.temporal_info.precision,
            event_metadata=json.dumps(data.metadata, ensure_ascii=False),
            processed=data.processed,
            processing_version=data.processing_version,
        )

        async with AsyncSession(self.engine) as session:
            session.add(raw_data_row)
            await session.commit()

        return data_id

    async def store_raw_data_batch(self, data_list: list[RawEventData]) -> list[str]:
        """Store multiple raw event data in batch."""
        data_ids = []
        raw_data_rows = []

        for data in data_list:
            data_id = self.validate_id(data.data_id)
            data_ids.append(data_id)

            raw_data_rows.append(
                RawDataTable(
                    data_id=data_id,
                    data_type=data.data_type.value,
                    content=json.dumps(data.content, ensure_ascii=False),
                    source=data.source,
                    timestamp=data.temporal_info.timestamp,
                    duration=data.temporal_info.duration,
                    timezone=data.temporal_info.timezone,
                    precision=data.temporal_info.precision,
                    event_metadata=json.dumps(data.metadata, ensure_ascii=False),
                    processed=data.processed,
                    processing_version=data.processing_version,
                )
            )

        async with AsyncSession(self.engine) as session:
            session.add_all(raw_data_rows)
            await session.commit()

        return data_ids

    async def get_raw_data(self, data_id: str) -> RawEventData | None:
        """Retrieve raw event data by ID."""
        data_id = self.validate_id(data_id)

        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(RawDataTable).where(RawDataTable.data_id == data_id))
            raw_data_row = result.scalars().first()

            if not raw_data_row:
                return None

            return self._row_to_raw_data(raw_data_row)

    async def get_raw_data_batch(self, data_ids: list[str]) -> list[RawEventData | None]:
        """Retrieve multiple raw event data by IDs."""
        if not data_ids:
            return []

        # Validate all IDs
        validated_ids = [self.validate_id(data_id) for data_id in data_ids]

        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(RawDataTable).where(RawDataTable.data_id.in_(validated_ids)))
            raw_data_rows = result.scalars().all()

            # Create mapping for quick lookup
            result_map = {row.data_id: self._row_to_raw_data(row) for row in raw_data_rows}

            # Return in the same order as requested
            return [result_map.get(data_id) for data_id in validated_ids]

    def _row_to_raw_data(self, row: RawDataTable) -> RawEventData:
        """Convert database row to RawEventData."""
        return RawEventData.from_dict(
            {
                "data_id": row.data_id,
                "data_type": row.data_type,
                "content": json.loads(row.content),
                "source": row.source,
                "temporal_info": {
                    "timestamp": row.timestamp.isoformat() if row.timestamp else datetime.now().isoformat(),
                    "duration": row.duration,
                    "timezone": row.timezone,
                    "precision": row.precision or "second",
                },
                "metadata": json.loads(row.event_metadata),
                "processed": row.processed,
                "processing_version": row.processing_version,
            }
        )

    async def search_raw_data(self, query: RawDataQuery) -> RawDataSearchResult:
        """Search raw event data based on query parameters."""
        start_time = time.time()

        # Validate inputs
        query.limit, query.offset = self.validate_limit_offset(query.limit, query.offset)

        async with AsyncSession(self.engine) as session:
            # Build base query
            stmt = select(RawDataTable)
            conditions = []

            # Apply filters
            if query.data_ids:
                validated_ids = [self.validate_id(data_id) for data_id in query.data_ids]
                conditions.append(RawDataTable.data_id.in_(validated_ids))

            if query.data_types:
                type_values = [dt.value for dt in query.data_types]
                conditions.append(RawDataTable.data_type.in_(type_values))

            if query.sources:
                conditions.append(RawDataTable.source.in_(query.sources))

            if query.time_range:
                if query.time_range.start:
                    conditions.append(RawDataTable.timestamp >= query.time_range.start)
                if query.time_range.end:
                    conditions.append(RawDataTable.timestamp <= query.time_range.end)

            if query.processed_only is not None:
                conditions.append(RawDataTable.processed == query.processed_only)

            if query.content_contains:
                search_term = self.sanitize_search_term(query.content_contains)
                conditions.append(RawDataTable.content.contains(search_term))

            # Apply conditions
            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Count total results
            count_stmt = select(func.count(RawDataTable.data_id))
            if conditions:
                count_stmt = count_stmt.where(and_(*conditions))

            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()

            # Apply sorting
            if query.sort_by == SortBy.TIMESTAMP:
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(RawDataTable.timestamp.desc())
                else:
                    stmt = stmt.order_by(RawDataTable.timestamp.asc())
            else:
                stmt = stmt.order_by(RawDataTable.timestamp.desc())

            # Apply pagination
            if query.limit:
                stmt = stmt.limit(query.limit)
            if query.offset:
                stmt = stmt.offset(query.offset)

            # Execute query
            result = await session.execute(stmt)
            results = result.scalars().all()

            # Convert to RawEventData objects
            data = [self._row_to_raw_data(row) for row in results]

            # Calculate if there are more results
            has_more = False
            if query.limit and query.offset is not None:
                has_more = (query.offset + len(data)) < total_count
            elif query.limit:
                has_more = len(data) == query.limit and total_count > query.limit

            query_time_ms = (time.time() - start_time) * 1000

            return RawDataSearchResult(
                data=data, total_count=total_count, has_more=has_more, query_time_ms=query_time_ms
            )

    async def update_raw_data(self, data_id: str, data: RawEventData) -> bool:
        """Update existing raw event data."""
        try:
            data_id = self.validate_id(data_id)

            async with AsyncSession(self.engine) as session:
                stmt = (
                    update(RawDataTable)
                    .where(RawDataTable.data_id == data_id)
                    .values(
                        data_type=data.data_type.value,
                        content=json.dumps(data.content, ensure_ascii=False),
                        source=data.source,
                        timestamp=data.temporal_info.timestamp,
                        duration=data.temporal_info.duration,
                        timezone=data.temporal_info.timezone,
                        precision=data.temporal_info.precision,
                        event_metadata=json.dumps(data.metadata, ensure_ascii=False),
                        processed=data.processed,
                        processing_version=data.processing_version,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception:
            return False

    async def mark_as_processed(self, data_id: str, processing_version: str) -> bool:
        """Mark raw data as processed."""
        try:
            data_id = self.validate_id(data_id)

            async with AsyncSession(self.engine) as session:
                stmt = (
                    update(RawDataTable)
                    .where(RawDataTable.data_id == data_id)
                    .values(processed=True, processing_version=processing_version)
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception:
            return False

    async def mark_batch_as_processed(self, data_ids: list[str], processing_version: str) -> list[bool]:
        """Mark multiple raw data as processed."""
        results = []
        for data_id in data_ids:
            success = await self.mark_as_processed(data_id, processing_version)
            results.append(success)
        return results

    async def delete_raw_data(self, data_id: str) -> bool:
        """Delete raw event data."""
        try:
            data_id = self.validate_id(data_id)

            async with AsyncSession(self.engine) as session:
                stmt = delete(RawDataTable).where(RawDataTable.data_id == data_id)
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception:
            return False

    async def get_unprocessed_data(
        self, data_type: DataType | None = None, limit: int | None = None
    ) -> list[RawEventData]:
        """Get unprocessed raw event data."""
        limit, _ = self.validate_limit_offset(limit, None)

        async with AsyncSession(self.engine) as session:
            stmt = select(RawDataTable).where(~RawDataTable.processed)

            if data_type:
                stmt = stmt.where(RawDataTable.data_type == data_type.value)

            stmt = stmt.order_by(RawDataTable.timestamp.asc())

            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            results = result.scalars().all()
            return [self._row_to_raw_data(row) for row in results]


class PostgreSQLEpisodicMemoryRepository(EpisodicMemoryRepository, BaseSQLRepository):
    """PostgreSQL implementation of episodic memory repository using SQLModel."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.connection_string = config.connection_string or "postgresql+asyncpg://localhost/nemori"
        self.engine: AsyncEngine | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the PostgreSQL storage with SQLModel."""
        # Create async engine
        self.engine = create_async_engine(
            self.connection_string,
            echo=False,
            pool_size=10,
            max_overflow=0,
        )

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create tables
        from sqlmodel import SQLModel

        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            async with AsyncSession(self.engine) as session:
                await session.execute(select(1))
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        async with AsyncSession(self.engine) as session:
            # Total episodes count
            result = await session.execute(select(func.count(EpisodeTable.episode_id)))
            stats.total_episodes = result.scalar()

            # Count by type
            result = await session.execute(
                select(EpisodeTable.episode_type, func.count(EpisodeTable.episode_id)).group_by(
                    EpisodeTable.episode_type
                )
            )
            type_results = result.fetchall()

            for episode_type_str, count in type_results:
                try:
                    episode_type = EpisodeType(episode_type_str)
                    stats.episodes_by_type[episode_type] = count
                except ValueError:
                    pass

            # Count by level
            result = await session.execute(
                select(EpisodeTable.level, func.count(EpisodeTable.episode_id)).group_by(EpisodeTable.level)
            )
            level_results = result.fetchall()

            for level_int, count in level_results:
                try:
                    episode_level = EpisodeLevel(level_int)
                    stats.episodes_by_level[episode_level] = count
                except ValueError:
                    pass

            # Temporal stats
            result = await session.execute(select(func.min(EpisodeTable.timestamp), func.max(EpisodeTable.timestamp)))
            temporal_result = result.fetchone()

            if temporal_result and temporal_result[0] and temporal_result[1]:
                stats.oldest_data = temporal_result[0]
                stats.newest_data = temporal_result[1]

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup using pg_dump."""
        try:
            import subprocess
            from urllib.parse import urlparse

            # Parse connection string
            parsed = urlparse(self.connection_string.replace("+asyncpg", ""))

            # Build pg_dump command
            cmd = [
                "pg_dump",
                "-h",
                parsed.hostname or "localhost",
                "-p",
                str(parsed.port or 5432),
                "-U",
                parsed.username or "postgres",
                "-d",
                parsed.path.lstrip("/") if parsed.path else "nemori",
                "-f",
                destination,
                "--no-password",
            ]

            # Set PGPASSWORD environment variable if password exists
            env = {}
            if parsed.password:
                env["PGPASSWORD"] = parsed.password

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup using pg_restore or psql."""
        try:
            import subprocess
            from urllib.parse import urlparse

            # Parse connection string
            parsed = urlparse(self.connection_string.replace("+asyncpg", ""))

            # Build psql command for SQL file restore
            cmd = [
                "psql",
                "-h",
                parsed.hostname or "localhost",
                "-p",
                str(parsed.port or 5432),
                "-U",
                parsed.username or "postgres",
                "-d",
                parsed.path.lstrip("/") if parsed.path else "nemori",
                "-f",
                source,
                "--no-password",
            ]

            # Set PGPASSWORD environment variable if password exists
            env = {}
            if parsed.password:
                env["PGPASSWORD"] = parsed.password

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode."""
        episode_id = self.validate_id(episode.episode_id)

        episode_row = EpisodeTable(
            episode_id=episode_id,
            owner_id=episode.owner_id,
            episode_type=episode.episode_type.value,
            level=episode.level.value,
            title=episode.title,
            content=episode.content,
            summary=episode.summary,
            timestamp=episode.temporal_info.timestamp,
            duration=episode.temporal_info.duration,
            timezone=episode.temporal_info.timezone,
            precision=episode.temporal_info.precision,
            event_metadata=json.dumps(episode.metadata.to_dict(), ensure_ascii=False),
            structured_data=json.dumps(episode.structured_data, ensure_ascii=False),
            search_keywords=json.dumps(episode.search_keywords, ensure_ascii=False),
            embedding_vector=(
                json.dumps(episode.embedding_vector, ensure_ascii=False) if episode.embedding_vector else None
            ),
            recall_count=episode.recall_count,
            importance_score=episode.importance_score,
            last_accessed=episode.last_accessed,
        )

        async with AsyncSession(self.engine) as session:
            session.add(episode_row)
            await session.commit()

        return episode_id

    async def store_episode_batch(self, episodes: list[Episode]) -> list[str]:
        """Store multiple episodes in batch."""
        episode_ids = []
        episode_rows = []

        for episode in episodes:
            episode_id = self.validate_id(episode.episode_id)
            episode_ids.append(episode_id)

            episode_rows.append(
                EpisodeTable(
                    episode_id=episode_id,
                    owner_id=episode.owner_id,
                    episode_type=episode.episode_type.value,
                    level=episode.level.value,
                    title=episode.title,
                    content=episode.content,
                    summary=episode.summary,
                    timestamp=episode.temporal_info.timestamp,
                    duration=episode.temporal_info.duration,
                    timezone=episode.temporal_info.timezone,
                    precision=episode.temporal_info.precision,
                    event_metadata=json.dumps(episode.metadata.to_dict(), ensure_ascii=False),
                    structured_data=json.dumps(episode.structured_data, ensure_ascii=False),
                    search_keywords=json.dumps(episode.search_keywords, ensure_ascii=False),
                    embedding_vector=(
                        json.dumps(episode.embedding_vector, ensure_ascii=False) if episode.embedding_vector else None
                    ),
                    recall_count=episode.recall_count,
                    importance_score=episode.importance_score,
                    last_accessed=episode.last_accessed,
                )
            )

        async with AsyncSession(self.engine) as session:
            session.add_all(episode_rows)
            await session.commit()

        return episode_ids

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID."""
        episode_id = self.validate_id(episode_id)

        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(EpisodeTable).where(EpisodeTable.episode_id == episode_id))
            episode_row = result.scalars().first()

            if not episode_row:
                return None

            return self._row_to_episode(episode_row)

    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """Retrieve multiple episodes by IDs."""
        if not episode_ids:
            return []

        # Validate all IDs
        validated_ids = [self.validate_id(episode_id) for episode_id in episode_ids]

        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(EpisodeTable).where(EpisodeTable.episode_id.in_(validated_ids)))
            episode_rows = result.scalars().all()

            # Create mapping for quick lookup
            result_map = {row.episode_id: self._row_to_episode(row) for row in episode_rows}

            # Return in the same order as requested
            return [result_map.get(episode_id) for episode_id in validated_ids]

    def _row_to_episode(self, row: EpisodeTable) -> Episode:
        """Convert database row to Episode."""
        from ..core.data_types import TemporalInfo
        from ..core.episode import EpisodeMetadata

        # Parse metadata
        metadata_dict = json.loads(row.event_metadata)
        metadata = EpisodeMetadata(
            source_data_ids=metadata_dict.get("source_data_ids", []),
            source_types={DataType(dt) for dt in metadata_dict.get("source_types", [])},
            processing_timestamp=datetime.fromisoformat(
                metadata_dict.get("processing_timestamp", datetime.now().isoformat())
            ),
            processing_version=metadata_dict.get("processing_version", "1.0"),
            entities=metadata_dict.get("entities", []),
            topics=metadata_dict.get("topics", []),
            emotions=metadata_dict.get("emotions", []),
            key_points=metadata_dict.get("key_points", []),
            time_references=metadata_dict.get("time_references", []),
            duration_seconds=metadata_dict.get("duration_seconds"),
            confidence_score=metadata_dict.get("confidence_score", 1.0),
            completeness_score=metadata_dict.get("completeness_score", 1.0),
            relevance_score=metadata_dict.get("relevance_score", 1.0),
            related_episode_ids=metadata_dict.get("related_episode_ids", []),
            custom_fields=metadata_dict.get("custom_fields", {}),
        )

        return Episode(
            episode_id=row.episode_id,
            owner_id=row.owner_id,
            episode_type=EpisodeType(row.episode_type),
            level=EpisodeLevel(row.level),
            title=row.title,
            content=row.content,
            summary=row.summary,
            temporal_info=TemporalInfo(
                timestamp=row.timestamp,
                duration=row.duration,
                timezone=row.timezone,
                precision=row.precision or "second",
            ),
            metadata=metadata,
            structured_data=json.loads(row.structured_data),
            search_keywords=json.loads(row.search_keywords),
            embedding_vector=json.loads(row.embedding_vector) if row.embedding_vector else None,
            recall_count=row.recall_count,
            importance_score=row.importance_score,
            last_accessed=row.last_accessed,
        )

    def _calculate_text_relevance(self, text: str, query_terms: list[str]) -> float:
        """Calculate text relevance score."""
        if not query_terms:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for term in query_terms if term.lower() in text_lower)
        return matches / len(query_terms)

    async def search_episodes(self, query: EpisodeQuery) -> EpisodeSearchResult:
        """Search episodes based on query parameters."""
        start_time = time.time()

        # Validate inputs
        query.limit, query.offset = self.validate_limit_offset(query.limit, query.offset)

        async with AsyncSession(self.engine) as session:
            # Build base query
            stmt = select(EpisodeTable)
            conditions = []

            # Apply filters
            if query.episode_ids:
                validated_ids = [self.validate_id(episode_id) for episode_id in query.episode_ids]
                conditions.append(EpisodeTable.episode_id.in_(validated_ids))

            if query.owner_ids:
                conditions.append(EpisodeTable.owner_id.in_(query.owner_ids))

            if query.episode_types:
                type_values = [et.value for et in query.episode_types]
                conditions.append(EpisodeTable.episode_type.in_(type_values))

            if query.levels:
                level_values = [level.value for level in query.levels]
                conditions.append(EpisodeTable.level.in_(level_values))

            if query.time_range:
                if query.time_range.start:
                    conditions.append(EpisodeTable.timestamp >= query.time_range.start)
                if query.time_range.end:
                    conditions.append(EpisodeTable.timestamp <= query.time_range.end)

            if query.recent_hours:
                time_threshold = datetime.now() - timedelta(hours=query.recent_hours)
                conditions.append(EpisodeTable.timestamp >= time_threshold)

            if query.min_importance is not None:
                conditions.append(EpisodeTable.importance_score >= query.min_importance)

            if query.max_importance is not None:
                conditions.append(EpisodeTable.importance_score <= query.max_importance)

            if query.min_recall_count is not None:
                conditions.append(EpisodeTable.recall_count >= query.min_recall_count)

            if query.max_recall_count is not None:
                conditions.append(EpisodeTable.recall_count <= query.max_recall_count)

            if query.text_search:
                search_term = self.sanitize_search_term(query.text_search)
                conditions.append(
                    or_(
                        EpisodeTable.title.contains(search_term),
                        EpisodeTable.content.contains(search_term),
                        EpisodeTable.summary.contains(search_term),
                    )
                )

            # Apply conditions
            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Count total results
            count_stmt = select(func.count(EpisodeTable.episode_id))
            if conditions:
                count_stmt = count_stmt.where(and_(*conditions))

            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()

            # Apply sorting
            if query.sort_by == SortBy.TIMESTAMP:
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(EpisodeTable.timestamp.desc())
                else:
                    stmt = stmt.order_by(EpisodeTable.timestamp.asc())
            elif query.sort_by == SortBy.IMPORTANCE:
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(EpisodeTable.importance_score.desc())
                else:
                    stmt = stmt.order_by(EpisodeTable.importance_score.asc())
            elif query.sort_by == SortBy.RECALL_COUNT:
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(EpisodeTable.recall_count.desc())
                else:
                    stmt = stmt.order_by(EpisodeTable.recall_count.asc())
            else:
                stmt = stmt.order_by(EpisodeTable.timestamp.desc())

            # Apply pagination
            if query.limit:
                stmt = stmt.limit(query.limit)
            if query.offset:
                stmt = stmt.offset(query.offset)

            # Execute query
            result = await session.execute(stmt)
            results = result.scalars().all()

            # Convert to Episode objects and calculate relevance
            episodes = []
            relevance_scores = []

            for row in results:
                episode = self._row_to_episode(row)
                episodes.append(episode)

                # Calculate relevance score
                relevance = 0.0

                if query.text_search:
                    combined_text = f"{episode.title} {episode.content} {episode.summary}"
                    relevance += self._calculate_text_relevance(combined_text, [query.text_search]) * 0.4

                if query.keywords:
                    combined_text = f"{episode.title} {episode.content} {episode.summary}"
                    relevance += self._calculate_text_relevance(combined_text, query.keywords) * 0.3

                if query.entities:
                    relevance += (
                        self._calculate_text_relevance(" ".join(episode.metadata.entities), query.entities) * 0.2
                    )

                if query.topics:
                    relevance += self._calculate_text_relevance(" ".join(episode.metadata.topics), query.topics) * 0.1

                # Default relevance based on importance and recency
                if relevance == 0.0:
                    relevance = episode.importance_score * 0.7 + (1.0 if episode.is_recent() else 0.0) * 0.3

                relevance_scores.append(relevance)

            # Calculate if there are more results
            has_more = False
            if query.limit and query.offset is not None:
                has_more = (query.offset + len(episodes)) < total_count
            elif query.limit:
                has_more = len(episodes) == query.limit and total_count > query.limit

            query_time_ms = (time.time() - start_time) * 1000

            return EpisodeSearchResult(
                episodes=episodes,
                total_count=total_count,
                has_more=has_more,
                query_time_ms=query_time_ms,
                relevance_scores=relevance_scores if any(s > 0 for s in relevance_scores) else None,
            )

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
        try:
            episode_id = self.validate_id(episode_id)

            async with AsyncSession(self.engine) as session:
                stmt = (
                    update(EpisodeTable)
                    .where(EpisodeTable.episode_id == episode_id)
                    .values(
                        owner_id=episode.owner_id,
                        episode_type=episode.episode_type.value,
                        level=episode.level.value,
                        title=episode.title,
                        content=episode.content,
                        summary=episode.summary,
                        timestamp=episode.temporal_info.timestamp,
                        duration=episode.temporal_info.duration,
                        timezone=episode.temporal_info.timezone,
                        precision=episode.temporal_info.precision,
                        event_metadata=json.dumps(episode.metadata.to_dict(), ensure_ascii=False),
                        structured_data=json.dumps(episode.structured_data, ensure_ascii=False),
                        search_keywords=json.dumps(episode.search_keywords, ensure_ascii=False),
                        embedding_vector=(
                            json.dumps(episode.embedding_vector, ensure_ascii=False)
                            if episode.embedding_vector
                            else None
                        ),
                        recall_count=episode.recall_count,
                        importance_score=episode.importance_score,
                        last_accessed=episode.last_accessed,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception:
            return False

    async def update_episode_importance(self, episode_id: str, importance_score: float) -> bool:
        """Update episode importance score."""
        try:
            episode_id = self.validate_id(episode_id)

            async with AsyncSession(self.engine) as session:
                stmt = (
                    update(EpisodeTable)
                    .where(EpisodeTable.episode_id == episode_id)
                    .values(importance_score=importance_score)
                )
                await session.execute(stmt)
                await session.commit()
                return True
        except Exception:
            return False

    async def mark_episode_accessed(self, episode_id: str) -> bool:
        """Mark an episode as accessed."""
        episode_id = self.validate_id(episode_id)

        async with AsyncSession(self.engine) as session:
            stmt = (
                update(EpisodeTable)
                .where(EpisodeTable.episode_id == episode_id)
                .values(recall_count=EpisodeTable.recall_count + 1, last_accessed=datetime.now())
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def link_episode_to_raw_data(self, episode_id: str, raw_data_ids: list[str]) -> bool:
        """Create association between episode and its source raw data."""
        try:
            episode_id = self.validate_id(episode_id)
            validated_raw_data_ids = [self.validate_id(raw_data_id) for raw_data_id in raw_data_ids]

            async with AsyncSession(self.engine) as session:
                link_rows = [
                    EpisodeRawDataTable(episode_id=episode_id, raw_data_id=raw_data_id)
                    for raw_data_id in validated_raw_data_ids
                ]
                session.add_all(link_rows)
                await session.commit()
                return True
        except Exception:
            return False

    async def get_episodes_for_raw_data(self, raw_data_id: str) -> list[Episode]:
        """Get all episodes that were created from specific raw data."""
        raw_data_id = self.validate_id(raw_data_id)

        async with AsyncSession(self.engine) as session:
            stmt = (
                select(EpisodeTable)
                .join(EpisodeRawDataTable, EpisodeTable.episode_id == EpisodeRawDataTable.episode_id)
                .where(EpisodeRawDataTable.raw_data_id == raw_data_id)
            )
            result = await session.execute(stmt)
            results = result.scalars().all()
            return [self._row_to_episode(row) for row in results]

    async def get_raw_data_for_episode(self, episode_id: str) -> list[str]:
        """Get raw data IDs that contributed to an episode."""
        episode_id = self.validate_id(episode_id)

        async with AsyncSession(self.engine) as session:
            stmt = select(EpisodeRawDataTable.raw_data_id).where(EpisodeRawDataTable.episode_id == episode_id)
            result = await session.execute(stmt)
            results = result.scalars().all()
            return results

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        try:
            episode_id = self.validate_id(episode_id)

            async with AsyncSession(self.engine) as session:
                # Delete related data first
                await session.execute(delete(EpisodeRawDataTable).where(EpisodeRawDataTable.episode_id == episode_id))

                # Delete the episode
                await session.execute(delete(EpisodeTable).where(EpisodeTable.episode_id == episode_id))
                await session.commit()
                return True
        except Exception:
            return False

    async def cleanup_old_episodes(self, max_age_days: int) -> int:
        """Clean up episodes older than specified age."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        async with AsyncSession(self.engine) as session:
            # Get old episode IDs
            result = await session.execute(select(EpisodeTable.episode_id).where(EpisodeTable.timestamp < cutoff_date))
            old_episode_ids = result.scalars().all()

            deleted_count = 0
            for episode_id in old_episode_ids:
                if await self.delete_episode(episode_id):
                    deleted_count += 1

            return deleted_count
