"""
SQLModel-based DuckDB storage implementation for Nemori.

This module provides a refactored storage implementation using SQLModel
for better type safety and reduced SQL string concatenation.
"""

import json
import time
from datetime import datetime, timedelta
try:
    from datetime import UTC
except ImportError:
    from datetime import timedelta, timezone
    UTC = timezone.utc
from pathlib import Path
from typing import Any
from sqlmodel import Session, SQLModel, and_, create_engine, delete, func, or_, select, update

from ..core.data_types import DataType, RawEventData
from ..core.episode import Episode, EpisodeLevel, EpisodeType
from .repository import EpisodicMemoryRepository, SemanticMemoryRepository, RawDataRepository
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
from nemori.storage.storage_types import (
    DuplicateKeyError,
    InvalidDataError,
    NotFoundError,
    SemanticStorageError,
    SemanticNodeQuery,
    SemanticRelationshipQuery,
    SemanticSearchResult,
)
from nemori.core.data_types import RelationshipType, SemanticNode, SemanticRelationship
 
class DuckDBRawDataRepository(RawDataRepository, BaseSQLRepository):
    """DuckDB implementation of raw data repository using SQLModel."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.db_path = config.connection_string or "nemori_raw.duckdb"
        self.engine = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the DuckDB storage with SQLModel."""
        # Create database directory if needed
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"duckdb:///{self.db_path}")

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create tables
        SQLModel.metadata.create_all(self.engine)
        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            with Session(self.engine) as session:
                session.exec(select(1))
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        with Session(self.engine) as session:
            # Total raw data count
            stats.total_raw_data = session.exec(select(func.count(RawDataTable.data_id))).one()

            # Processed data count
            stats.processed_raw_data = session.exec(
                select(func.count(RawDataTable.data_id)).where(RawDataTable.processed)
            ).one()

            # Count by type
            type_results = session.exec(
                select(RawDataTable.data_type, func.count(RawDataTable.data_id)).group_by(RawDataTable.data_type)
            ).all()

            for data_type_str, count in type_results:
                try:
                    data_type = DataType(data_type_str)
                    stats.raw_data_by_type[data_type] = count
                except ValueError:
                    pass

            # Storage size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats.storage_size_mb = db_path.stat().st_size / (1024 * 1024)

            # Temporal stats
            temporal_result = session.exec(
                select(func.min(RawDataTable.timestamp), func.max(RawDataTable.timestamp))
            ).one()

            if temporal_result and temporal_result[0] and temporal_result[1]:
                stats.oldest_data = temporal_result[0]
                stats.newest_data = temporal_result[1]

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup by copying the database file."""
        try:
            import shutil

            # Dispose engine to ensure data is written
            if self.engine:
                self.engine.dispose()

            # Copy the database file
            shutil.copy2(self.db_path, destination)

            # Recreate engine
            self.engine = create_engine(f"duckdb:///{self.db_path}")
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup by copying the database file."""
        try:
            import shutil

            # Dispose current engine
            if self.engine:
                self.engine.dispose()

            # Copy backup file
            shutil.copy2(source, self.db_path)

            # Recreate engine and tables
            self.engine = create_engine(f"duckdb:///{self.db_path}")
            SQLModel.metadata.create_all(self.engine)
            self._initialized = True
            return True
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

        with Session(self.engine) as session:
            session.add(raw_data_row)
            session.commit()

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

        with Session(self.engine) as session:
            session.add_all(raw_data_rows)
            session.commit()

        return data_ids

    async def get_raw_data(self, data_id: str) -> RawEventData | None:
        """Retrieve raw event data by ID."""
        data_id = self.validate_id(data_id)

        with Session(self.engine) as session:
            raw_data_row = session.exec(select(RawDataTable).where(RawDataTable.data_id == data_id)).first()

            if not raw_data_row:
                return None

            return self._row_to_raw_data(raw_data_row)

    async def get_raw_data_batch(self, data_ids: list[str]) -> list[RawEventData | None]:
        """Retrieve multiple raw event data by IDs."""
        if not data_ids:
            return []

        # Validate all IDs
        validated_ids = [self.validate_id(data_id) for data_id in data_ids]

        with Session(self.engine) as session:
            raw_data_rows = session.exec(select(RawDataTable).where(RawDataTable.data_id.in_(validated_ids))).all()

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

        with Session(self.engine) as session:
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
            total_count = session.exec(count_stmt).one()

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
            results = session.exec(stmt).all()

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

            with Session(self.engine) as session:
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
                session.exec(stmt)
                session.commit()
                return True
        except Exception:
            return False

    async def mark_as_processed(self, data_id: str, processing_version: str) -> bool:
        """Mark raw data as processed."""
        try:
            data_id = self.validate_id(data_id)

            with Session(self.engine) as session:
                stmt = (
                    update(RawDataTable)
                    .where(RawDataTable.data_id == data_id)
                    .values(processed=True, processing_version=processing_version)
                )
                session.exec(stmt)
                session.commit()
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

            with Session(self.engine) as session:
                stmt = delete(RawDataTable).where(RawDataTable.data_id == data_id)
                session.exec(stmt)
                session.commit()
                return True
        except Exception:
            return False

    async def get_unprocessed_data(
        self, data_type: DataType | None = None, limit: int | None = None
    ) -> list[RawEventData]:
        """Get unprocessed raw event data."""
        limit, _ = self.validate_limit_offset(limit, None)

        with Session(self.engine) as session:
            stmt = select(RawDataTable).where(~RawDataTable.processed)

            if data_type:
                stmt = stmt.where(RawDataTable.data_type == data_type.value)

            stmt = stmt.order_by(RawDataTable.timestamp.asc())

            if limit:
                stmt = stmt.limit(limit)

            results = session.exec(stmt).all()
            return [self._row_to_raw_data(row) for row in results]


class DuckDBEpisodicMemoryRepository(EpisodicMemoryRepository, BaseSQLRepository):
    """DuckDB implementation of episodic memory repository using SQLModel."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.db_path = config.connection_string or "nemori_episodes.duckdb"
        self.engine = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the DuckDB storage with SQLModel."""
        # Create database directory if needed
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"duckdb:///{self.db_path}")

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create tables
        SQLModel.metadata.create_all(self.engine)
        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            with Session(self.engine) as session:
                session.exec(select(1))
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        stats = StorageStats()

        with Session(self.engine) as session:
            # Total episodes count
            stats.total_episodes = session.exec(select(func.count(EpisodeTable.episode_id))).one()

            # Count by type
            type_results = session.exec(
                select(EpisodeTable.episode_type, func.count(EpisodeTable.episode_id)).group_by(
                    EpisodeTable.episode_type
                )
            ).all()

            for episode_type_str, count in type_results:
                try:
                    episode_type = EpisodeType(episode_type_str)
                    stats.episodes_by_type[episode_type] = count
                except ValueError:
                    pass

            # Count by level
            level_results = session.exec(
                select(EpisodeTable.level, func.count(EpisodeTable.episode_id)).group_by(EpisodeTable.level)
            ).all()

            for level_int, count in level_results:
                try:
                    episode_level = EpisodeLevel(level_int)
                    stats.episodes_by_level[episode_level] = count
                except ValueError:
                    pass

            # Storage size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats.storage_size_mb = db_path.stat().st_size / (1024 * 1024)

            # Temporal stats
            temporal_result = session.exec(
                select(func.min(EpisodeTable.timestamp), func.max(EpisodeTable.timestamp))
            ).one()

            if temporal_result and temporal_result[0] and temporal_result[1]:
                stats.oldest_data = temporal_result[0]
                stats.newest_data = temporal_result[1]

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup by copying the database file."""
        try:
            import shutil

            # Dispose engine to ensure data is written
            if self.engine:
                self.engine.dispose()

            # Copy the database file
            shutil.copy2(self.db_path, destination)

            # Recreate engine
            self.engine = create_engine(f"duckdb:///{self.db_path}")
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup by copying the database file."""
        try:
            import shutil

            # Dispose current engine
            if self.engine:
                self.engine.dispose()

            # Copy backup file
            shutil.copy2(source, self.db_path)

            # Recreate engine and tables
            self.engine = create_engine(f"duckdb:///{self.db_path}")
            SQLModel.metadata.create_all(self.engine)
            self._initialized = True
            return True
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

        with Session(self.engine) as session:
            session.add(episode_row)
            session.commit()

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

        with Session(self.engine) as session:
            session.add_all(episode_rows)
            session.commit()

        return episode_ids

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID."""
        episode_id = self.validate_id(episode_id)

        with Session(self.engine) as session:
            episode_row = session.exec(select(EpisodeTable).where(EpisodeTable.episode_id == episode_id)).first()

            if not episode_row:
                return None

            return self._row_to_episode(episode_row)

    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """Retrieve multiple episodes by IDs."""
        if not episode_ids:
            return []

        # Validate all IDs
        validated_ids = [self.validate_id(episode_id) for episode_id in episode_ids]

        with Session(self.engine) as session:
            episode_rows = session.exec(select(EpisodeTable).where(EpisodeTable.episode_id.in_(validated_ids))).all()

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

        with Session(self.engine) as session:
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
            total_count = session.exec(count_stmt).one()

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
            results = session.exec(stmt).all()

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

            with Session(self.engine) as session:
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
                session.exec(stmt)
                session.commit()
                return True
        except Exception:
            return False

    async def update_episode_importance(self, episode_id: str, importance_score: float) -> bool:
        """Update episode importance score."""
        try:
            episode_id = self.validate_id(episode_id)

            with Session(self.engine) as session:
                stmt = (
                    update(EpisodeTable)
                    .where(EpisodeTable.episode_id == episode_id)
                    .values(importance_score=importance_score)
                )
                session.exec(stmt)
                session.commit()
                return True
        except Exception:
            return False

    async def mark_episode_accessed(self, episode_id: str) -> bool:
        """Mark an episode as accessed."""
        try:
            episode_id = self.validate_id(episode_id)

            with Session(self.engine) as session:
                stmt = (
                    update(EpisodeTable)
                    .where(EpisodeTable.episode_id == episode_id)
                    .values(recall_count=EpisodeTable.recall_count + 1, last_accessed=datetime.now(UTC))
                )
                session.exec(stmt)
                session.commit()
                return True
        except Exception:
            return False

    async def link_episode_to_raw_data(self, episode_id: str, raw_data_ids: list[str]) -> bool:
        """Create association between episode and its source raw data."""
        try:
            episode_id = self.validate_id(episode_id)
            validated_raw_data_ids = [self.validate_id(raw_data_id) for raw_data_id in raw_data_ids]

            with Session(self.engine) as session:
                link_rows = [
                    EpisodeRawDataTable(episode_id=episode_id, raw_data_id=raw_data_id)
                    for raw_data_id in validated_raw_data_ids
                ]
                session.add_all(link_rows)
                session.commit()
                return True
        except Exception:
            return False

    async def get_episodes_for_raw_data(self, raw_data_id: str) -> list[Episode]:
        """Get all episodes that were created from specific raw data."""
        raw_data_id = self.validate_id(raw_data_id)

        with Session(self.engine) as session:
            stmt = (
                select(EpisodeTable)
                .join(EpisodeRawDataTable, EpisodeTable.episode_id == EpisodeRawDataTable.episode_id)
                .where(EpisodeRawDataTable.raw_data_id == raw_data_id)
            )
            results = session.exec(stmt).all()
            return [self._row_to_episode(row) for row in results]

    async def get_raw_data_for_episode(self, episode_id: str) -> list[str]:
        """Get raw data IDs that contributed to an episode."""
        episode_id = self.validate_id(episode_id)

        with Session(self.engine) as session:
            stmt = select(EpisodeRawDataTable.raw_data_id).where(EpisodeRawDataTable.episode_id == episode_id)
            results = session.exec(stmt).all()
            return results

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode."""
        try:
            episode_id = self.validate_id(episode_id)

            with Session(self.engine) as session:
                # Delete related data first
                session.exec(delete(EpisodeRawDataTable).where(EpisodeRawDataTable.episode_id == episode_id))

                # Delete the episode
                session.exec(delete(EpisodeTable).where(EpisodeTable.episode_id == episode_id))
                session.commit()
                return True
        except Exception:
            return False

    async def cleanup_old_episodes(self, max_age_days: int) -> int:
        """Clean up episodes older than specified age."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        with Session(self.engine) as session:
            # Get old episode IDs
            old_episode_ids = session.exec(
                select(EpisodeTable.episode_id).where(EpisodeTable.timestamp < cutoff_date)
            ).all()

            deleted_count = 0
            for episode_id in old_episode_ids:
                if await self.delete_episode(episode_id):
                    deleted_count += 1

            return deleted_count


class DuckDBSemanticMemoryRepository(SemanticMemoryRepository, BaseSQLRepository):
    """DuckDB implementation of semantic memory repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.engine = create_engine(
            config.connection_string or f"duckdb:///{config.connection_string or 'nemori_semantic.duckdb'}",
            echo=False,
        )

    async def initialize(self) -> None:
        """Initialize semantic memory tables."""
        from .sql_models import SemanticNodeTable, SemanticRelationshipTable

        # Create tables
        SQLModel.metadata.create_all(self.engine)

    async def close(self) -> None:
        """Close the semantic memory storage."""
        self.engine.dispose()

    async def health_check(self) -> bool:
        """Check if semantic memory storage is healthy."""
        try:
            with Session(self.engine) as session:
                from .sql_models import SemanticNodeTable

                session.exec(select(func.count(SemanticNodeTable.node_id))).one()
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get semantic memory statistics."""
        with Session(self.engine) as session:
            from .sql_models import SemanticNodeTable, SemanticRelationshipTable

            node_count = session.exec(select(func.count(SemanticNodeTable.node_id))).one()
            relationship_count = session.exec(select(func.count(SemanticRelationshipTable.relationship_id))).one()

            return StorageStats(
                # Add semantic-specific stats
                total_raw_data=node_count,  # Using raw_data field for semantic nodes
                total_episodes=relationship_count,  # Using episodes field for relationships
            )

    async def backup(self, destination: str) -> bool:
        """Create backup of semantic memory."""
        try:
            # Simple backup approach for DuckDB
            source_path = Path(self.config.connection_string.replace("duckdb:///", ""))
            destination_path = Path(destination)
            
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, destination_path)
                return True
            return False
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore semantic memory from backup."""
        try:
            source_path = Path(source)
            destination_path = Path(self.config.connection_string.replace("duckdb:///", ""))
            
            if source_path.exists():
                import shutil
                shutil.copy2(source_path, destination_path)
                # Reinitialize engine
                self.engine.dispose()
                self.engine = create_engine(self.config.connection_string, echo=False)
                return True
            return False
        except Exception:
            return False

    async def search_semantic_nodes(self, query: SemanticNodeQuery) -> SemanticSearchResult:
        """Search semantic nodes with complex query parameters."""
        from .sql_models import SemanticNodeTable

        start_time = time.time()
        
        with Session(self.engine) as session:
            stmt = select(SemanticNodeTable).where(SemanticNodeTable.owner_id == query.owner_id)
            conditions = []

            # Apply filters
            if query.key_pattern:
                conditions.append(SemanticNodeTable.key.contains(query.key_pattern))
            
            if query.value_pattern:
                conditions.append(SemanticNodeTable.value.contains(query.value_pattern))
            
            if query.text_search:
                search_term = self.sanitize_search_term(query.text_search)
                conditions.append(
                    or_(
                        SemanticNodeTable.key.contains(search_term),
                        SemanticNodeTable.value.contains(search_term),
                        SemanticNodeTable.context.contains(search_term)
                    )
                )
            
            if query.min_confidence:
                conditions.append(SemanticNodeTable.confidence >= query.min_confidence)
            
            if query.min_importance:
                conditions.append(SemanticNodeTable.importance_score >= query.min_importance)
            
            if query.discovery_episode_id:
                conditions.append(SemanticNodeTable.discovery_episode_id == query.discovery_episode_id)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Count total results
            count_stmt = select(func.count(SemanticNodeTable.node_id)).where(
                SemanticNodeTable.owner_id == query.owner_id
            )
            if conditions:
                count_stmt = count_stmt.where(and_(*conditions))
            total_nodes = session.exec(count_stmt).one()

            # Apply sorting
            if query.sort_by == "confidence":
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(SemanticNodeTable.confidence.desc())
                else:
                    stmt = stmt.order_by(SemanticNodeTable.confidence.asc())
            elif query.sort_by == "importance_score":
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(SemanticNodeTable.importance_score.desc())
                else:
                    stmt = stmt.order_by(SemanticNodeTable.importance_score.asc())
            else:  # Default to created_at
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(SemanticNodeTable.created_at.desc())
                else:
                    stmt = stmt.order_by(SemanticNodeTable.created_at.asc())

            # Apply pagination
            if query.limit:
                stmt = stmt.limit(query.limit)
            if query.offset:
                stmt = stmt.offset(query.offset)

            # Execute query
            results = session.exec(stmt).all()
            semantic_nodes = [self._row_to_semantic_node(row) for row in results]

            query_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchResult(
                semantic_nodes=semantic_nodes,
                total_nodes=total_nodes,
                has_more_nodes=(len(semantic_nodes) == query.limit) if query.limit else False,
                query_time_ms=query_time_ms,
            )

    async def search_semantic_relationships(self, query: SemanticRelationshipQuery) -> SemanticSearchResult:
        """Search semantic relationships with complex query parameters."""
        from .sql_models import SemanticRelationshipTable

        start_time = time.time()
        
        with Session(self.engine) as session:
            stmt = select(SemanticRelationshipTable)
            conditions = []

            # Apply filters
            if query.source_node_id:
                conditions.append(SemanticRelationshipTable.source_node_id == query.source_node_id)
            
            if query.target_node_id:
                conditions.append(SemanticRelationshipTable.target_node_id == query.target_node_id)
            
            if query.involves_node_id:
                conditions.append(
                    or_(
                        SemanticRelationshipTable.source_node_id == query.involves_node_id,
                        SemanticRelationshipTable.target_node_id == query.involves_node_id
                    )
                )
            
            if query.relationship_types:
                conditions.append(SemanticRelationshipTable.relationship_type.in_(query.relationship_types))
            
            if query.min_strength:
                conditions.append(SemanticRelationshipTable.strength >= query.min_strength)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Count total results
            count_stmt = select(func.count(SemanticRelationshipTable.relationship_id))
            if conditions:
                count_stmt = count_stmt.where(and_(*conditions))
            total_relationships = session.exec(count_stmt).one()

            # Apply sorting
            if query.sort_by == "strength":
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(SemanticRelationshipTable.strength.desc())
                else:
                    stmt = stmt.order_by(SemanticRelationshipTable.strength.asc())
            else:  # Default to created_at
                if query.sort_order == SortOrder.DESC:
                    stmt = stmt.order_by(SemanticRelationshipTable.created_at.desc())
                else:
                    stmt = stmt.order_by(SemanticRelationshipTable.created_at.asc())

            # Apply pagination
            if query.limit:
                stmt = stmt.limit(query.limit)
            if query.offset:
                stmt = stmt.offset(query.offset)

            # Execute query
            results = session.exec(stmt).all()
            semantic_relationships = [self._row_to_semantic_relationship(row) for row in results]

            query_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchResult(
                semantic_relationships=semantic_relationships,
                total_relationships=total_relationships,
                has_more_relationships=(len(semantic_relationships) == query.limit) if query.limit else False,
                query_time_ms=query_time_ms,
            )

    async def store_semantic_node(self, node: SemanticNode) -> None:
        """Store a semantic node."""
        try:
            from .sql_models import SemanticNodeTable

            node_id = self.validate_id(node.node_id)
            
            with Session(self.engine) as session:
                # Check for duplicate key
                existing = session.exec(
                    select(SemanticNodeTable).where(
                        and_(
                            SemanticNodeTable.owner_id == node.owner_id,
                            SemanticNodeTable.key == node.key
                        )
                    )
                ).first()
                
                if existing:
                    raise DuplicateKeyError(f"Node with key {node.key} already exists for owner {node.owner_id}")

                # Convert domain object to table row
                row = SemanticNodeTable(
                    node_id=node.node_id,
                    owner_id=node.owner_id,
                    key=node.key,
                    value=node.value,
                    context=node.context,
                    confidence=node.confidence,
                    version=node.version,
                    evolution_history=json.dumps(node.evolution_history, ensure_ascii=False),
                    created_at=node.created_at,
                    last_updated=node.last_updated,
                    last_accessed=node.last_accessed,
                    discovery_episode_id=node.discovery_episode_id,
                    discovery_method=node.discovery_method,
                    linked_episode_ids=json.dumps(node.linked_episode_ids, ensure_ascii=False),
                    evolution_episode_ids=json.dumps(node.evolution_episode_ids, ensure_ascii=False),
                    search_keywords=json.dumps(node.search_keywords, ensure_ascii=False),
                    embedding_vector=json.dumps(node.embedding_vector, ensure_ascii=False) if node.embedding_vector else None,
                    access_count=node.access_count,
                    relevance_score=node.relevance_score,
                    importance_score=node.importance_score,
                )
                
                session.add(row)
                session.commit()

        except DuplicateKeyError:
            raise
        except Exception as e:
            raise SemanticStorageError(f"Failed to store semantic node: {e}")

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        """Retrieve a semantic node by its ID."""
        try:
            from .sql_models import SemanticNodeTable

            node_id = self.validate_id(node_id)
            
            with Session(self.engine) as session:
                row = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id)
                ).first()
                
                return self._row_to_semantic_node(row) if row else None

        except Exception:
            return None

    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        """Find semantic node by owner and key combination."""
        try:
            from .sql_models import SemanticNodeTable

            with Session(self.engine) as session:
                row = session.exec(
                    select(SemanticNodeTable).where(
                        and_(
                            SemanticNodeTable.owner_id == owner_id,
                            SemanticNodeTable.key == key
                        )
                    )
                ).first()
                
                return self._row_to_semantic_node(row) if row else None

        except Exception:
            return None

    async def update_semantic_node(self, node: SemanticNode) -> None:
        """Update an existing semantic node."""
        try:
            from .sql_models import SemanticNodeTable

            node_id = self.validate_id(node.node_id)
            
            with Session(self.engine) as session:
                # Check if node exists
                existing = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id)
                ).first()
                
                if not existing:
                    raise NotFoundError(f"Node {node_id} not found")

                # Update the row
                stmt = (
                    update(SemanticNodeTable)
                    .where(SemanticNodeTable.node_id == node_id)
                    .values(
                        key=node.key,
                        value=node.value,
                        context=node.context,
                        confidence=node.confidence,
                        version=node.version,
                        evolution_history=json.dumps(node.evolution_history, ensure_ascii=False),
                        last_updated=node.last_updated,
                        last_accessed=node.last_accessed,
                        linked_episode_ids=json.dumps(node.linked_episode_ids, ensure_ascii=False),
                        evolution_episode_ids=json.dumps(node.evolution_episode_ids, ensure_ascii=False),
                        search_keywords=json.dumps(node.search_keywords, ensure_ascii=False),
                        embedding_vector=json.dumps(node.embedding_vector, ensure_ascii=False) if node.embedding_vector else None,
                        access_count=node.access_count,
                        relevance_score=node.relevance_score,
                        importance_score=node.importance_score,
                    )
                )
                
                session.exec(stmt)
                session.commit()

        except (NotFoundError, SemanticStorageError):
            raise
        except Exception as e:
            raise SemanticStorageError(f"Failed to update semantic node: {e}")

    async def delete_semantic_node(self, node_id: str) -> bool:
        """Delete a semantic node by ID."""
        try:
            from .sql_models import SemanticNodeTable

            node_id = self.validate_id(node_id)
            
            with Session(self.engine) as session:
                stmt = delete(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id)
                result = session.exec(stmt)
                session.commit()
                
                return result.rowcount > 0

        except Exception:
            return False

    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        """Find all semantic nodes discovered from a specific episode."""
        try:
            from .sql_models import SemanticNodeTable

            episode_id = self.validate_id(episode_id)
            
            with Session(self.engine) as session:
                rows = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.discovery_episode_id == episode_id)
                ).all()
                
                return [self._row_to_semantic_node(row) for row in rows]

        except Exception:
            return []

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        """Find all semantic nodes that have the episode in their linked_episode_ids."""
        try:
            from .sql_models import SemanticNodeTable

            episode_id = self.validate_id(episode_id)
            
            with Session(self.engine) as session:
                # For DuckDB, we'll use a simple contains query
                rows = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.linked_episode_ids.contains(episode_id))
                ).all()
                
                # Filter to exact matches (since contains is not perfect for JSON arrays)
                result = []
                for row in rows:
                    node = self._row_to_semantic_node(row)
                    if episode_id in node.linked_episode_ids:
                        result.append(node)
                
                return result

        except Exception:
            return []

    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """Search semantic nodes by similarity to query text."""
        try:
            from .sql_models import SemanticNodeTable

            query_term = self.sanitize_search_term(query)
            
            with Session(self.engine) as session:
                # Simple text-based similarity search
                stmt = (
                    select(SemanticNodeTable)
                    .where(
                        and_(
                            SemanticNodeTable.owner_id == owner_id,
                            or_(
                                SemanticNodeTable.key.contains(query_term),
                                SemanticNodeTable.value.contains(query_term),
                                SemanticNodeTable.context.contains(query_term)
                            )
                        )
                    )
                    .order_by(SemanticNodeTable.importance_score.desc(), SemanticNodeTable.confidence.desc())
                    .limit(limit)
                )
                
                rows = session.exec(stmt).all()
                return [self._row_to_semantic_node(row) for row in rows]

        except Exception:
            return []

    async def store_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """Store a semantic relationship."""
        try:
            from .sql_models import SemanticRelationshipTable

            relationship_id = self.validate_id(relationship.relationship_id)
            
            with Session(self.engine) as session:
                # Convert domain object to table row
                row = SemanticRelationshipTable(
                    relationship_id=relationship.relationship_id,
                    source_node_id=relationship.source_node_id,
                    target_node_id=relationship.target_node_id,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    description=relationship.description,
                    created_at=relationship.created_at,
                    last_reinforced=relationship.last_reinforced,
                    discovery_episode_id=relationship.discovery_episode_id,
                )
                
                session.add(row)
                session.commit()

        except Exception as e:
            raise SemanticStorageError(f"Failed to store semantic relationship: {e}")

    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        """Retrieve a semantic relationship by its ID."""
        try:
            from .sql_models import SemanticRelationshipTable

            relationship_id = self.validate_id(relationship_id)
            
            with Session(self.engine) as session:
                row = session.exec(
                    select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
                ).first()
                
                return self._row_to_semantic_relationship(row) if row else None

        except Exception:
            return None

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """Find all relationships and related nodes for a given semantic node."""
        try:
            from .sql_models import SemanticRelationshipTable, SemanticNodeTable

            node_id = self.validate_id(node_id)
            
            with Session(self.engine) as session:
                # Find relationships where this node is either source or target
                relationships = session.exec(
                    select(SemanticRelationshipTable).where(
                        or_(
                            SemanticRelationshipTable.source_node_id == node_id,
                            SemanticRelationshipTable.target_node_id == node_id
                        )
                    )
                ).all()
                
                result = []
                for rel_row in relationships:
                    relationship = self._row_to_semantic_relationship(rel_row)
                    
                    # Find the related node (not the current node)
                    related_node_id = (
                        rel_row.target_node_id if rel_row.source_node_id == node_id 
                        else rel_row.source_node_id
                    )
                    
                    related_node_row = session.exec(
                        select(SemanticNodeTable).where(SemanticNodeTable.node_id == related_node_id)
                    ).first()
                    
                    if related_node_row:
                        related_node = self._row_to_semantic_node(related_node_row)
                        result.append((related_node, relationship))
                
                return result

        except Exception:
            return []

    async def update_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """Update an existing semantic relationship."""
        try:
            from .sql_models import SemanticRelationshipTable

            relationship_id = self.validate_id(relationship.relationship_id)
            
            with Session(self.engine) as session:
                # Check if relationship exists
                existing = session.exec(
                    select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
                ).first()
                
                if not existing:
                    raise NotFoundError(f"Relationship {relationship_id} not found")

                # Update the row
                stmt = (
                    update(SemanticRelationshipTable)
                    .where(SemanticRelationshipTable.relationship_id == relationship_id)
                    .values(
                        relationship_type=relationship.relationship_type.value,
                        strength=relationship.strength,
                        description=relationship.description,
                        last_reinforced=relationship.last_reinforced,
                    )
                )
                
                session.exec(stmt)
                session.commit()

        except (NotFoundError, SemanticStorageError):
            raise
        except Exception as e:
            raise SemanticStorageError(f"Failed to update semantic relationship: {e}")

    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        """Delete a semantic relationship by ID."""
        try:
            from .sql_models import SemanticRelationshipTable

            relationship_id = self.validate_id(relationship_id)
            
            with Session(self.engine) as session:
                stmt = delete(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
                result = session.exec(stmt)
                session.commit()
                
                return result.rowcount > 0

        except Exception:
            return False

    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        """Retrieve multiple semantic nodes by their IDs."""
        try:
            from .sql_models import SemanticNodeTable

            validated_ids = [self.validate_id(node_id) for node_id in node_ids]
            
            with Session(self.engine) as session:
                rows = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.node_id.in_(validated_ids))
                ).all()
                
                return [self._row_to_semantic_node(row) for row in rows]

        except Exception:
            return []

    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        """Retrieve all semantic nodes for a specific owner."""
        try:
            from .sql_models import SemanticNodeTable

            with Session(self.engine) as session:
                rows = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.owner_id == owner_id)
                ).all()
                
                return [self._row_to_semantic_node(row) for row in rows]

        except Exception:
            return []

    async def get_semantic_statistics(self, owner_id: str) -> dict[str, any]:
        """Get statistics about semantic memory for an owner."""
        try:
            from .sql_models import SemanticNodeTable, SemanticRelationshipTable

            with Session(self.engine) as session:
                # Count nodes for this owner
                node_count = session.exec(
                    select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == owner_id)
                ).one()
                
                # Get average confidence
                avg_confidence = session.exec(
                    select(func.avg(SemanticNodeTable.confidence)).where(SemanticNodeTable.owner_id == owner_id)
                ).one() or 0.0
                
                # Count relationships involving this owner's nodes
                relationship_count = session.exec(
                    select(func.count(SemanticRelationshipTable.relationship_id))
                    .select_from(
                        SemanticRelationshipTable.join(
                            SemanticNodeTable,
                            or_(
                                SemanticRelationshipTable.source_node_id == SemanticNodeTable.node_id,
                                SemanticRelationshipTable.target_node_id == SemanticNodeTable.node_id
                            )
                        )
                    )
                    .where(SemanticNodeTable.owner_id == owner_id)
                ).one()
                
                # Total access count
                total_access_count = session.exec(
                    select(func.sum(SemanticNodeTable.access_count)).where(SemanticNodeTable.owner_id == owner_id)
                ).one() or 0

                return {
                    "total_nodes": node_count,
                    "total_relationships": relationship_count,
                    "average_confidence": float(avg_confidence),
                    "total_access_count": int(total_access_count),
                }

        except Exception:
            return {
                "total_nodes": 0,
                "total_relationships": 0,
                "average_confidence": 0.0,
                "total_access_count": 0,
            }

    async def cleanup_orphaned_relationships(self) -> int:
        """Clean up relationships that reference non-existent nodes."""
        try:
            from .sql_models import SemanticRelationshipTable, SemanticNodeTable

            with Session(self.engine) as session:
                # Find orphaned relationships
                orphaned_stmt = (
                    select(SemanticRelationshipTable.relationship_id)
                    .outerjoin(
                        SemanticNodeTable,
                        SemanticRelationshipTable.source_node_id == SemanticNodeTable.node_id,
                        full=False
                    )
                    .where(SemanticNodeTable.node_id.is_(None))
                )
                
                orphaned_ids = session.exec(orphaned_stmt).all()
                
                if orphaned_ids:
                    delete_stmt = delete(SemanticRelationshipTable).where(
                        SemanticRelationshipTable.relationship_id.in_(orphaned_ids)
                    )
                    session.exec(delete_stmt)
                    session.commit()
                
                return len(orphaned_ids)

        except Exception:
            return 0

    def _row_to_semantic_node(self, row) -> SemanticNode:
        """Convert database row to SemanticNode domain object."""
        return SemanticNode(
            node_id=row.node_id,
            owner_id=row.owner_id,
            key=row.key,
            value=row.value,
            context=row.context or "",
            confidence=row.confidence,
            version=row.version,
            evolution_history=json.loads(row.evolution_history) if row.evolution_history else [],
            created_at=row.created_at,
            last_updated=row.last_updated,
            last_accessed=row.last_accessed,
            discovery_episode_id=row.discovery_episode_id,
            discovery_method=row.discovery_method,
            linked_episode_ids=json.loads(row.linked_episode_ids) if row.linked_episode_ids else [],
            evolution_episode_ids=json.loads(row.evolution_episode_ids) if row.evolution_episode_ids else [],
            search_keywords=json.loads(row.search_keywords) if row.search_keywords else [],
            embedding_vector=json.loads(row.embedding_vector) if row.embedding_vector else None,
            access_count=row.access_count,
            relevance_score=row.relevance_score,
            importance_score=row.importance_score,
        )

    def _row_to_semantic_relationship(self, row) -> SemanticRelationship:
        """Convert database row to SemanticRelationship domain object."""
        from nemori.core.data_types import RelationshipType
        
        return SemanticRelationship(
            relationship_id=row.relationship_id,
            source_node_id=row.source_node_id,
            target_node_id=row.target_node_id,
            relationship_type=RelationshipType(row.relationship_type),
            strength=row.strength,
            description=row.description or "",
            created_at=row.created_at,
            last_reinforced=row.last_reinforced,
            discovery_episode_id=row.discovery_episode_id,
        )