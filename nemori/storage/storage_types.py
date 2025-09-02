"""
Storage layer data types and query structures for Nemori.

This module defines the data structures used for storage configuration,
querying, and search results in the episodic memory system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.data_types import DataType, SemanticNode, SemanticRelationship
from ..core.episode import Episode, EpisodeLevel, EpisodeType


class SortOrder(Enum):
    """Sort order for search results."""

    ASC = "asc"
    DESC = "desc"


class SortBy(Enum):
    """Sort criteria for search results."""

    TIMESTAMP = "timestamp"
    IMPORTANCE = "importance"
    RECALL_COUNT = "recall_count"
    RELEVANCE = "relevance"


@dataclass(frozen=True)
class TimeRange:
    """Time range for temporal filtering."""

    start: datetime | None = None
    end: datetime | None = None

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this range."""
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True


@dataclass
class RawDataQuery:
    """Query parameters for raw event data search."""

    # Filtering
    data_ids: list[str] | None = None
    data_types: list[DataType] | None = None
    sources: list[str] | None = None
    time_range: TimeRange | None = None
    processed_only: bool | None = None

    # Text search
    content_contains: str | None = None
    metadata_filters: dict[str, Any] | None = None

    # Pagination
    limit: int | None = None
    offset: int | None = None

    # Sorting
    sort_by: SortBy = SortBy.TIMESTAMP
    sort_order: SortOrder = SortOrder.DESC


@dataclass
class EpisodeQuery:
    """Query parameters for episode search."""

    # Filtering
    episode_ids: list[str] | None = None
    owner_ids: list[str] | None = None
    episode_types: list[EpisodeType] | None = None
    levels: list[EpisodeLevel] | None = None
    time_range: TimeRange | None = None

    # Text search
    text_search: str | None = None  # Search in title, content, summary
    keywords: list[str] | None = None
    entities: list[str] | None = None
    topics: list[str] | None = None

    # Semantic search
    embedding_query: list[float] | None = None
    similarity_threshold: float | None = None

    # Importance and recall
    min_importance: float | None = None
    max_importance: float | None = None
    min_recall_count: int | None = None
    max_recall_count: int | None = None

    # Recent episodes
    recent_hours: int | None = None

    # Pagination
    limit: int | None = None
    offset: int | None = None

    # Sorting
    sort_by: SortBy = SortBy.TIMESTAMP
    sort_order: SortOrder = SortOrder.DESC


@dataclass
class RawDataSearchResult:
    """Result container for raw data queries."""

    data: list[Any]  # List of RawEventData objects
    total_count: int
    has_more: bool
    query_time_ms: float

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.data)


@dataclass
class EpisodeSearchResult:
    """Result container for episode queries."""

    episodes: list[Episode]
    total_count: int
    has_more: bool
    query_time_ms: float
    relevance_scores: list[float] | None = None

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.episodes)


@dataclass
class StorageStats:
    """Storage system statistics."""

    # Raw data stats
    total_raw_data: int = 0
    raw_data_by_type: dict[DataType, int] = field(default_factory=dict)
    processed_raw_data: int = 0

    # Episode stats
    total_episodes: int = 0
    episodes_by_type: dict[EpisodeType, int] = field(default_factory=dict)
    episodes_by_level: dict[EpisodeLevel, int] = field(default_factory=dict)

    # Storage stats
    storage_size_mb: float = 0.0
    index_size_mb: float = 0.0

    # Performance stats
    avg_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    # Temporal stats
    oldest_data: datetime | None = None
    newest_data: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_raw_data": self.total_raw_data,
            "raw_data_by_type": {dt.value: count for dt, count in self.raw_data_by_type.items()},
            "processed_raw_data": self.processed_raw_data,
            "total_episodes": self.total_episodes,
            "episodes_by_type": {et.value: count for et, count in self.episodes_by_type.items()},
            "episodes_by_level": {el.value: count for el, count in self.episodes_by_level.items()},
            "storage_size_mb": self.storage_size_mb,
            "index_size_mb": self.index_size_mb,
            "avg_query_time_ms": self.avg_query_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "oldest_data": self.oldest_data.isoformat() if self.oldest_data else None,
            "newest_data": self.newest_data.isoformat() if self.newest_data else None,
        }


@dataclass
class StorageConfig:
    """Configuration for storage layer."""

    # Storage backend
    backend_type: str = "memory"  # memory, duckdb, postgresql, etc.
    connection_string: str | None = None

    # Performance settings
    batch_size: int = 1000
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600

    # Indexing settings
    enable_full_text_search: bool = True
    enable_semantic_search: bool = True
    embedding_dimensions: int = 1536  # OpenAI embedding dimensions

    # Retention settings
    max_raw_data_age_days: int | None = None
    max_episode_age_days: int | None = None
    auto_cleanup_enabled: bool = False

    # Backup settings
    backup_enabled: bool = False
    backup_interval_hours: int = 24
    backup_retention_days: int = 30

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "backend_type": self.backend_type,
            "connection_string": self.connection_string,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "enable_full_text_search": self.enable_full_text_search,
            "enable_semantic_search": self.enable_semantic_search,
            "embedding_dimensions": self.embedding_dimensions,
            "max_raw_data_age_days": self.max_raw_data_age_days,
            "max_episode_age_days": self.max_episode_age_days,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "backup_enabled": self.backup_enabled,
            "backup_interval_hours": self.backup_interval_hours,
            "backup_retention_days": self.backup_retention_days,
        }


# === Semantic Memory Storage Types ===


@dataclass
class SemanticNodeQuery:
    """Query parameters for semantic node search."""

    # Owner filtering
    owner_id: str

    # Content filtering
    key_pattern: str | None = None
    value_pattern: str | None = None
    text_search: str | None = None  # Full-text search across key, value, context

    # Temporal filtering
    time_range: TimeRange | None = None
    created_after: datetime | None = None
    updated_after: datetime | None = None

    # Discovery filtering
    discovery_episode_id: str | None = None
    discovery_method: str | None = None

    # Confidence and quality filtering
    min_confidence: float | None = None
    min_importance: float | None = None
    min_access_count: int | None = None

    # Version filtering
    min_version: int | None = None
    max_version: int | None = None

    # Relationship filtering
    has_relationships: bool | None = None
    linked_to_episode: str | None = None

    # Search parameters
    similarity_query: str | None = None  # Semantic similarity search
    limit: int = 50
    offset: int = 0

    # Sorting
    sort_by: str = "created_at"  # created_at, updated_at, confidence, importance_score, access_count
    sort_order: SortOrder = SortOrder.DESC


@dataclass
class SemanticRelationshipQuery:
    """Query parameters for semantic relationship search."""

    # Node filtering
    source_node_id: str | None = None
    target_node_id: str | None = None
    involves_node_id: str | None = None  # Either source or target

    # Relationship filtering
    relationship_types: list[str] | None = None
    min_strength: float | None = None
    max_strength: float | None = None

    # Temporal filtering
    time_range: TimeRange | None = None
    created_after: datetime | None = None
    reinforced_after: datetime | None = None

    # Discovery filtering
    discovery_episode_id: str | None = None

    # Search parameters
    limit: int = 50
    offset: int = 0

    # Sorting
    sort_by: str = "created_at"  # created_at, last_reinforced, strength
    sort_order: SortOrder = SortOrder.DESC


@dataclass
class SemanticSearchResult:
    """Result from semantic memory search operations."""

    # Semantic nodes results
    semantic_nodes: list[SemanticNode] = field(default_factory=list)
    semantic_relationships: list[SemanticRelationship] = field(default_factory=list)

    # Result metadata
    total_nodes: int = 0
    total_relationships: int = 0
    has_more_nodes: bool = False
    has_more_relationships: bool = False

    # Performance metrics
    query_time_ms: float = 0.0

    # Search context
    query_info: dict[str, Any] = field(default_factory=dict)


# === Storage Exceptions ===


class SemanticStorageError(Exception):
    """Base exception for semantic storage operations."""

    pass


class DuplicateKeyError(SemanticStorageError):
    """Raised when attempting to store a semantic node with duplicate (owner_id, key)."""

    pass


class NotFoundError(SemanticStorageError):
    """Raised when attempting to operate on non-existent semantic data."""

    pass


class InvalidDataError(SemanticStorageError):
    """Raised when semantic data validation fails."""

    pass
