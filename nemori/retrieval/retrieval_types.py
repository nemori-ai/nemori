"""
Data types for the retrieval system in Nemori.

This module defines the core data structures used for retrieval queries,
results, and configuration across different retrieval providers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.episode import Episode


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    BM25 = "bm25"
    EMBEDDING = "embedding"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievalQuery:
    """
    Universal query structure for episode retrieval.

    Different providers can use different fields as appropriate for their strategy.
    """

    # Core query
    text: str
    owner_id: str

    # Query options
    limit: int = 10
    strategy: RetrievalStrategy = RetrievalStrategy.BM25

    # Filtering options (optional)
    episode_types: list[str] | None = None
    time_range_hours: int | None = None
    min_importance: float | None = None

    # Strategy-specific parameters
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "owner_id": self.owner_id,
            "limit": self.limit,
            "strategy": self.strategy.value,
            "episode_types": self.episode_types,
            "time_range_hours": self.time_range_hours,
            "min_importance": self.min_importance,
            "strategy_params": self.strategy_params,
        }


@dataclass
class RetrievalResult:
    """
    Unified result structure for episode retrieval.

    Contains episodes with relevance scores and metadata about the search.
    """

    episodes: list[Episode]
    scores: list[float]
    total_candidates: int
    query_time_ms: float
    strategy_used: RetrievalStrategy

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of episodes returned."""
        return len(self.episodes)

    @property
    def max_score(self) -> float:
        """Highest relevance score."""
        return max(self.scores) if self.scores else 0.0

    @property
    def min_score(self) -> float:
        """Lowest relevance score."""
        return min(self.scores) if self.scores else 0.0

    def get_episode_with_score(self, index: int) -> tuple[Episode, float]:
        """Get episode and its score by index."""
        return self.episodes[index], self.scores[index]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "episodes": [ep.to_dict() for ep in self.episodes],
            "scores": self.scores,
            "total_candidates": self.total_candidates,
            "query_time_ms": self.query_time_ms,
            "strategy_used": self.strategy_used.value,
            "count": self.count,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "metadata": self.metadata,
        }


class RetrievalStorageType(Enum):
    """Available storage types for retrieval indices."""

    MEMORY = "memory"  # In-memory only, no persistence
    DISK = "disk"  # Disk-based persistence (pickle files)
    DUCKDB = "duckdb"  # DuckDB database storage
    REDIS = "redis"  # Redis storage
    QDRANT = "qdrant"  # Vector database storage
    # Add more storage types as needed


@dataclass
class RetrievalConfig:
    """Configuration for retrieval providers."""

    # Storage configuration - each provider handles its own storage type
    storage_type: RetrievalStorageType = RetrievalStorageType.MEMORY
    storage_config: dict[str, Any] = field(default_factory=dict)

    # Performance settings
    batch_size: int = 1000
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600

    # Index settings
    auto_rebuild_threshold: int = 100  # Rebuild index after N new episodes
    rebuild_on_startup: bool = True

    # Quality settings
    min_score_threshold: float = 0.0
    max_results: int = 100
    api_key: str = ""
    base_url: str = ""
    embed_model: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "storage_type": self.storage_type.value,
            "storage_config": self.storage_config,
            "batch_size": self.batch_size,
            "cache_size": self.cache_size,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "auto_rebuild_threshold": self.auto_rebuild_threshold,
            "rebuild_on_startup": self.rebuild_on_startup,
            "min_score_threshold": self.min_score_threshold,
            "max_results": self.max_results,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "embed_model": self.embed_model
        }


@dataclass
class IndexStats:
    """Statistics about a retrieval index."""

    total_episodes: int = 0
    total_documents: int = 0  # For BM25, may differ from episodes due to tokenization
    index_size_mb: float = 0.0
    last_updated: datetime | None = None
    build_time_ms: float = 0.0

    # Provider-specific stats
    provider_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_episodes": self.total_episodes,
            "total_documents": self.total_documents,
            "index_size_mb": self.index_size_mb,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "build_time_ms": self.build_time_ms,
            "provider_stats": self.provider_stats,
        }
