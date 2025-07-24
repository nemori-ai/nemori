"""
Storage layer for Nemori episodic memory system.

This module provides abstract interfaces and implementations for storing
and retrieving raw event data, episodes, and their relationships.
"""

from .duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from .factory import (
    StorageError,
    UnsupportedBackendError,
    create_duckdb_config,
    create_episodic_memory_repository,
    create_jsonl_config,
    create_memory_config,
    create_postgresql_config,
    create_raw_data_repository,
    create_repositories,
    get_supported_backends,
    validate_config,
)
from .jsonl_storage import JSONLEpisodicMemoryRepository, JSONLRawDataRepository
from .memory_storage import MemoryEpisodicMemoryRepository, MemoryRawDataRepository
from .postgresql_storage import PostgreSQLEpisodicMemoryRepository, PostgreSQLRawDataRepository
from .repository import EpisodicMemoryRepository, RawDataRepository, StorageRepository
from .storage_types import (
    EpisodeQuery,
    EpisodeSearchResult,
    RawDataQuery,
    RawDataSearchResult,
    StorageConfig,
    StorageStats,
    TimeRange,
)

__all__ = [
    # Repository interfaces
    "StorageRepository",
    "RawDataRepository",
    "EpisodicMemoryRepository",
    # Memory implementations
    "MemoryRawDataRepository",
    "MemoryEpisodicMemoryRepository",
    # DuckDB implementations
    "DuckDBRawDataRepository",
    "DuckDBEpisodicMemoryRepository",
    # PostgreSQL implementations
    "PostgreSQLRawDataRepository",
    "PostgreSQLEpisodicMemoryRepository",
    # JSONL implementations
    "JSONLRawDataRepository",
    "JSONLEpisodicMemoryRepository",
    # Factory functions
    "create_repositories",
    "create_raw_data_repository",
    "create_episodic_memory_repository",
    "create_postgresql_config",
    "create_duckdb_config",
    "create_memory_config",
    "create_jsonl_config",
    "get_supported_backends",
    "validate_config",
    # Exceptions
    "StorageError",
    "UnsupportedBackendError",
    # Query types
    "EpisodeQuery",
    "EpisodeSearchResult",
    "RawDataQuery",
    "RawDataSearchResult",
    "TimeRange",
    # Configuration
    "StorageConfig",
    "StorageStats",
]
