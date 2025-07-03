"""
Storage layer for Nemori episodic memory system.

This module provides abstract interfaces and implementations for storing
and retrieving raw event data, episodes, and their relationships.
"""

from .duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from .memory_storage import MemoryEpisodicMemoryRepository, MemoryRawDataRepository
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
