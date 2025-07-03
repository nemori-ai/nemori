"""
Tests for retrieval data types in Nemori.

This module contains tests for the retrieval type system including
RetrievalQuery, RetrievalResult, RetrievalConfig, and IndexStats.
"""

from datetime import datetime

import pytest

from nemori.core.episode import Episode, EpisodeLevel, EpisodeType
from nemori.retrieval.retrieval_types import (
    IndexStats,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStorageType,
    RetrievalStrategy,
)


class TestRetrievalStrategy:
    """Test cases for RetrievalStrategy enum."""

    def test_strategy_values(self):
        """Test that strategy enum has expected values."""
        assert RetrievalStrategy.BM25.value == "bm25"
        assert RetrievalStrategy.EMBEDDING.value == "embedding"
        assert RetrievalStrategy.KEYWORD.value == "keyword"
        assert RetrievalStrategy.HYBRID.value == "hybrid"


class TestRetrievalStorageType:
    """Test cases for RetrievalStorageType enum."""

    def test_storage_type_values(self):
        """Test that storage type enum has expected values."""
        assert RetrievalStorageType.MEMORY.value == "memory"
        assert RetrievalStorageType.DISK.value == "disk"
        assert RetrievalStorageType.DUCKDB.value == "duckdb"
        assert RetrievalStorageType.REDIS.value == "redis"
        assert RetrievalStorageType.QDRANT.value == "qdrant"


class TestRetrievalQuery:
    """Test cases for RetrievalQuery."""

    def test_query_creation_with_defaults(self):
        """Test creating a query with default values."""
        query = RetrievalQuery(text="test query", owner_id="user123")

        assert query.text == "test query"
        assert query.owner_id == "user123"
        assert query.limit == 10
        assert query.strategy == RetrievalStrategy.BM25
        assert query.episode_types is None
        assert query.time_range_hours is None
        assert query.min_importance is None
        assert query.strategy_params == {}

    def test_query_creation_with_custom_values(self):
        """Test creating a query with custom values."""
        query = RetrievalQuery(
            text="machine learning",
            owner_id="user456",
            limit=20,
            strategy=RetrievalStrategy.EMBEDDING,
            episode_types=["conversational", "behavioral"],
            time_range_hours=24,
            min_importance=0.7,
            strategy_params={"threshold": 0.8},
        )

        assert query.text == "machine learning"
        assert query.owner_id == "user456"
        assert query.limit == 20
        assert query.strategy == RetrievalStrategy.EMBEDDING
        assert query.episode_types == ["conversational", "behavioral"]
        assert query.time_range_hours == 24
        assert query.min_importance == 0.7
        assert query.strategy_params == {"threshold": 0.8}

    def test_query_to_dict(self):
        """Test converting query to dictionary."""
        query = RetrievalQuery(
            text="test",
            owner_id="user123",
            limit=5,
            strategy=RetrievalStrategy.BM25,
            episode_types=["conversational"],
            min_importance=0.5,
            strategy_params={"param1": "value1"},
        )

        query_dict = query.to_dict()

        expected = {
            "text": "test",
            "owner_id": "user123",
            "limit": 5,
            "strategy": "bm25",
            "episode_types": ["conversational"],
            "time_range_hours": None,
            "min_importance": 0.5,
            "strategy_params": {"param1": "value1"},
        }

        assert query_dict == expected


class TestRetrievalResult:
    """Test cases for RetrievalResult."""

    @pytest.fixture
    def sample_episodes(self):
        """Create sample episodes for testing."""
        return [
            Episode(
                episode_id="ep1",
                owner_id="user123",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title="First Episode",
                content="Content of first episode",
            ),
            Episode(
                episode_id="ep2",
                owner_id="user123",
                episode_type=EpisodeType.BEHAVIORAL,
                level=EpisodeLevel.COMPOUND,
                title="Second Episode",
                content="Content of second episode",
            ),
        ]

    def test_result_creation(self, sample_episodes):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            episodes=sample_episodes,
            scores=[0.9, 0.7],
            total_candidates=5,
            query_time_ms=15.5,
            strategy_used=RetrievalStrategy.BM25,
        )

        assert len(result.episodes) == 2
        assert result.scores == [0.9, 0.7]
        assert result.total_candidates == 5
        assert result.query_time_ms == 15.5
        assert result.strategy_used == RetrievalStrategy.BM25
        assert result.metadata == {}

    def test_result_properties(self, sample_episodes):
        """Test result computed properties."""
        result = RetrievalResult(
            episodes=sample_episodes,
            scores=[0.9, 0.7, 0.3],
            total_candidates=10,
            query_time_ms=25.0,
            strategy_used=RetrievalStrategy.BM25,
        )

        assert result.count == 2  # Episodes count
        assert result.max_score == 0.9
        assert result.min_score == 0.3

    def test_result_empty_scores(self):
        """Test result properties with empty scores."""
        result = RetrievalResult(
            episodes=[], scores=[], total_candidates=0, query_time_ms=10.0, strategy_used=RetrievalStrategy.BM25
        )

        assert result.count == 0
        assert result.max_score == 0.0
        assert result.min_score == 0.0

    def test_get_episode_with_score(self, sample_episodes):
        """Test getting episode with its score."""
        result = RetrievalResult(
            episodes=sample_episodes,
            scores=[0.9, 0.7],
            total_candidates=2,
            query_time_ms=10.0,
            strategy_used=RetrievalStrategy.BM25,
        )

        episode, score = result.get_episode_with_score(0)
        assert episode.episode_id == "ep1"
        assert score == 0.9

        episode, score = result.get_episode_with_score(1)
        assert episode.episode_id == "ep2"
        assert score == 0.7

    def test_result_to_dict(self, sample_episodes):
        """Test converting result to dictionary."""
        result = RetrievalResult(
            episodes=sample_episodes[:1],  # Just first episode
            scores=[0.9],
            total_candidates=3,
            query_time_ms=12.5,
            strategy_used=RetrievalStrategy.BM25,
            metadata={"query_tokens": ["test"]},
        )

        result_dict = result.to_dict()

        assert "episodes" in result_dict
        assert len(result_dict["episodes"]) == 1
        assert result_dict["scores"] == [0.9]
        assert result_dict["total_candidates"] == 3
        assert result_dict["query_time_ms"] == 12.5
        assert result_dict["strategy_used"] == "bm25"
        assert result_dict["count"] == 1
        assert result_dict["max_score"] == 0.9
        assert result_dict["min_score"] == 0.9
        assert result_dict["metadata"] == {"query_tokens": ["test"]}


class TestRetrievalConfig:
    """Test cases for RetrievalConfig."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = RetrievalConfig()

        assert config.storage_type == RetrievalStorageType.MEMORY
        assert config.storage_config == {}
        assert config.batch_size == 1000
        assert config.cache_size == 10000
        assert config.cache_ttl_seconds == 3600
        assert config.auto_rebuild_threshold == 100
        assert config.rebuild_on_startup is True
        assert config.min_score_threshold == 0.0
        assert config.max_results == 100

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = RetrievalConfig(
            storage_type=RetrievalStorageType.DUCKDB,
            storage_config={"path": "/tmp/retrieval.db"},
            batch_size=500,
            cache_size=5000,
            cache_ttl_seconds=1800,
            auto_rebuild_threshold=50,
            rebuild_on_startup=False,
            min_score_threshold=0.1,
            max_results=50,
        )

        assert config.storage_type == RetrievalStorageType.DUCKDB
        assert config.storage_config == {"path": "/tmp/retrieval.db"}
        assert config.batch_size == 500
        assert config.cache_size == 5000
        assert config.cache_ttl_seconds == 1800
        assert config.auto_rebuild_threshold == 50
        assert config.rebuild_on_startup is False
        assert config.min_score_threshold == 0.1
        assert config.max_results == 50

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = RetrievalConfig(
            storage_type=RetrievalStorageType.REDIS, storage_config={"host": "localhost", "port": 6379}, batch_size=2000
        )

        config_dict = config.to_dict()

        expected = {
            "storage_type": "redis",
            "storage_config": {"host": "localhost", "port": 6379},
            "batch_size": 2000,
            "cache_size": 10000,
            "cache_ttl_seconds": 3600,
            "auto_rebuild_threshold": 100,
            "rebuild_on_startup": True,
            "min_score_threshold": 0.0,
            "max_results": 100,
        }

        assert config_dict == expected


class TestIndexStats:
    """Test cases for IndexStats."""

    def test_stats_creation_with_defaults(self):
        """Test creating stats with default values."""
        stats = IndexStats()

        assert stats.total_episodes == 0
        assert stats.total_documents == 0
        assert stats.index_size_mb == 0.0
        assert stats.last_updated is None
        assert stats.build_time_ms == 0.0
        assert stats.provider_stats == {}

    def test_stats_creation_with_custom_values(self):
        """Test creating stats with custom values."""
        last_updated = datetime.now()

        stats = IndexStats(
            total_episodes=100,
            total_documents=120,
            index_size_mb=25.5,
            last_updated=last_updated,
            build_time_ms=1500.0,
            provider_stats={"tokenizer": "nltk", "stemmer": "porter"},
        )

        assert stats.total_episodes == 100
        assert stats.total_documents == 120
        assert stats.index_size_mb == 25.5
        assert stats.last_updated == last_updated
        assert stats.build_time_ms == 1500.0
        assert stats.provider_stats == {"tokenizer": "nltk", "stemmer": "porter"}

    def test_stats_to_dict_with_datetime(self):
        """Test converting stats to dictionary with datetime."""
        last_updated = datetime(2024, 1, 15, 10, 30, 0)

        stats = IndexStats(
            total_episodes=50,
            total_documents=60,
            index_size_mb=12.3,
            last_updated=last_updated,
            build_time_ms=800.0,
            provider_stats={"method": "bm25"},
        )

        stats_dict = stats.to_dict()

        expected = {
            "total_episodes": 50,
            "total_documents": 60,
            "index_size_mb": 12.3,
            "last_updated": "2024-01-15T10:30:00",
            "build_time_ms": 800.0,
            "provider_stats": {"method": "bm25"},
        }

        assert stats_dict == expected

    def test_stats_to_dict_with_none_datetime(self):
        """Test converting stats to dictionary with None datetime."""
        stats = IndexStats(total_episodes=10, last_updated=None)

        stats_dict = stats.to_dict()

        assert stats_dict["last_updated"] is None
        assert stats_dict["total_episodes"] == 10
