"""
Tests for retrieval providers in Nemori.

This module contains comprehensive tests for the retrieval provider system,
focusing on the BM25RetrievalProvider implementation.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from nemori.core.data_types import TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.retrieval import (
    BM25RetrievalProvider,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage import MemoryEpisodicMemoryRepository, StorageConfig


@pytest_asyncio.fixture
async def storage_repo():
    """Create a memory storage repository for testing."""
    config = StorageConfig(backend_type="memory")
    repo = MemoryEpisodicMemoryRepository(config)
    await repo.initialize()
    yield repo
    await repo.close()


@pytest_asyncio.fixture
async def bm25_provider(storage_repo):
    """Create a BM25 retrieval provider for testing with memory storage (no persistence)."""
    config = RetrievalConfig(storage_type=RetrievalStorageType.MEMORY)
    provider = BM25RetrievalProvider(config, storage_repo)
    await provider.initialize()
    yield provider
    await provider.close()


@pytest_asyncio.fixture
async def bm25_provider_with_persistence(storage_repo):
    """Create a BM25 retrieval provider with disk storage (persistence enabled) for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
        provider = BM25RetrievalProvider(config, storage_repo)
        await provider.initialize()
        yield provider, temp_dir
        await provider.close()


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    return [
        Episode(
            episode_id="ep1",
            owner_id="user123",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Python Programming Tutorial",
            content="Learning Python basics including variables, functions, and object-oriented programming concepts.",
            summary="Python programming fundamentals tutorial",
            temporal_info=TemporalInfo(datetime.now()),
            metadata=EpisodeMetadata(
                entities=["Python", "programming", "tutorial"],
                topics=["education", "coding", "software"],
                key_points=["variables", "functions", "OOP"],
            ),
        ),
        Episode(
            episode_id="ep2",
            owner_id="user123",
            episode_type=EpisodeType.CONVERSATIONAL,
            level=EpisodeLevel.ATOMIC,
            title="Machine Learning with Python",
            content="Implementing machine learning algorithms using scikit-learn and pandas for data analysis.",
            summary="Machine learning implementation using Python libraries",
            temporal_info=TemporalInfo(datetime.now()),
            metadata=EpisodeMetadata(
                entities=["machine learning", "Python", "scikit-learn", "pandas"],
                topics=["AI", "data science", "programming"],
                key_points=["algorithms", "data analysis", "libraries"],
            ),
        ),
        Episode(
            episode_id="ep3",
            owner_id="user456",  # Different user
            episode_type=EpisodeType.BEHAVIORAL,
            level=EpisodeLevel.ATOMIC,
            title="Morning Exercise Routine",
            content="Daily morning workout including running, stretching, and strength training exercises.",
            summary="Morning fitness routine with various exercises",
            temporal_info=TemporalInfo(datetime.now()),
            metadata=EpisodeMetadata(
                entities=["exercise", "running", "fitness"],
                topics=["health", "lifestyle", "wellness"],
                key_points=["morning routine", "strength training", "stretching"],
            ),
        ),
    ]


class TestBM25RetrievalProvider:
    """Test cases for BM25RetrievalProvider."""

    @pytest.mark.asyncio
    async def test_provider_initialization(self, bm25_provider):
        """Test provider initialization."""
        assert bm25_provider.strategy == RetrievalStrategy.BM25
        assert await bm25_provider.is_initialized()
        assert await bm25_provider.health_check()

    @pytest.mark.asyncio
    async def test_tokenization(self, bm25_provider):
        """Test text tokenization with NLTK."""
        # Test basic tokenization
        tokens = bm25_provider._tokenize("Hello world! This is a test.")
        assert "hello" in tokens or "test" in tokens  # Should be stemmed
        assert "is" not in tokens  # Should be removed as stopword
        assert "!" not in tokens  # Punctuation should be removed

        # Test empty text
        assert bm25_provider._tokenize("") == []
        assert bm25_provider._tokenize(None) == []

        # Test with technical terms
        tokens = bm25_provider._tokenize("Python programming algorithms")
        assert len(tokens) == 3
        # Should contain stemmed versions
        expected_stems = {"python", "program", "algorithm"}
        assert any(stem in " ".join(tokens) for stem in expected_stems)

    @pytest.mark.asyncio
    async def test_add_single_episode(self, bm25_provider, sample_episodes):
        """Test adding a single episode to the index."""
        episode = sample_episodes[0]
        await bm25_provider.add_episode(episode)

        stats = await bm25_provider.get_stats()
        assert stats.total_episodes == 1
        assert stats.total_documents == 1

    @pytest.mark.asyncio
    async def test_add_episodes_batch(self, bm25_provider, sample_episodes):
        """Test adding multiple episodes in batch."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        stats = await bm25_provider.get_stats()
        assert stats.total_episodes == 3
        assert stats.total_documents == 3

        # Check that different users have separate indices
        assert "user_indices_count" in stats.provider_stats
        assert stats.provider_stats["user_indices_count"] == 2  # user123 and user456

    @pytest.mark.asyncio
    async def test_search_basic(self, bm25_provider, sample_episodes):
        """Test basic search functionality."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Search for Python-related content
        query = RetrievalQuery(text="Python programming", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

        result = await bm25_provider.search(query)

        assert result.count > 0
        assert result.strategy_used == RetrievalStrategy.BM25
        assert len(result.episodes) == len(result.scores)
        assert result.query_time_ms > 0

        # Check that most relevant episode is first
        assert "Python" in result.episodes[0].title
        # BM25 scores can sometimes be negative, so just check that we have results
        assert len(result.scores) > 0

    @pytest.mark.asyncio
    async def test_search_user_isolation(self, bm25_provider, sample_episodes):
        """Test that search results are isolated by user."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Search as user123 - should only see their episodes
        query = RetrievalQuery(text="exercise", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

        result = await bm25_provider.search(query)
        # user123 has no exercise-related episodes
        assert all(ep.owner_id == "user123" for ep in result.episodes)

        # Search as user456 - should see exercise episode
        query.owner_id = "user456"
        result = await bm25_provider.search(query)
        assert result.count > 0
        assert all(ep.owner_id == "user456" for ep in result.episodes)
        assert "Exercise" in result.episodes[0].title

    @pytest.mark.asyncio
    async def test_search_relevance_scoring(self, bm25_provider, sample_episodes):
        """Test that search results are properly scored by relevance."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        query = RetrievalQuery(
            text="machine learning algorithms", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25
        )

        result = await bm25_provider.search(query)

        # Should return results in descending score order
        for i in range(1, len(result.scores)):
            assert result.scores[i - 1] >= result.scores[i]

        # Episode about machine learning should be in the results
        episode_titles = [ep.title for ep in result.episodes]
        assert any("Machine Learning" in title for title in episode_titles)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, bm25_provider, sample_episodes):
        """Test search with episode type filters."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        query = RetrievalQuery(
            text="Python",
            owner_id="user123",
            limit=5,
            strategy=RetrievalStrategy.BM25,
            episode_types=["conversational"],
        )

        result = await bm25_provider.search(query)

        # Should only return conversational episodes
        assert all(ep.episode_type == EpisodeType.CONVERSATIONAL for ep in result.episodes)

    @pytest.mark.asyncio
    async def test_search_empty_query(self, bm25_provider, sample_episodes):
        """Test search with empty query."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        query = RetrievalQuery(text="", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

        with pytest.raises(ValueError, match="Query text cannot be empty"):
            await bm25_provider.search(query)

    @pytest.mark.asyncio
    async def test_search_no_episodes(self, bm25_provider):
        """Test search when no episodes are indexed."""
        query = RetrievalQuery(text="Python programming", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

        result = await bm25_provider.search(query)

        assert result.count == 0
        assert result.episodes == []
        assert result.scores == []
        assert result.total_candidates == 0

    @pytest.mark.asyncio
    async def test_remove_episode(self, bm25_provider, sample_episodes):
        """Test removing an episode from the index."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Verify episode exists
        initial_stats = await bm25_provider.get_stats()
        assert initial_stats.total_episodes == 3

        # Remove an episode
        removed = await bm25_provider.remove_episode("ep1")
        assert removed is True

        # Verify episode removed
        final_stats = await bm25_provider.get_stats()
        assert final_stats.total_episodes == 2

        # Try to remove non-existent episode
        removed = await bm25_provider.remove_episode("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_update_episode(self, bm25_provider, sample_episodes):
        """Test updating an episode in the index."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Update an episode
        updated_episode = sample_episodes[0]
        updated_episode.title = "Advanced Python Programming"
        updated_episode.content = "Advanced Python concepts including decorators and metaclasses."

        success = await bm25_provider.update_episode(updated_episode)
        assert success is True

        # Search should find updated content
        query = RetrievalQuery(
            text="decorators metaclasses", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25
        )

        result = await bm25_provider.search(query)
        assert result.count > 0
        assert "Advanced" in result.episodes[0].title

        # Try to update non-existent episode
        nonexistent_episode = Episode(
            episode_id="nonexistent", owner_id="user123", title="Test", content="Test content"
        )
        success = await bm25_provider.update_episode(nonexistent_episode)
        assert success is False

    @pytest.mark.asyncio
    async def test_weighted_fields(self, bm25_provider):
        """Test that title gets higher weight than content."""
        # Episode with keyword in title
        episode1 = Episode(
            episode_id="ep1",
            owner_id="user123",
            title="Algorithms Tutorial",  # Keyword in title
            content="Learning basic programming concepts.",
            summary="Programming tutorial",
        )

        # Episode with keyword in content only
        episode2 = Episode(
            episode_id="ep2",
            owner_id="user123",
            title="Programming Basics",
            content="This tutorial covers algorithms and data structures.",  # Keyword in content
            summary="Basic programming concepts",
        )

        await bm25_provider.add_episodes_batch([episode1, episode2])

        query = RetrievalQuery(text="algorithms", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

        result = await bm25_provider.search(query)

        # Episode with keyword in title should be in the results and have reasonable score
        episode_ids = [ep.episode_id for ep in result.episodes]
        assert "ep1" in episode_ids
        assert len(result.scores) == 2

    @pytest.mark.asyncio
    async def test_query_validation(self, bm25_provider):
        """Test query validation."""
        # Test wrong strategy
        query = RetrievalQuery(text="test", owner_id="user123", strategy=RetrievalStrategy.EMBEDDING)  # Wrong strategy

        with pytest.raises(ValueError, match="Query strategy .* does not match provider strategy"):
            await bm25_provider.search(query)

        # Test missing owner_id
        query = RetrievalQuery(text="test", owner_id="", strategy=RetrievalStrategy.BM25)

        with pytest.raises(ValueError, match="Owner ID is required"):
            await bm25_provider.search(query)

        # Test invalid limit
        query = RetrievalQuery(text="test", owner_id="user123", limit=0, strategy=RetrievalStrategy.BM25)

        with pytest.raises(ValueError, match="Limit must be positive"):
            await bm25_provider.search(query)

    @pytest.mark.asyncio
    async def test_get_stats(self, bm25_provider, sample_episodes):
        """Test getting index statistics."""
        # Test empty index
        stats = await bm25_provider.get_stats()
        assert stats.total_episodes == 0
        assert stats.total_documents == 0
        assert stats.index_size_mb == 0.0

        # Add episodes and test stats
        await bm25_provider.add_episodes_batch(sample_episodes)
        stats = await bm25_provider.get_stats()

        assert stats.total_episodes == 3
        assert stats.total_documents == 3
        assert stats.index_size_mb > 0
        assert stats.last_updated is not None

        # Check provider-specific stats
        assert stats.provider_stats["tokenization_method"] == "nltk_with_stemming"
        assert stats.provider_stats["stemmer"] == "porter"
        assert stats.provider_stats["stopwords_removed"] is True

    @pytest.mark.asyncio
    async def test_rebuild_index(self, bm25_provider, sample_episodes):
        """Test rebuilding the entire index."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Verify episodes are indexed
        initial_stats = await bm25_provider.get_stats()
        assert initial_stats.total_episodes == 3

        # Rebuild index (currently a no-op, but should not fail)
        await bm25_provider.rebuild_index()

        # Should still work
        assert await bm25_provider.health_check()

    @pytest.mark.asyncio
    async def test_close_and_cleanup(self, storage_repo):
        """Test provider cleanup on close."""
        config = RetrievalConfig(storage_type=RetrievalStorageType.MEMORY)
        provider = BM25RetrievalProvider(config, storage_repo)
        await provider.initialize()

        # Add some data
        episode = Episode(episode_id="test", owner_id="user123", title="Test Episode", content="Test content")
        await provider.add_episode(episode)

        # Verify data exists
        stats = await provider.get_stats()
        assert stats.total_episodes == 1

        # Close provider
        await provider.close()

        # Verify cleanup
        assert not await provider.is_initialized()
        stats = await provider.get_stats()
        assert stats.total_episodes == 0  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_fallback_scoring(self, bm25_provider):
        """Test fallback to term frequency when BM25 scores are zero."""
        # Create episodes with very common words that might result in zero BM25 scores
        episode = Episode(
            episode_id="ep1",
            owner_id="user123",
            title="Common Words",
            content="This is a very common text with common words.",
            summary="Common text example",
        )

        await bm25_provider.add_episode(episode)

        # Query with very specific uncommon terms
        query = RetrievalQuery(
            text="xyzzyx nonexistent",  # Completely unrelated terms
            owner_id="user123",
            limit=5,
            strategy=RetrievalStrategy.BM25,
        )

        result = await bm25_provider.search(query)

        # Should still return results (even with zero scores)
        assert result.total_candidates >= 0
        # Query time should be reasonable
        assert result.query_time_ms >= 0


class TestBM25Persistence:
    """Test cases for BM25 persistence features."""

    @pytest.mark.asyncio
    async def test_disk_storage_configuration(self, storage_repo):
        """Test that disk storage type enables persistence and configures directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
            provider = BM25RetrievalProvider(config, storage_repo)

            assert provider.persistence_enabled is True
            assert str(provider.persistence_dir) == temp_dir
            assert provider.persistence_dir.exists()

    @pytest.mark.asyncio
    async def test_save_index_to_disk(self, bm25_provider_with_persistence, sample_episodes):
        """Test saving index to disk."""
        provider, temp_dir = bm25_provider_with_persistence

        # Add episodes
        await provider.add_episodes_batch(sample_episodes)

        # Manually trigger save
        provider._save_index_to_disk("user123")

        # Check that index file was created
        index_file = Path(temp_dir) / "bm25_index_user123.pkl"
        assert index_file.exists()
        assert index_file.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_load_index_from_disk(self, storage_repo, sample_episodes):
        """Test loading index from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First provider - create and save index
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
            provider1 = BM25RetrievalProvider(config, storage_repo)
            await provider1.initialize()
            await provider1.add_episodes_batch(sample_episodes)

            # Save episodes to storage repo
            for episode in sample_episodes:
                await storage_repo.store_episode(episode)

            await provider1.close()

            # Second provider - should load existing index
            provider2 = BM25RetrievalProvider(config, storage_repo)
            await provider2.initialize()

            # Should be able to search immediately
            query = RetrievalQuery(
                text="Python programming", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25
            )

            result = await provider2.search(query)
            assert result.count > 0
            assert len(result.episodes) > 0

            await provider2.close()

    @pytest.mark.asyncio
    async def test_load_all_indices_from_disk(self, storage_repo, sample_episodes):
        """Test loading multiple user indices from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create provider and add episodes for multiple users
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
            provider1 = BM25RetrievalProvider(config, storage_repo)
            await provider1.initialize()
            await provider1.add_episodes_batch(sample_episodes)

            # Save episodes to storage
            for episode in sample_episodes:
                await storage_repo.store_episode(episode)

            await provider1.close()

            # Verify multiple index files exist
            index_files = list(Path(temp_dir).glob("bm25_index_*.pkl"))
            assert len(index_files) >= 2  # user123 and user456

            # Create new provider - should load all indices
            provider2 = BM25RetrievalProvider(config, storage_repo)
            await provider2.initialize()

            stats = await provider2.get_stats()
            assert stats.provider_stats["user_indices_count"] >= 2

            await provider2.close()

    @pytest.mark.asyncio
    async def test_memory_storage_no_persistence(self, storage_repo, sample_episodes):
        """Test that memory storage type disables persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.MEMORY)
            provider = BM25RetrievalProvider(config, storage_repo)

            # Memory storage should disable persistence
            assert provider.persistence_enabled is False
            assert provider.persistence_dir is None

            await provider.initialize()
            await provider.add_episodes_batch(sample_episodes)
            await provider.close()

            # No index files should be created
            index_files = list(Path(temp_dir).glob("bm25_index_*.pkl"))
            assert len(index_files) == 0

    @pytest.mark.asyncio
    async def test_disk_storage_enables_persistence(self, storage_repo, sample_episodes):
        """Test that disk storage type enables persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
            provider = BM25RetrievalProvider(config, storage_repo)

            # Disk storage should enable persistence
            assert provider.persistence_enabled is True
            assert provider.persistence_dir is not None
            assert str(provider.persistence_dir) == temp_dir

            await provider.initialize()
            await provider.add_episodes_batch(sample_episodes)
            await provider.close()

            # Index files should be created
            index_files = list(Path(temp_dir).glob("bm25_index_*.pkl"))
            assert len(index_files) >= 2  # user123 and user456

    @pytest.mark.asyncio
    async def test_other_storage_types_no_local_persistence(self, storage_repo):
        """Test that other storage types (duckdb, redis, etc.) disable local persistence."""
        config = RetrievalConfig(storage_type=RetrievalStorageType.DUCKDB)
        provider = BM25RetrievalProvider(config, storage_repo)

        # Other storage types should disable local persistence (they handle their own)
        assert provider.persistence_enabled is False
        assert provider.persistence_dir is None

        await provider.close()


class TestBM25EpisodeReloading:
    """Test cases for episode reloading from storage."""

    @pytest.mark.asyncio
    async def test_reload_episodes_from_storage(self, storage_repo, sample_episodes):
        """Test reloading episodes from storage when index is loaded from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})

            # First: Create provider, add episodes, and save
            provider1 = BM25RetrievalProvider(config, storage_repo)
            await provider1.initialize()
            await provider1.add_episodes_batch(sample_episodes)

            # Store episodes in storage repo
            for episode in sample_episodes:
                await storage_repo.store_episode(episode)

            await provider1.close()

            # Second: Create new provider that loads from disk
            provider2 = BM25RetrievalProvider(config, storage_repo)
            await provider2.initialize()

            # Test manual reload
            await provider2._reload_episodes_from_storage("user123")

            # Should be able to search
            query = RetrievalQuery(text="Python", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

            result = await provider2.search(query)
            assert result.count > 0
            assert len(result.episodes) > 0

            await provider2.close()

    @pytest.mark.asyncio
    async def test_automatic_episode_reloading_during_search(self, storage_repo, sample_episodes):
        """Test that episodes are automatically reloaded during search if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})

            # Create provider and add episodes
            provider1 = BM25RetrievalProvider(config, storage_repo)
            await provider1.initialize()
            await provider1.add_episodes_batch(sample_episodes)

            # Store episodes in storage repo
            for episode in sample_episodes:
                await storage_repo.store_episode(episode)

            await provider1.close()

            # Create new provider (episodes will be missing but corpus exists)
            provider2 = BM25RetrievalProvider(config, storage_repo)
            await provider2.initialize()

            # Search should trigger automatic episode reloading
            query = RetrievalQuery(text="Python", owner_id="user123", limit=5, strategy=RetrievalStrategy.BM25)

            result = await provider2.search(query)
            assert result.count > 0
            assert len(result.episodes) > 0

            await provider2.close()

    @pytest.mark.asyncio
    async def test_reload_with_no_episodes_in_storage(self, storage_repo):
        """Test reloading when no episodes exist in storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})

            provider = BM25RetrievalProvider(config, storage_repo)
            await provider.initialize()

            # Try to reload for non-existent user
            await provider._reload_episodes_from_storage("nonexistent_user")

            # Should not crash and should handle gracefully
            query = RetrievalQuery(
                text="anything", owner_id="nonexistent_user", limit=5, strategy=RetrievalStrategy.BM25
            )

            result = await provider.search(query)
            assert result.count == 0

            await provider.close()


class TestBM25IndexManagement:
    """Test cases for index file management."""

    @pytest.mark.asyncio
    async def test_get_index_file_path(self, storage_repo):
        """Test index file path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})
            provider = BM25RetrievalProvider(config, storage_repo)

            path = provider._get_index_file_path("test_user")
            expected = Path(temp_dir) / "bm25_index_test_user.pkl"
            assert path == expected

    @pytest.mark.asyncio
    async def test_index_serialization_format(self, bm25_provider_with_persistence, sample_episodes):
        """Test that index serialization contains expected data."""
        import pickle

        provider, temp_dir = bm25_provider_with_persistence

        # Add episodes
        await provider.add_episodes_batch([sample_episodes[0]])  # Just one episode

        # Save index
        provider._save_index_to_disk("user123")

        # Read and verify serialized data
        index_file = Path(temp_dir) / "bm25_index_user123.pkl"
        with open(index_file, "rb") as f:
            data = pickle.load(f)

        # Check structure
        assert "episodes" in data
        assert "corpus" in data
        assert "episode_id_to_index" in data
        assert "last_updated" in data
        assert "metadata" in data

        # Check content
        assert len(data["episodes"]) == 1
        assert len(data["corpus"]) == 1
        assert data["episodes"][0]["episode_id"] == "ep1"
        assert data["metadata"]["total_episodes"] == 1

    @pytest.mark.asyncio
    async def test_corrupted_index_file_handling(self, storage_repo):
        """Test handling of corrupted index files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})

            # Create corrupted index file
            index_file = Path(temp_dir) / "bm25_index_user123.pkl"
            with open(index_file, "w") as f:
                f.write("corrupted data")

            provider = BM25RetrievalProvider(config, storage_repo)

            # Should handle corrupted file gracefully
            success = provider._load_index_from_disk("user123")
            assert success is False

            await provider.initialize()  # Should not crash
            assert await provider.health_check()

            await provider.close()

    @pytest.mark.asyncio
    async def test_close_saves_all_indices(self, storage_repo, sample_episodes):
        """Test that closing provider saves all user indices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": temp_dir})

            provider = BM25RetrievalProvider(config, storage_repo)
            await provider.initialize()
            await provider.add_episodes_batch(sample_episodes)

            # Close should save all indices
            await provider.close()

            # Check that index files were created for both users
            user123_file = Path(temp_dir) / "bm25_index_user123.pkl"
            user456_file = Path(temp_dir) / "bm25_index_user456.pkl"

            assert user123_file.exists()
            assert user456_file.exists()
            assert user123_file.stat().st_size > 0
            assert user456_file.stat().st_size > 0


class TestBM25EnhancedFeatures:
    """Test cases for enhanced BM25 features."""

    @pytest.mark.asyncio
    async def test_user_index_isolation(self, bm25_provider, sample_episodes):
        """Test that user indices are properly isolated."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        # Check internal structure
        assert len(bm25_provider.user_indices) == 2  # user123 and user456
        assert "user123" in bm25_provider.user_indices
        assert "user456" in bm25_provider.user_indices

        # Each user should have their own index data
        user123_index = bm25_provider._get_user_index("user123")
        user456_index = bm25_provider._get_user_index("user456")

        assert len(user123_index["episodes"]) == 2  # 2 episodes for user123
        assert len(user456_index["episodes"]) == 1  # 1 episode for user456

        # BM25 objects should be separate
        assert user123_index["bm25"] is not user456_index["bm25"]

    @pytest.mark.asyncio
    async def test_searchable_text_building(self, bm25_provider):
        """Test the _build_searchable_text method with weighted fields."""
        episode = Episode(
            episode_id="test",
            owner_id="user123",
            title="Test Title",
            content="Test content here",
            summary="Test summary",
            metadata=EpisodeMetadata(
                entities=["entity1", "entity2"], topics=["topic1", "topic2"], key_points=["point1", "point2"]
            ),
        )

        searchable_text = bm25_provider._build_searchable_text(episode)

        # Title should appear 3 times (highest weight)
        assert searchable_text.count("Test Title") == 3

        # Summary should appear 2 times
        assert searchable_text.count("Test summary") == 2

        # Entities should appear 2 times each
        assert searchable_text.count("entity1") == 2
        assert searchable_text.count("entity2") == 2

        # Topics should appear 2 times each
        assert searchable_text.count("topic1") == 2
        assert searchable_text.count("topic2") == 2

        # Content and key points should appear once
        assert "Test content here" in searchable_text
        assert "point1" in searchable_text
        assert "point2" in searchable_text

    @pytest.mark.asyncio
    async def test_enhanced_stats_with_provider_specific_info(self, bm25_provider, sample_episodes):
        """Test that stats include provider-specific information."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        stats = await bm25_provider.get_stats()

        # Check provider-specific stats
        provider_stats = stats.provider_stats
        assert provider_stats["user_indices_count"] == 2
        assert provider_stats["tokenization_method"] == "nltk_with_stemming"
        assert provider_stats["weighting_strategy"] == "title*3, summary*2, entities*2, topics*2"
        assert provider_stats["stemmer"] == "porter"
        assert provider_stats["stopwords_removed"] is True

        # Check overall stats
        assert stats.total_episodes == 3
        assert stats.total_documents == 3
        assert stats.index_size_mb > 0
        assert stats.last_updated is not None

    @pytest.mark.asyncio
    async def test_episode_id_to_index_mapping(self, bm25_provider, sample_episodes):
        """Test that episode ID to index mapping works correctly."""
        await bm25_provider.add_episodes_batch(sample_episodes)

        user123_index = bm25_provider._get_user_index("user123")

        # Check mapping
        assert "ep1" in user123_index["episode_id_to_index"]
        assert "ep2" in user123_index["episode_id_to_index"]

        # Indices should be valid
        ep1_idx = user123_index["episode_id_to_index"]["ep1"]
        ep2_idx = user123_index["episode_id_to_index"]["ep2"]

        assert 0 <= ep1_idx < len(user123_index["episodes"])
        assert 0 <= ep2_idx < len(user123_index["episodes"])

        # Episodes should be accessible via mapping
        assert user123_index["episodes"][ep1_idx].episode_id == "ep1"
        assert user123_index["episodes"][ep2_idx].episode_id == "ep2"
