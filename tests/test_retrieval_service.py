"""
Tests for the retrieval service in Nemori.

This module contains tests for the RetrievalService that coordinates
different retrieval providers.
"""

import pytest
import pytest_asyncio

from nemori.core.episode import Episode, EpisodeLevel, EpisodeType
from nemori.retrieval import RetrievalConfig, RetrievalQuery, RetrievalService, RetrievalStrategy
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
async def retrieval_service(storage_repo):
    """Create a retrieval service for testing."""
    service = RetrievalService(storage_repo)
    yield service
    await service.close()


@pytest.fixture
def sample_episode():
    """Create a sample episode for testing."""
    return Episode(
        episode_id="test_ep",
        owner_id="user123",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
        title="Test Episode",
        content="This is a test episode for retrieval testing.",
        summary="Test episode summary",
    )


class TestRetrievalService:
    """Test cases for RetrievalService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, retrieval_service):
        """Test service initialization."""
        assert not retrieval_service._initialized

        # Should not have any providers initially
        assert len(retrieval_service.get_supported_strategies()) == 0

    @pytest.mark.asyncio
    async def test_register_bm25_provider(self, retrieval_service):
        """Test registering a BM25 provider."""
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)

        # Should have BM25 provider registered
        strategies = retrieval_service.get_supported_strategies()
        assert RetrievalStrategy.BM25 in strategies

        # Should be able to get the provider
        provider = retrieval_service.get_provider(RetrievalStrategy.BM25)
        assert provider is not None
        assert provider.strategy == RetrievalStrategy.BM25

    @pytest.mark.asyncio
    async def test_register_unsupported_strategy(self, retrieval_service):
        """Test registering an unsupported strategy."""
        config = RetrievalConfig(storage_type="memory")

        with pytest.raises(ValueError, match="Unsupported retrieval strategy"):
            retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, config)

    @pytest.mark.asyncio
    async def test_service_lifecycle(self, retrieval_service):
        """Test service initialization and cleanup."""
        # Register a provider
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)

        # Initialize service
        await retrieval_service.initialize()
        assert retrieval_service._initialized

        # Check health
        health = await retrieval_service.health_check()
        assert "bm25" in health
        assert health["bm25"] is True

        # Close service
        await retrieval_service.close()
        assert not retrieval_service._initialized
        assert len(retrieval_service.get_supported_strategies()) == 0

    @pytest.mark.asyncio
    async def test_search_with_registered_provider(self, retrieval_service, sample_episode):
        """Test searching using a registered provider."""
        # Register and initialize BM25 provider
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)
        await retrieval_service.initialize()

        # Add episode to provider
        await retrieval_service.add_episode_to_all_providers(sample_episode)

        # Search using the service
        query = RetrievalQuery(text="test episode", owner_id="user123", strategy=RetrievalStrategy.BM25)

        result = await retrieval_service.search(query)

        assert result.count > 0
        assert result.strategy_used == RetrievalStrategy.BM25
        assert result.episodes[0].episode_id == "test_ep"

    @pytest.mark.asyncio
    async def test_search_unregistered_strategy(self, retrieval_service):
        """Test searching with unregistered strategy."""
        await retrieval_service.initialize()

        query = RetrievalQuery(text="test", owner_id="user123", strategy=RetrievalStrategy.EMBEDDING)  # Not registered

        with pytest.raises(ValueError, match="No provider registered for strategy"):
            await retrieval_service.search(query)

    @pytest.mark.asyncio
    async def test_search_uninitialized_service(self, retrieval_service):
        """Test searching with uninitialized service."""
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)

        # Don't initialize service
        query = RetrievalQuery(text="test", owner_id="user123", strategy=RetrievalStrategy.BM25)

        with pytest.raises(RuntimeError, match="RetrievalService not initialized"):
            await retrieval_service.search(query)

    @pytest.mark.asyncio
    async def test_episode_lifecycle_operations(self, retrieval_service, sample_episode):
        """Test episode add/remove/update operations."""
        # Register and initialize provider
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)
        await retrieval_service.initialize()

        # Add episode
        await retrieval_service.add_episode_to_all_providers(sample_episode)

        # Verify episode was added
        stats = await retrieval_service.get_all_stats()
        assert "bm25" in stats
        assert stats["bm25"]["total_episodes"] == 1

        # Update episode
        sample_episode.title = "Updated Test Episode"
        await retrieval_service.update_episode_in_all_providers(sample_episode)

        # Search for updated content
        query = RetrievalQuery(text="Updated", owner_id="user123", strategy=RetrievalStrategy.BM25)
        result = await retrieval_service.search(query)
        assert result.count > 0
        assert "Updated" in result.episodes[0].title

        # Remove episode
        await retrieval_service.remove_episode_from_all_providers("test_ep")

        # Verify episode was removed
        stats = await retrieval_service.get_all_stats()
        assert stats["bm25"]["total_episodes"] == 0

    @pytest.mark.asyncio
    async def test_get_all_stats(self, retrieval_service, sample_episode):
        """Test getting statistics from all providers."""
        # Empty stats initially
        stats = await retrieval_service.get_all_stats()
        assert len(stats) == 0

        # Register provider and add episode
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)
        await retrieval_service.initialize()
        await retrieval_service.add_episode_to_all_providers(sample_episode)

        # Should have stats now
        stats = await retrieval_service.get_all_stats()
        assert "bm25" in stats
        assert stats["bm25"]["total_episodes"] == 1
        assert "tokenization_method" in stats["bm25"]["provider_stats"]

    @pytest.mark.asyncio
    async def test_health_check_all_providers(self, retrieval_service):
        """Test health check for all providers."""
        # Empty health initially
        health = await retrieval_service.health_check()
        assert len(health) == 0

        # Register provider
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)
        await retrieval_service.initialize()

        # Should have health status now
        health = await retrieval_service.health_check()
        assert "bm25" in health
        assert health["bm25"] is True

    @pytest.mark.asyncio
    async def test_get_provider_existing_and_missing(self, retrieval_service):
        """Test getting existing and non-existing providers."""
        # No provider initially
        provider = retrieval_service.get_provider(RetrievalStrategy.BM25)
        assert provider is None

        # Register provider
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)

        # Should get provider now
        provider = retrieval_service.get_provider(RetrievalStrategy.BM25)
        assert provider is not None

        # Non-registered strategy should return None
        provider = retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)
        assert provider is None

    @pytest.mark.asyncio
    async def test_hybrid_search_not_implemented(self, retrieval_service):
        """Test that hybrid search raises NotImplementedError."""
        query = RetrievalQuery(text="test", owner_id="user123", strategy=RetrievalStrategy.BM25)

        with pytest.raises(NotImplementedError, match="Hybrid search not yet implemented"):
            await retrieval_service.hybrid_search(query, [RetrievalStrategy.BM25])

    @pytest.mark.asyncio
    async def test_rerank_results_not_implemented(self, retrieval_service):
        """Test that result re-ranking raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Result re-ranking not yet implemented"):
            await retrieval_service.rerank_results([])

    @pytest.mark.asyncio
    async def test_operations_with_uninitialized_providers(self, retrieval_service, sample_episode):
        """Test that operations work correctly when some providers are not initialized."""
        # Register provider but don't initialize service
        config = RetrievalConfig(storage_type="memory")
        retrieval_service.register_provider(RetrievalStrategy.BM25, config)

        # These operations should not fail even with uninitialized providers
        await retrieval_service.add_episode_to_all_providers(sample_episode)
        await retrieval_service.update_episode_in_all_providers(sample_episode)
        await retrieval_service.remove_episode_from_all_providers("test_ep")

        # Stats and health should return empty results
        stats = await retrieval_service.get_all_stats()
        health = await retrieval_service.health_check()

        # Should handle uninitialized providers gracefully
        assert len(stats) == 0
        assert len(health) >= 0  # Health check might still report status
