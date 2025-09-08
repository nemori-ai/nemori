"""
Main retrieval service for Nemori episodic memory.

This module provides the high-level retrieval service that coordinates
different retrieval providers and can combine results from multiple strategies.
"""

from typing import Any

from ..storage.repository import EpisodicMemoryRepository
from .providers import BM25RetrievalProvider, RetrievalProvider
from .retrieval_types import RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy
from ..core.episode import Episode
from ..storage.repository import EpisodicMemoryRepository
from ..storage.storage_types import EpisodeQuery
from .providers import BM25RetrievalProvider, RetrievalProvider, EmbeddingRetrievalProvider

class RetrievalService:
    """
    Main retrieval service that coordinates different retrieval providers.

    This service can:
    1. Route queries to appropriate providers based on strategy
    2. Manage multiple providers simultaneously
    3. Combine results from multiple strategies (future: hybrid search)
    4. Handle provider lifecycle (initialization, cleanup)
    """

    def __init__(self, storage_repo: EpisodicMemoryRepository):
        """
        Initialize retrieval service.

        Args:
            storage_repo: Main storage repository for episodes
        """
        self.storage_repo = storage_repo
        self.providers: dict[RetrievalStrategy, RetrievalProvider] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the retrieval service."""
        if self._initialized:
            return

        # Initialize all registered providers
        for provider in self.providers.values():
            await provider.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close the service and all providers."""
        for provider in self.providers.values():
            await provider.close()

        self.providers.clear()
        self._initialized = False

    def register_provider(self, strategy: RetrievalStrategy, config: RetrievalConfig) -> None:
        """
        Register a retrieval provider for a specific strategy.

        Args:
            strategy: The retrieval strategy
            config: Configuration for the provider
        """
        if strategy == RetrievalStrategy.BM25:
            provider = BM25RetrievalProvider(config, self.storage_repo)
        elif strategy == RetrievalStrategy.EMBEDDING:
            provider = EmbeddingRetrievalProvider(config, self.storage_repo)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        self.providers[strategy] = provider

    def get_provider(self, strategy: RetrievalStrategy) -> RetrievalProvider | None:
        """Get a registered provider by strategy."""
        return self.providers.get(strategy)

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Search for episodes using the specified strategy.

        Args:
            query: The search query

        Returns:
            Search results from the appropriate provider

        Raises:
            ValueError: If strategy is not supported or not registered
            RuntimeError: If service is not initialized
        """
        if not self._initialized:
            raise RuntimeError("RetrievalService not initialized. Call initialize() first.")

        provider = self.providers.get(query.strategy)
        if not provider:
            raise ValueError(f"No provider registered for strategy: {query.strategy}")

        # Delegate to the appropriate provider
        return await provider.search(query)

    async def add_episode_to_all_providers(self, episode: Any) -> None:
        """
        Add an episode to all registered providers.

        This should be called when a new episode is created in storage.

        Args:
            episode: The episode to add to indices
        """
        for provider in self.providers.values():
            if await provider.is_initialized():
                await provider.add_episode(episode)

    async def remove_episode_from_all_providers(self, episode_id: str) -> None:
        """
        Remove an episode from all registered providers.

        This should be called when an episode is deleted from storage.

        Args:
            episode_id: ID of the episode to remove
        """
        for provider in self.providers.values():
            if await provider.is_initialized():
                await provider.remove_episode(episode_id)

    async def update_episode_in_all_providers(self, episode: Any) -> None:
        """
        Update an episode in all registered providers.

        This should be called when an episode is updated in storage.

        Args:
            episode: The updated episode
        """
        for provider in self.providers.values():
            if await provider.is_initialized():
                await provider.update_episode(episode)

    async def get_all_stats(self) -> dict[str, Any]:
        """
        Get statistics from all registered providers.

        Returns:
            Dictionary mapping strategy names to their stats
        """
        stats = {}
        for strategy, provider in self.providers.items():
            if await provider.is_initialized():
                provider_stats = await provider.get_stats()
                stats[strategy.value] = provider_stats.to_dict()

        return stats

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all registered providers.

        Returns:
            Dictionary mapping strategy names to health status
        """
        health = {}
        for strategy, provider in self.providers.items():
            health[strategy.value] = await provider.health_check()

        return health

    def get_supported_strategies(self) -> list[RetrievalStrategy]:
        """Get list of currently supported/registered strategies."""
        return list(self.providers.keys())

    # Future methods for hybrid search:

    async def hybrid_search(
        self,
        query: RetrievalQuery,
        strategies: list[RetrievalStrategy],
        weights: dict[RetrievalStrategy, float] | None = None,
    ) -> RetrievalResult:
        """
        Perform hybrid search using multiple strategies (future implementation).

        Args:
            query: The search query
            strategies: List of strategies to use
            weights: Optional weights for combining results

        Returns:
            Combined search results
        """
        # TODO: Implement hybrid search that:
        # 1. Runs query against multiple providers
        # 2. Combines and re-ranks results
        # 3. Returns unified results
        raise NotImplementedError("Hybrid search not yet implemented")

    async def rerank_results(
        self, results: list[RetrievalResult], rerank_strategy: str = "reciprocal_rank_fusion"
    ) -> RetrievalResult:
        """
        Re-rank results from multiple providers (future implementation).

        Args:
            results: Results from different providers
            rerank_strategy: Strategy for combining results

        Returns:
            Re-ranked unified results
        """
        # TODO: Implement result fusion strategies like:
        # - Reciprocal Rank Fusion (RRF)
        # - Weighted score combination
        # - Learning-to-rank approaches
        raise NotImplementedError("Result re-ranking not yet implemented")
