"""
Abstract base class for retrieval providers in Nemori.

This module defines the common interface that all retrieval providers must implement,
ensuring consistent behavior across different retrieval strategies.
"""

from abc import ABC, abstractmethod

from ...core.episode import Episode
from ...storage.repository import EpisodicMemoryRepository
from ..retrieval_types import IndexStats, RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy


class RetrievalProvider(ABC):
    """
    Abstract base class for episode retrieval providers.

    Each provider implements a specific retrieval strategy (BM25, embedding, etc.)
    and manages its own indexing and storage optimized for that strategy.
    """

    def __init__(self, config: RetrievalConfig, storage_repo: EpisodicMemoryRepository):
        """
        Initialize the retrieval provider.

        Args:
            config: Configuration for this provider
            storage_repo: Main storage repository for fetching episodes
        """
        self.config = config
        self.storage_repo = storage_repo
        self._initialized = False

    @property
    @abstractmethod
    def strategy(self) -> RetrievalStrategy:
        """Return the retrieval strategy this provider implements."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider and build initial index.

        This method should:
        1. Set up any provider-specific storage/connections
        2. Build the initial index from existing episodes
        3. Mark the provider as ready for queries
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the provider and cleanup resources.

        This method should cleanup any connections, close files, etc.
        """
        pass

    @abstractmethod
    async def add_episode(self, episode: Episode) -> None:
        """
        Add a new episode to the index.

        Args:
            episode: The episode to add to the index
        """
        pass

    @abstractmethod
    async def add_episodes_batch(self, episodes: list[Episode]) -> None:
        """
        Add multiple episodes to the index in batch.

        Args:
            episodes: List of episodes to add
        """
        pass

    @abstractmethod
    async def remove_episode(self, episode_id: str) -> bool:
        """
        Remove an episode from the index.

        Args:
            episode_id: ID of the episode to remove

        Returns:
            True if episode was found and removed, False otherwise
        """
        pass

    @abstractmethod
    async def update_episode(self, episode: Episode) -> bool:
        """
        Update an existing episode in the index.

        Args:
            episode: Updated episode data

        Returns:
            True if episode was found and updated, False otherwise
        """
        pass

    @abstractmethod
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Search for relevant episodes based on the query.

        Args:
            query: The search query

        Returns:
            Search results with episodes and relevance scores
        """
        pass

    @abstractmethod
    async def rebuild_index(self) -> None:
        """
        Rebuild the entire index from storage.

        This should fetch all episodes from storage and rebuild the index.
        Useful for maintenance or after bulk updates.
        """
        pass

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """
        Get statistics about the current index.

        Returns:
            Statistics about index size, performance, etc.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and ready to serve requests.

        Returns:
            True if healthy, False if there are issues
        """
        pass

    async def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        return self._initialized

    async def _fetch_all_episodes(self, owner_id: str | None = None) -> list[Episode]:
        """
        Helper method to fetch all episodes from storage.

        Args:
            owner_id: Optional filter by owner

        Returns:
            List of all episodes
        """
        if owner_id:
            result = await self.storage_repo.get_episodes_by_owner(owner_id)
            return result.episodes
        else:
            # Fetch all episodes - we might need to implement this method in storage
            # For now, we'll need to work with owner-specific queries
            raise NotImplementedError("Fetching all episodes across all owners not yet supported")

    def _validate_query(self, query: RetrievalQuery) -> None:
        """
        Validate that the query is appropriate for this provider.

        Args:
            query: The query to validate

        Raises:
            ValueError: If query is invalid for this provider
        """
        if query.strategy != self.strategy:
            raise ValueError(f"Query strategy {query.strategy} does not match provider strategy {self.strategy}")

        if not query.text.strip():
            raise ValueError("Query text cannot be empty")

        if not query.owner_id:
            raise ValueError("Owner ID is required")

        if query.limit <= 0:
            raise ValueError("Limit must be positive")
