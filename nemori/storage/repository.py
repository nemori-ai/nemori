"""
Repository interfaces for storage layer in Nemori.

This module defines abstract repository interfaces for managing raw event data,
episodes, and their relationships in the episodic memory system.
"""

from abc import ABC, abstractmethod

from ..core.data_types import DataType, RawEventData
from ..core.episode import Episode
from .storage_types import (
    EpisodeQuery,
    EpisodeSearchResult,
    RawDataQuery,
    RawDataSearchResult,
    StorageConfig,
    StorageStats,
)


class StorageRepository(ABC):
    """
    Base repository interface for storage operations.

    Provides common storage functionality like configuration, statistics,
    and lifecycle management.
    """

    def __init__(self, config: StorageConfig):
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the storage backend is healthy."""
        pass

    @abstractmethod
    async def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        pass

    @abstractmethod
    async def backup(self, destination: str) -> bool:
        """Create a backup of the storage."""
        pass

    @abstractmethod
    async def restore(self, source: str) -> bool:
        """Restore storage from a backup."""
        pass


class RawDataRepository(StorageRepository):
    """
    Repository interface for raw event data storage.

    Handles storage and retrieval of original user data before processing
    into episodes. Maintains complete data integrity and supports efficient
    querying for episode generation.
    """

    @abstractmethod
    async def store_raw_data(self, data: RawEventData) -> str:
        """
        Store raw event data.

        Args:
            data: The raw event data to store

        Returns:
            The data_id of the stored data
        """
        pass

    @abstractmethod
    async def store_raw_data_batch(self, data_list: list[RawEventData]) -> list[str]:
        """
        Store multiple raw event data in batch.

        Args:
            data_list: List of raw event data to store

        Returns:
            List of data_ids of the stored data
        """
        pass

    @abstractmethod
    async def get_raw_data(self, data_id: str) -> RawEventData | None:
        """
        Retrieve raw event data by ID.

        Args:
            data_id: The unique identifier of the data

        Returns:
            The raw event data or None if not found
        """
        pass

    @abstractmethod
    async def get_raw_data_batch(self, data_ids: list[str]) -> list[RawEventData | None]:
        """
        Retrieve multiple raw event data by IDs.

        Args:
            data_ids: List of data identifiers

        Returns:
            List of raw event data (None for not found items)
        """
        pass

    @abstractmethod
    async def search_raw_data(self, query: RawDataQuery) -> RawDataSearchResult:
        """
        Search raw event data based on query parameters.

        Args:
            query: The search query parameters

        Returns:
            Search results containing matching raw data
        """
        pass

    @abstractmethod
    async def update_raw_data(self, data_id: str, data: RawEventData) -> bool:
        """
        Update existing raw event data.

        Args:
            data_id: The unique identifier of the data
            data: The updated raw event data

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def mark_as_processed(self, data_id: str, processing_version: str) -> bool:
        """
        Mark raw data as processed.

        Args:
            data_id: The unique identifier of the data
            processing_version: Version of the processing pipeline

        Returns:
            True if marking was successful, False otherwise
        """
        pass

    @abstractmethod
    async def mark_batch_as_processed(self, data_ids: list[str], processing_version: str) -> list[bool]:
        """
        Mark multiple raw data as processed.

        Args:
            data_ids: List of data identifiers
            processing_version: Version of the processing pipeline

        Returns:
            List of success status for each data_id
        """
        pass

    @abstractmethod
    async def delete_raw_data(self, data_id: str) -> bool:
        """
        Delete raw event data.

        Args:
            data_id: The unique identifier of the data

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_unprocessed_data(
        self, data_type: DataType | None = None, limit: int | None = None
    ) -> list[RawEventData]:
        """
        Get unprocessed raw event data.

        Args:
            data_type: Optional filter by data type
            limit: Optional limit on number of results

        Returns:
            List of unprocessed raw event data
        """
        pass


class EpisodicMemoryRepository(StorageRepository):
    """
    Repository interface for episodic memory storage.

    Handles storage and retrieval of processed episodes, supports various
    search methods including semantic search, and maintains relationships
    between episodes and their source data.
    """

    @abstractmethod
    async def store_episode(self, episode: Episode) -> str:
        """
        Store an episode.

        Args:
            episode: The episode to store

        Returns:
            The episode_id of the stored episode
        """
        pass

    @abstractmethod
    async def store_episode_batch(self, episodes: list[Episode]) -> list[str]:
        """
        Store multiple episodes in batch.

        Args:
            episodes: List of episodes to store

        Returns:
            List of episode_ids of the stored episodes
        """
        pass

    @abstractmethod
    async def get_episode(self, episode_id: str) -> Episode | None:
        """
        Retrieve an episode by ID.

        Args:
            episode_id: The unique identifier of the episode

        Returns:
            The episode or None if not found
        """
        pass

    @abstractmethod
    async def get_episode_batch(self, episode_ids: list[str]) -> list[Episode | None]:
        """
        Retrieve multiple episodes by IDs.

        Args:
            episode_ids: List of episode identifiers

        Returns:
            List of episodes (None for not found items)
        """
        pass

    @abstractmethod
    async def search_episodes(self, query: EpisodeQuery) -> EpisodeSearchResult:
        """
        Search episodes based on query parameters.

        Args:
            query: The search query parameters

        Returns:
            Search results containing matching episodes
        """
        pass

    @abstractmethod
    async def search_episodes_by_text(
        self, text: str, owner_id: str | None = None, limit: int | None = None
    ) -> EpisodeSearchResult:
        """
        Search episodes by text content.

        Args:
            text: Text to search for
            owner_id: Optional filter by owner
            limit: Optional limit on results

        Returns:
            Search results containing matching episodes
        """
        pass

    @abstractmethod
    async def search_episodes_by_keywords(
        self, keywords: list[str], owner_id: str | None = None, limit: int | None = None
    ) -> EpisodeSearchResult:
        """
        Search episodes by keywords.

        Args:
            keywords: Keywords to search for
            owner_id: Optional filter by owner
            limit: Optional limit on results

        Returns:
            Search results containing matching episodes
        """
        pass

    @abstractmethod
    async def search_episodes_by_embedding(
        self,
        embedding: list[float],
        owner_id: str | None = None,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> EpisodeSearchResult:
        """
        Search episodes by semantic similarity using embeddings.

        Args:
            embedding: Query embedding vector
            owner_id: Optional filter by owner
            limit: Optional limit on results
            threshold: Optional similarity threshold

        Returns:
            Search results with relevance scores
        """
        pass

    @abstractmethod
    async def get_episodes_by_owner(
        self, owner_id: str, limit: int | None = None, offset: int | None = None
    ) -> EpisodeSearchResult:
        """
        Get episodes for a specific owner.

        Args:
            owner_id: The owner identifier
            limit: Optional limit on results
            offset: Optional offset for pagination

        Returns:
            Episodes belonging to the owner
        """
        pass

    @abstractmethod
    async def get_recent_episodes(
        self, owner_id: str | None = None, hours: int = 24, limit: int | None = None
    ) -> EpisodeSearchResult:
        """
        Get recent episodes within specified time window.

        Args:
            owner_id: Optional filter by owner
            hours: Time window in hours
            limit: Optional limit on results

        Returns:
            Recent episodes
        """
        pass

    @abstractmethod
    async def update_episode(self, episode_id: str, episode: Episode) -> bool:
        """
        Update an existing episode.

        Args:
            episode_id: The unique identifier of the episode
            episode: The updated episode

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_episode_importance(self, episode_id: str, importance_score: float) -> bool:
        """
        Update episode importance score.

        Args:
            episode_id: The unique identifier of the episode
            importance_score: New importance score (0.0 to 1.0)

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def mark_episode_accessed(self, episode_id: str) -> bool:
        """
        Mark an episode as accessed (increment recall count).

        Args:
            episode_id: The unique identifier of the episode

        Returns:
            True if marking was successful, False otherwise
        """
        pass

    @abstractmethod
    async def link_episode_to_raw_data(self, episode_id: str, raw_data_ids: list[str]) -> bool:
        """
        Create association between episode and its source raw data.

        Args:
            episode_id: The episode identifier
            raw_data_ids: List of raw data identifiers that contributed to this episode

        Returns:
            True if linking was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_episodes_for_raw_data(self, raw_data_id: str) -> list[Episode]:
        """
        Get all episodes that were created from specific raw data.

        Args:
            raw_data_id: The raw data identifier

        Returns:
            List of episodes derived from the raw data
        """
        pass

    @abstractmethod
    async def get_raw_data_for_episode(self, episode_id: str) -> list[str]:
        """
        Get raw data IDs that contributed to an episode.

        Args:
            episode_id: The episode identifier

        Returns:
            List of raw data identifiers
        """
        pass

    @abstractmethod
    async def link_related_episodes(self, episode_id1: str, episode_id2: str) -> bool:
        """
        Create bidirectional relationship between episodes.

        Args:
            episode_id1: First episode identifier
            episode_id2: Second episode identifier

        Returns:
            True if linking was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_related_episodes(self, episode_id: str) -> list[Episode]:
        """
        Get episodes related to a specific episode.

        Args:
            episode_id: The episode identifier

        Returns:
            List of related episodes
        """
        pass

    @abstractmethod
    async def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode.

        Args:
            episode_id: The unique identifier of the episode

        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    async def cleanup_old_episodes(self, max_age_days: int) -> int:
        """
        Clean up episodes older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of episodes deleted
        """
        pass
