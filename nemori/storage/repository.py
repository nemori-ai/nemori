"""
Repository interfaces for storage layer in Nemori.

This module defines abstract repository interfaces for managing raw event data,
episodes, and their relationships in the episodic memory system.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..core.data_types import DataType, RawEventData, SemanticNode, SemanticRelationship
from ..core.episode import Episode
from .storage_types import (
    EpisodeQuery,
    EpisodeSearchResult,
    RawDataQuery,
    RawDataSearchResult,
    SemanticNodeQuery,
    SemanticRelationshipQuery,
    SemanticSearchResult,
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


class SemanticMemoryRepository(StorageRepository):
    """
    Repository interface for semantic memory storage.

    Handles storage and retrieval of semantic nodes and relationships,
    supporting the core requirements of the semantic memory system.
    """

    # === Semantic Node Operations ===

    @abstractmethod
    async def store_semantic_node(self, node: SemanticNode) -> None:
        """
        Store a semantic node.

        Args:
            node: The semantic node to store

        Raises:
            DuplicateKeyError: If node with same (owner_id, key) already exists
            SemanticStorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        """
        Retrieve a semantic node by its ID.

        Args:
            node_id: The unique identifier of the semantic node

        Returns:
            The semantic node if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        """
        Find semantic node by owner and key combination.

        Args:
            owner_id: The owner of the semantic knowledge
            key: The knowledge key/identifier

        Returns:
            The semantic node if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_semantic_node(self, node: SemanticNode) -> None:
        """
        Update an existing semantic node.

        Args:
            node: The updated semantic node

        Raises:
            NotFoundError: If node doesn't exist
            SemanticStorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def delete_semantic_node(self, node_id: str) -> bool:
        """
        Delete a semantic node by ID.

        Args:
            node_id: The unique identifier of the semantic node

        Returns:
            True if node was deleted, False if not found
        """
        pass

    @abstractmethod
    async def search_semantic_nodes(self, query: SemanticNodeQuery) -> SemanticSearchResult:
        """
        Search semantic nodes with complex query parameters.

        Args:
            query: The search query parameters

        Returns:
            Search results containing matching semantic nodes
        """
        pass

    @abstractmethod
    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """
        Search semantic nodes by similarity to query text.

        Args:
            owner_id: The owner of the semantic knowledge
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of semantic nodes ranked by similarity
        """
        pass

    @abstractmethod
    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        """
        Find all semantic nodes discovered from a specific episode.

        Args:
            episode_id: The episode identifier

        Returns:
            List of semantic nodes discovered from the episode
        """
        pass
    
    @abstractmethod
    async def store_semantic_node_with_embedding(self, node: SemanticNode, content_for_embedding: str | None = None) -> None:
        """Store semantic node and generate embedding if possible."""
        pass

    @abstractmethod
    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        """
        Find all semantic nodes that have the episode in their linked_episode_ids.

        Args:
            episode_id: The episode identifier

        Returns:
            List of semantic nodes linked to the episode
        """
        pass

    # === Semantic Relationship Operations ===

    @abstractmethod
    async def store_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """
        Store a semantic relationship.

        Args:
            relationship: The semantic relationship to store

        Raises:
            SemanticStorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        """
        Retrieve a semantic relationship by its ID.

        Args:
            relationship_id: The unique identifier of the relationship

        Returns:
            The semantic relationship if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """
        Find all relationships and related nodes for a given semantic node.

        Args:
            node_id: The semantic node identifier

        Returns:
            List of tuples containing (related_node, relationship)
        """
        pass

    @abstractmethod
    async def search_semantic_relationships(self, query: SemanticRelationshipQuery) -> SemanticSearchResult:
        """
        Search semantic relationships with complex query parameters.

        Args:
            query: The search query parameters

        Returns:
            Search results containing matching semantic relationships
        """
        pass

    @abstractmethod
    async def update_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """
        Update an existing semantic relationship.

        Args:
            relationship: The updated semantic relationship

        Raises:
            NotFoundError: If relationship doesn't exist
            SemanticStorageError: If storage operation fails
        """
        pass

    @abstractmethod
    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        """
        Delete a semantic relationship by ID.

        Args:
            relationship_id: The unique identifier of the relationship

        Returns:
            True if relationship was deleted, False if not found
        """
        pass

    # === Bulk Operations ===

    @abstractmethod
    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        """
        Retrieve multiple semantic nodes by their IDs.

        Args:
            node_ids: List of semantic node identifiers

        Returns:
            List of semantic nodes (may be shorter than input if some not found)
        """
        pass

    @abstractmethod
    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        """
        Retrieve all semantic nodes for a specific owner.

        Args:
            owner_id: The owner identifier

        Returns:
            List of all semantic nodes owned by the user
        """
        pass

    # === Statistics and Maintenance ===

    @abstractmethod
    async def get_semantic_statistics(self, owner_id: str) -> dict[str, Any]:
        """
        Get statistics about semantic memory for an owner.

        Args:
            owner_id: The owner identifier

        Returns:
            Dictionary containing statistics like node count, relationship count, etc.
        """
        pass

    @abstractmethod
    async def cleanup_orphaned_relationships(self) -> int:
        """
        Clean up relationships that reference non-existent nodes.

        Returns:
            Number of orphaned relationships cleaned up
        """
        pass
