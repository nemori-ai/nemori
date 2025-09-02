"""
Episode Manager for Nemori - Coordinates storage, retrieval, and episode building.

This module provides the high-level interface for managing episodes throughout
their lifecycle, from creation via builders to storage and indexing for retrieval.
"""

from typing import Any

from .core.builders import EpisodeBuilderRegistry
from .core.data_types import RawEventData
from .core.episode import Episode
from .retrieval import RetrievalService
from .storage.repository import EpisodicMemoryRepository, RawDataRepository


class EpisodeManager:
    """
    High-level manager that coordinates episode creation, storage, and retrieval.

    This class provides the main interface for:
    - Building episodes from raw data using appropriate builders
    - Storing episodes in the storage layer
    - Automatically indexing episodes for retrieval
    - Managing the complete episode lifecycle
    """

    def __init__(
        self,
        raw_data_repo: RawDataRepository,
        episode_repo: EpisodicMemoryRepository,
        builder_registry: EpisodeBuilderRegistry,
        retrieval_service: RetrievalService | None = None,
    ):
        """
        Initialize the episode manager.

        Args:
            raw_data_repo: Repository for raw data storage
            episode_repo: Repository for episode storage
            builder_registry: Registry of episode builders
            retrieval_service: Optional retrieval service for indexing
        """
        self.raw_data_repo = raw_data_repo
        self.episode_repo = episode_repo
        self.builder_registry = builder_registry
        self.retrieval_service = retrieval_service

    async def ingest_event(self, raw_data: RawEventData, owner_id: str) -> None:
        """
        Ingest raw event data into storage without immediate processing.

        Args:
            raw_data: Raw event data to store
            owner_id: Owner of the event data
        """
        raw_data.metadata["_owner_id"] = owner_id
        await self.raw_data_repo.store_raw_data(raw_data)

    async def process_raw_data_to_episode(
        self, raw_data: RawEventData, owner_id: str, auto_index: bool = True
    ) -> Episode | None:
        """
        Process raw data into an episode with full lifecycle management.

        Args:
            raw_data: Raw event data to process
            owner_id: Owner of the episode
            auto_index: Whether to automatically add to retrieval index

        Returns:
            Created episode or None if no suitable builder found
        """
        # 1. Build episode using appropriate builder
        episode = await self.builder_registry.build_episode(raw_data, owner_id)
        if not episode:
            print(f"No builder available for data type: {raw_data.data_type}")
            return None

        # 2. Store episode
        episode_id = await self.episode_repo.store_episode(episode)
        print(f"Stored episode with ID: {episode_id}")

        # 3. Link episode to raw data
        await self.episode_repo.link_episode_to_raw_data(episode_id, [raw_data.data_id])

        # 4. Mark raw data as processed
        await self.raw_data_repo.mark_as_processed(raw_data.data_id, "1.0")

        # 5. Add to retrieval index if service is available
        if auto_index and self.retrieval_service:
            await self.retrieval_service.add_episode_to_all_providers(episode)
            print(f"Added episode to retrieval index: {episode.title[:50]}...")
          
        return episode

    async def process_raw_data(self, raw_data: RawEventData, owner_id: str, auto_index: bool = True) -> Episode | None:
        """
        Process raw data into an episode with full lifecycle management (legacy method).

        Args:
            raw_data: Raw event data to process
            owner_id: Owner of the episode
            auto_index: Whether to automatically add to retrieval index

        Returns:
            Created episode or None if no suitable builder found
        """
        # 1. Store raw data first
        await self.ingest_event(raw_data, owner_id)

        # 2. Then process it into episode
        return await self.process_raw_data_to_episode(raw_data, owner_id, auto_index)

    async def create_episode(self, episode: Episode, auto_index: bool = True) -> str:
        """
        Create and store an episode with automatic indexing.

        Args:
            episode: Episode to create
            auto_index: Whether to automatically add to retrieval index

        Returns:
            Episode ID of the created episode
        """
        # Store episode
        episode_id = await self.episode_repo.store_episode(episode)

        # Add to retrieval index if service is available
        if auto_index and self.retrieval_service:
            try:
                await self.retrieval_service.add_episode_to_all_providers(episode)
                print(f"Added episode to retrieval index: {episode.title[:50]}...")
            except Exception as e:
                print(f"Failed to add episode to retrieval index: {e}")

        return episode_id

    async def update_episode(self, episode_id: str, updated_episode: Episode, auto_reindex: bool = True) -> bool:
        """
        Update an episode and reindex if needed.

        Args:
            episode_id: ID of episode to update
            updated_episode: Updated episode data
            auto_reindex: Whether to automatically update retrieval index

        Returns:
            True if update was successful
        """
        # Update in storage
        success = await self.episode_repo.update_episode(episode_id, updated_episode)

        if success and auto_reindex and self.retrieval_service:
            try:
                # Update in retrieval index
                await self.retrieval_service.update_episode_in_all_providers(updated_episode)
                print(f"Updated episode in retrieval index: {updated_episode.title[:50]}...")
            except Exception as e:
                print(f"Failed to update episode in retrieval index: {e}")

        return success

    async def delete_episode(self, episode_id: str, auto_remove_from_index: bool = True) -> bool:
        """
        Delete an episode and remove from index.

        Args:
            episode_id: ID of episode to delete
            auto_remove_from_index: Whether to automatically remove from retrieval index

        Returns:
            True if deletion was successful
        """
        # Remove from retrieval index first
        if auto_remove_from_index and self.retrieval_service:
            try:
                await self.retrieval_service.remove_episode_from_all_providers(episode_id)
                print(f"Removed episode from retrieval index: {episode_id}")
            except Exception as e:
                print(f"Failed to remove episode from retrieval index: {e}")

        # Delete from storage
        success = await self.episode_repo.delete_episode(episode_id)
        return success

    async def search_episodes(self, query_text: str, owner_id: str, **kwargs) -> Any:
        """
        Search for episodes using the retrieval service.

        Args:
            query_text: Text to search for
            owner_id: Owner of episodes to search
            **kwargs: Additional query parameters

        Returns:
            Search results from retrieval service
        """
        if not self.retrieval_service:
            raise RuntimeError("No retrieval service configured")

        from .retrieval import RetrievalQuery, RetrievalStrategy

        # Create retrieval query with defaults
        query = RetrievalQuery(
            text=query_text,
            owner_id=owner_id,
            strategy=kwargs.get("strategy", RetrievalStrategy.BM25),
            limit=kwargs.get("limit", 10),
            episode_types=kwargs.get("episode_types"),
            time_range_hours=kwargs.get("time_range_hours"),
            min_importance=kwargs.get("min_importance"),
        )

        return await self.retrieval_service.search(query)

    async def get_episode(self, episode_id: str, mark_accessed: bool = True) -> Episode | None:
        """
        Get an episode by ID with optional access tracking.

        Args:
            episode_id: ID of episode to retrieve
            mark_accessed: Whether to mark the episode as accessed

        Returns:
            Episode or None if not found
        """
        episode = await self.episode_repo.get_episode(episode_id)

        if episode and mark_accessed:
            # Mark as accessed in storage
            await self.episode_repo.mark_episode_accessed(episode_id)

            # Update access tracking in episode object
            episode.mark_accessed()

        return episode

    async def get_episodes_by_owner(self, owner_id: str, **kwargs) -> Any:
        """
        Get all episodes for an owner.

        Args:
            owner_id: Owner ID
            **kwargs: Additional query parameters (limit, offset)

        Returns:
            Episode search result
        """
        return await self.episode_repo.get_episodes_by_owner(
            owner_id, limit=kwargs.get("limit"), offset=kwargs.get("offset")
        )

    async def initialize_retrieval_index(self, owner_id: str | None = None) -> None:
        """
        Initialize or rebuild the retrieval index from existing episodes.

        Args:
            owner_id: Optional owner ID to rebuild index for specific user only
        """
        if not self.retrieval_service:
            print("No retrieval service configured - skipping index initialization")
            return

        print("Initializing retrieval index from existing episodes...")

        if owner_id:
            # Rebuild for specific user
            result = await self.episode_repo.get_episodes_by_owner(owner_id)
            episodes = result.episodes
            print(f"Found {len(episodes)} episodes for user {owner_id}")
        else:
            # TODO: Implement get_all_episodes across all users
            print("Rebuilding index for all users not yet supported")
            return

        # Add episodes to index in batches
        if episodes:
            try:
                # Group episodes by owner for efficient processing
                episodes_by_owner = {}
                for episode in episodes:
                    if episode.owner_id not in episodes_by_owner:
                        episodes_by_owner[episode.owner_id] = []
                    episodes_by_owner[episode.owner_id].append(episode)

                # Add each user's episodes to the index
                for user_id, user_episodes in episodes_by_owner.items():
                    print(f"Adding {len(user_episodes)} episodes for user {user_id} to index...")
                    for episode in user_episodes:
                        await self.retrieval_service.add_episode_to_all_providers(episode)

                print(f"Successfully added {len(episodes)} episodes to retrieval index")

            except Exception as e:
                print(f"Failed to initialize retrieval index: {e}")

    async def get_retrieval_stats(self) -> dict[str, Any]:
        """Get statistics from the retrieval service."""
        if not self.retrieval_service:
            return {}

        return await self.retrieval_service.get_all_stats()

    async def health_check(self) -> dict[str, bool]:
        """
        Perform health check on all components.

        Returns:
            Dictionary with health status of each component
        """
        health = {}

        # Check storage repositories
        try:
            health["raw_data_storage"] = await self.raw_data_repo.health_check()
        except Exception:
            health["raw_data_storage"] = False

        try:
            health["episode_storage"] = await self.episode_repo.health_check()
        except Exception:
            health["episode_storage"] = False

        # Check retrieval service
        if self.retrieval_service:
            try:
                retrieval_health = await self.retrieval_service.health_check()
                health.update({f"retrieval_{k}": v for k, v in retrieval_health.items()})
            except Exception:
                health["retrieval_service"] = False
        else:
            health["retrieval_service"] = None  # Not configured

        return health
