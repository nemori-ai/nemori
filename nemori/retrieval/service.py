"""
Main retrieval service for Nemori episodic memory.

This module provides the high-level retrieval service that coordinates
different retrieval providers and can combine results from multiple strategies.
"""

from typing import Any

from ..core.data_types import SemanticNode
from ..core.episode import Episode
from ..storage.repository import EpisodicMemoryRepository, SemanticMemoryRepository
from ..storage.storage_types import EpisodeQuery
from .providers import BM25RetrievalProvider, RetrievalProvider
from .retrieval_types import RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy


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


class UnifiedRetrievalService:
    """
    Unified service providing similarity-based retrieval for both episodic and semantic memories.

    This service implements the core retrieval requirements:
    1. Independent similarity search for episodic and semantic memories
    2. Bidirectional ID-based associations between episodes and semantic nodes
    3. Context-aware retrieval for semantic discovery
    """

    def __init__(self, episode_repository: EpisodicMemoryRepository, semantic_repository: SemanticMemoryRepository):
        """
        Initialize unified retrieval service.

        Args:
            episode_repository: Repository for episodic memory access
            semantic_repository: Repository for semantic memory access
        """
        self.episode_repository = episode_repository
        self.semantic_repository = semantic_repository

    # === Independent Similarity Search ===

    async def search_episodic_memories(self, owner_id: str, query: str, limit: int = 10) -> list[Episode]:
        """
        Independent similarity search for episodic memories.

        Args:
            owner_id: The owner of the memories
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of episodes ranked by similarity to query
        """
        search_query = EpisodeQuery(owner_ids=[owner_id], text_search=query, limit=limit)
        search_result = await self.episode_repository.search_episodes(search_query)
        return search_result.episodes

    async def search_semantic_memories(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """
        Independent similarity search for semantic memories.

        Args:
            owner_id: The owner of the memories
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of semantic nodes ranked by similarity to query
        """
        return await self.semantic_repository.similarity_search_semantic_nodes(
            owner_id=owner_id, query=query, limit=limit
        )

    # === Bidirectional ID-based Associations ===

    async def get_episode_semantics(self, episode_id: str) -> list[SemanticNode]:
        """
        Get all semantic nodes discovered from or linked to a specific episode.

        Args:
            episode_id: The episode identifier

        Returns:
            List of semantic nodes associated with the episode
        """
        # Get nodes discovered from this episode
        discovered_nodes = await self.semantic_repository.find_semantic_nodes_by_episode(episode_id)

        # Get nodes that have this episode in their linked_episode_ids
        linked_nodes = await self.semantic_repository.find_semantic_nodes_by_linked_episode(episode_id)

        # Combine and deduplicate
        all_nodes = discovered_nodes + linked_nodes
        unique_nodes = {node.node_id: node for node in all_nodes}

        return list(unique_nodes.values())

    async def get_semantic_episodes(self, semantic_node_id: str) -> dict[str, list[Episode]]:
        """
        Get all episodes associated with a semantic node, including evolution history.

        Args:
            semantic_node_id: The semantic node identifier

        Returns:
            Dictionary containing:
            - "linked_episodes": Episodes that reference this knowledge
            - "evolution_episodes": Episodes that caused knowledge evolution
            - "discovery_episode": Episode that initially discovered this knowledge
        """
        semantic_node = await self.semantic_repository.get_semantic_node_by_id(semantic_node_id)
        if not semantic_node:
            return {"linked_episodes": [], "evolution_episodes": [], "discovery_episode": []}

        # Get linked episodes
        linked_episodes = []
        if semantic_node.linked_episode_ids:
            episodes_batch = await self.episode_repository.get_episode_batch(semantic_node.linked_episode_ids)
            linked_episodes = [ep for ep in episodes_batch if ep is not None]

        # Get evolution episodes
        evolution_episodes = []
        if semantic_node.evolution_episode_ids:
            episodes_batch = await self.episode_repository.get_episode_batch(semantic_node.evolution_episode_ids)
            evolution_episodes = [ep for ep in episodes_batch if ep is not None]

        # Get discovery episode
        discovery_episode = []
        if semantic_node.discovery_episode_id:
            discovery_ep = await self.episode_repository.get_episode(semantic_node.discovery_episode_id)
            if discovery_ep:
                discovery_episode = [discovery_ep]

        return {
            "linked_episodes": linked_episodes,
            "evolution_episodes": evolution_episodes,
            "discovery_episode": discovery_episode,
        }

    # === Context-Aware Retrieval ===

    async def get_discovery_context(
        self, episode: Episode, semantic_limit: int = 5, episode_limit: int = 3
    ) -> dict[str, Any]:
        """
        Gather related semantic memories and historical episodes for context-aware discovery.

        Args:
            episode: The episode to gather context for
            semantic_limit: Maximum number of related semantic memories to retrieve
            episode_limit: Maximum number of related historical episodes to retrieve

        Returns:
            Dictionary containing related memories and episodes for context
        """
        # Search for related semantic memories using episode content
        related_semantics = await self.search_semantic_memories(
            owner_id=episode.owner_id, query=f"{episode.title} {episode.summary}", limit=semantic_limit
        )

        # Search for related historical episodes
        related_episodes = await self.search_episodic_memories(
            owner_id=episode.owner_id, query=episode.content, limit=episode_limit
        )

        return {
            "related_semantic_memories": related_semantics,
            "related_historical_episodes": related_episodes,
            "current_episode": episode,
        }

    # === Evolution History Tracking ===

    async def get_semantic_evolution_history(self, semantic_node_id: str) -> dict[str, Any]:
        """
        Get comprehensive evolution history including all related episodes.

        Args:
            semantic_node_id: The semantic node identifier

        Returns:
            Dictionary containing complete evolution timeline with episodes
        """
        semantic_node = await self.semantic_repository.get_semantic_node_by_id(semantic_node_id)
        if not semantic_node:
            return {}

        # Build evolution timeline
        evolution_timeline = []

        # Add historical versions with their corresponding episodes
        for i, historical_value in enumerate(semantic_node.evolution_history):
            episode_id = (
                semantic_node.evolution_episode_ids[i] if i < len(semantic_node.evolution_episode_ids) else None
            )
            episode = None
            if episode_id:
                episode = await self.episode_repository.get_episode(episode_id)

            evolution_timeline.append(
                {
                    "version": i + 1,
                    "value": historical_value,
                    "episode": episode,
                    "timestamp": episode.temporal_info.timestamp if episode else semantic_node.created_at,
                }
            )

        # Add current version
        evolution_timeline.append(
            {
                "version": semantic_node.version,
                "value": semantic_node.value,
                "episode": None,  # Current version doesn't have a triggering episode
                "timestamp": semantic_node.last_updated,
            }
        )

        # Get associated episodes
        associated_episodes = await self.get_semantic_episodes(semantic_node_id)

        return {
            "node": semantic_node,
            "evolution_timeline": sorted(evolution_timeline, key=lambda x: x["version"]),
            "linked_episodes": associated_episodes["linked_episodes"],
            "evolution_episodes": associated_episodes["evolution_episodes"],
            "discovery_episode": associated_episodes["discovery_episode"],
        }

    # === Quality-based Retrieval ===

    async def get_memory_for_query(
        self, owner_id: str, query: str, quality_preference: str = "balanced", limit: int = 10
    ) -> dict[str, Any]:
        """
        Get memory with specified quality preference for business use cases.

        Args:
            owner_id: The owner of the memories
            query: The search query
            quality_preference: One of "factual", "contextual", "comprehensive", "balanced"
            limit: Maximum number of results per category

        Returns:
            Dictionary containing memory results based on quality preference
        """
        if quality_preference == "factual":
            # Prioritize semantic memories for precise facts
            return {
                "primary": await self.search_semantic_memories(owner_id, query, limit),
                "secondary": [],
                "quality_type": "factual",
            }

        elif quality_preference == "contextual":
            # Prioritize episodic memories for rich context
            return {
                "primary": await self.search_episodic_memories(owner_id, query, limit),
                "secondary": [],
                "quality_type": "contextual",
            }

        elif quality_preference == "comprehensive":
            # Combine both types with bidirectional linking
            semantic_results = await self.search_semantic_memories(owner_id, query, limit)
            episodic_results = await self.search_episodic_memories(owner_id, query, limit)

            # Enrich semantic results with associated episodes
            enriched_results = []
            for semantic_node in semantic_results:
                associated_episodes = await self.get_semantic_episodes(semantic_node.node_id)
                enriched_results.append(
                    {
                        "semantic_knowledge": semantic_node,
                        "supporting_episodes": associated_episodes["linked_episodes"],
                        "evolution_context": associated_episodes["evolution_episodes"],
                    }
                )

            return {"primary": enriched_results, "secondary": episodic_results, "quality_type": "comprehensive"}

        else:  # balanced
            # Return both types separately for business logic to decide
            semantic_results = await self.search_semantic_memories(owner_id, query, limit)
            episodic_results = await self.search_episodic_memories(owner_id, query, limit)

            return {
                "semantic_memories": semantic_results,
                "episodic_memories": episodic_results,
                "quality_type": "balanced",
            }

    # === Utility Methods ===

    async def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> dict[str, list[SemanticNode]]:
        """
        Get related knowledge nodes through relationship traversal.

        Args:
            node_id: The starting semantic node identifier
            max_depth: Maximum relationship traversal depth

        Returns:
            Dictionary containing direct and indirect related nodes
        """
        visited = {node_id}
        result = {"direct": [], "indirect": []}

        # Get direct relationships
        direct_related = await self.semantic_repository.find_relationships_for_node(node_id)
        result["direct"] = [node for node, _ in direct_related]

        for node, _ in direct_related:
            visited.add(node.node_id)

        # Get indirect relationships if depth > 1
        if max_depth > 1:
            for node, _ in direct_related:
                indirect_related = await self.semantic_repository.find_relationships_for_node(node.node_id)
                for indirect_node, _ in indirect_related:
                    if indirect_node.node_id not in visited:
                        result["indirect"].append(indirect_node)
                        visited.add(indirect_node.node_id)

        return result
