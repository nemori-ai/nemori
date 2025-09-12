"""
Unified retrieval service providing similarity-based retrieval for both episodic and semantic memories.

This module implements the dual retrieval capability that allows independent
searches of episodic and semantic memories with bidirectional associations.
"""

from typing import Any, Dict, List

from ..core.data_types import SemanticNode
from ..core.episode import Episode
from ..retrieval.retrieval_types import (
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)
from ..storage.repository import EpisodicMemoryRepository, SemanticMemoryRepository
from ..storage.storage_types import SemanticNodeQuery, SortOrder
from ..retrieval.service import RetrievalService
from .providers import BM25RetrievalProvider, RetrievalProvider, EmbeddingRetrievalProvider
from .retrieval_types import RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy


class UnifiedRetrievalService:
    """
    Unified service providing similarity-based retrieval for both episodic and semantic memories.
    为情景记忆和语义记忆提供相似度检索的统一服务。
    """

    def __init__(
        self,
        episodic_storage: EpisodicMemoryRepository,
        semantic_storage: SemanticMemoryRepository,
    ):
        self.episodic_storage = episodic_storage
        self.semantic_storage = semantic_storage
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

    def register_provider(self, strategy: RetrievalStrategy, config: RetrievalConfig, llm_provider=None) -> None:
        """
        Register a retrieval provider for a specific strategy.

        Args:
            strategy: The retrieval strategy
            config: Configuration for the provider
            llm_provider: Optional LLM provider (required for enhanced embedding)
        """
        if strategy == RetrievalStrategy.EMBEDDING:
            # For unified service, we need specialized providers
            provider_episodic = EmbeddingRetrievalProvider(config, self.episodic_storage)
            
            # Import and use semantic-specific embedding provider
            from .providers.semantic_embedding_provider import SemanticEmbeddingProvider
            provider_semantic = SemanticEmbeddingProvider(
                semantic_storage=self.semantic_storage,
                api_key=config.api_key or "EMPTY",
                base_url=config.base_url or "http://localhost:6007/v1", 
                embed_model=config.embed_model or "qwen3-emb",
                persistence_dir=config.storage_config.get("directory") if config.storage_config else None,
                enable_persistence=True
            )
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        self.providers[strategy] = {
            'episodic': provider_episodic,
            'semantic': provider_semantic
        }

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

        provider_episodic, provider_semantic = self.providers.get(query.strategy)
        if not provider_episodic and not provider_semantic:
            raise ValueError(f"No provider registered for strategy: {query.strategy}")
        episodic_result = await provider_episodic.search(query)
        semantic_result = await provider_semantic.search(query)
        # Delegate to the appropriate provider
        return [episodic_result, semantic_result]

    async def search_episodic_memories(self, query: RetrievalQuery) -> RetrievalResult:
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

        provider_episodic, provider_semantic = self.providers.get(query.strategy)
        if not provider_episodic and not provider_semantic:
            raise ValueError(f"No provider registered for strategy: {query.strategy}")
        episodic_result = await provider_episodic.search(query)
        # Delegate to the appropriate provider
        return episodic_result

    async def search_semantic_memories(self, query: RetrievalQuery) -> RetrievalResult:
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

        provider_episodic, provider_semantic = self.providers.get(query.strategy)
        if not provider_episodic and not provider_semantic:
            raise ValueError(f"No provider registered for strategy: {query.strategy}")
        semantic_result = await provider_semantic.search(query)
        # Delegate to the appropriate provider
        return semantic_result

    async def get_episode_semantics(self, episode_id: str) -> list[SemanticNode]:
        """
        Get all semantic nodes discovered from a specific episode.
        获取从特定情景发现的所有语义节点。
        """
        return await self.semantic_storage.find_by_discovery_episode(episode_id)

    async def get_semantic_episodes(self, semantic_node_id: str) -> dict[str, list[Episode]]:
        """
        Get all episodes associated with a semantic node, including evolution history.
        获取与语义节点关联的所有情景，包括演变历史。

        Returns:
            {
                "linked_episodes": [episodes that reference this knowledge],
                "evolution_episodes": [episodes that caused knowledge evolution]
            }
        """
        semantic_node = await self.semantic_storage.get_by_id(semantic_node_id)
        if not semantic_node:
            return {"linked_episodes": [], "evolution_episodes": []}

        linked_episodes = await self.episodic_storage.get_by_ids(semantic_node.linked_episode_ids)

        evolution_episodes = await self.episodic_storage.get_by_ids(semantic_node.evolution_episode_ids)

        return {"linked_episodes": linked_episodes, "evolution_episodes": evolution_episodes}
