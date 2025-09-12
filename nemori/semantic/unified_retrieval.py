"""
Unified retrieval service providing similarity-based retrieval for both episodic and semantic memories.

This module implements the dual retrieval capability that allows independent
searches of episodic and semantic memories with bidirectional associations.
"""

from typing import Any

from ..core.data_types import SemanticNode
from ..core.episode import Episode
from ..storage.repository import EpisodicMemoryRepository, SemanticMemoryRepository
from ..storage.storage_types import SemanticNodeQuery, SortOrder
from ..retrieval.retrieval_types import RetrievalConfig, RetrievalQuery, RetrievalResult, RetrievalStrategy
from ..retrieval.providers import BM25RetrievalProvider, RetrievalProvider, EmbeddingRetrievalProvider
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

        self._initialized = True

    async def close(self) -> None:
        """Close the service and all providers."""
        self._initialized = False

    def register_provider(self, strategy: RetrievalStrategy, config: RetrievalConfig) -> None:
        """
        Register a retrieval provider for a specific strategy.

        Args:
            strategy: The retrieval strategy
            config: Configuration for the provider
        """
        if strategy == RetrievalStrategy.BM25:
            provider = BM25RetrievalProvider(config, self.episodic_storage)
        elif strategy == RetrievalStrategy.EMBEDDING:
            provider = EmbeddingRetrievalProvider(config, self.episodic_storage)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        self.providers[strategy] = provider

    def get_provider(self, strategy: RetrievalStrategy) -> RetrievalProvider | None:
        """Get a registered provider by strategy."""
        return self.providers.get(strategy)
    
    async def search_episodic_memories(self, owner_id: str, query: str, limit: int = 10) -> list[Episode]:
        """
        Independent similarity search for episodic memories.
        情景记忆的独立相似度搜索。
        """
        try:
            # Simple text search for episodic memories
            episode_query = RetrievalQuery(text=query, owner_id=owner_id, limit=limit, strategy=RetrievalStrategy.EMBEDDING)
            if not self._initialized:
                print(f"⚠️ UnifiedRetrievalService not initialized for episodic search")
                raise RuntimeError("RetrievalService not initialized. Call initialize() first.")

            provider = self.providers.get(RetrievalStrategy.EMBEDDING)
            if not provider:
                print(f"⚠️ No EMBEDDING provider found for episodic search")
                return []

            print(f"🔍 Searching episodic memories for owner {owner_id} with query: '{query[:50]}...'")
            
            # Delegate to the appropriate provider
            result = await provider.search(episode_query)
            
            # Handle the result properly
            if hasattr(result, 'episodes'):
                episodes = result.episodes
                print(f"   ✅ Found {len(episodes)} episodic memories")
                return episodes
            else:
                episodes = result if result else []
                print(f"   ✅ Found {len(episodes)} episodic memories (direct result)")
                return episodes
                
        except Exception as e:
            print(f"❌ Error searching episodic memories: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def search_semantic_memories(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """
        Independent similarity search for semantic memories.
        语义记忆的独立相似度搜索。
        """
        try:
            print(f"🔍 UnifiedRetrievalService searching semantic memories for '{owner_id}' with query: '{query[:50]}...'")
            
            # Direct call to the storage layer's similarity search
            results = await self.semantic_storage.similarity_search_semantic_nodes(
                owner_id=owner_id, query=query, limit=limit
            )
            
            if results:
                print(f"   ✅ UnifiedRetrievalService found {len(results)} semantic results")
                for i, node in enumerate(results[:2]):  # Show first 2
                    print(f"      {i+1}. {node.key}: {node.value[:50]}...")
            else:
                print(f"   ❌ UnifiedRetrievalService found no semantic results")
                
            return results
            
        except Exception as e:
            print(f"   ❌ Error in UnifiedRetrievalService semantic search: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_episode_semantics(self, episode_id: str) -> list[SemanticNode]:
        """
        Get all semantic nodes discovered from a specific episode.
        获取从特定情景发现的所有语义节点。
        """
        try:
            return await self.semantic_storage.find_semantic_nodes_by_episode(episode_id)
        except Exception as e:
            print(f"Error getting episode semantics: {e}")
            return []
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
        try:
            semantic_node = await self.semantic_storage.get_semantic_node_by_id(semantic_node_id)
            if not semantic_node:
                return {"linked_episodes": [], "evolution_episodes": []}
            # Get linked episodes
            linked_episodes = []
            if semantic_node.linked_episode_ids:
                linked_episodes = await self.get_episodes_by_ids(semantic_node.linked_episode_ids)
            # Get evolution episodes
            evolution_episodes = []
            if semantic_node.evolution_episode_ids:
                evolution_episodes = await self.get_episodes_by_ids(semantic_node.evolution_episode_ids)

            return {"linked_episodes": linked_episodes, "evolution_episodes": evolution_episodes}

        except Exception as e:
            print(f"Error getting semantic episodes: {e}")
            return {"linked_episodes": [], "evolution_episodes": []}

    async def get_episodes_by_ids(self, episode_ids: list[str]) -> list[Episode]:
        """
        Get episodes by their IDs.
        通过ID获取情景。
        """
        try:
            episodes = await self.episodic_storage.get_episode_batch(episode_ids)
            return [ep for ep in episodes if ep is not None]
        except Exception as e:
            print(f"Error getting episodes by IDs: {e}")
            return []

    async def enhanced_query(
        self,
        owner_id: str,
        query: str,
        include_semantic: bool = True,
        episode_limit: int = 10,
        semantic_limit: int = 10,
    ) -> dict[str, Any]:
        """
        Enhanced query combining episodic and semantic results.
        结合情景和语义结果的增强查询。
        """
        try:
            results = {
                "episodes": await self.search_episodic_memories(owner_id, query, episode_limit),
                "semantic_knowledge": [],
            }

            if include_semantic:
                results["semantic_knowledge"] = await self.search_semantic_memories(owner_id, query, semantic_limit)

            return results

        except Exception as e:
            print(f"Error in enhanced query: {e}")
            return {"episodes": [], "semantic_knowledge": []}

    async def get_contextual_knowledge(
        self, episode: Episode, semantic_limit: int = 5, episode_limit: int = 3
    ) -> dict[str, Any]:
        """
        Get contextual knowledge for an episode (related semantics and episodes).
        获取情景的上下文知识（相关语义和情景）。
        """
        try:
            # Get semantic knowledge related to this episode
            episode_semantics = await self.get_episode_semantics(episode.episode_id)

            # Search for related semantic memories based on episode content
            related_semantics = await self.search_semantic_memories(
                owner_id=episode.owner_id, query=f"{episode.title} {episode.summary}", limit=semantic_limit
            )

            # Remove duplicates
            seen_ids = {node.node_id for node in episode_semantics}
            related_semantics = [node for node in related_semantics if node.node_id not in seen_ids]

            # Search for related episodes
            related_episodes = await self.search_episodic_memories(
                owner_id=episode.owner_id,
                query=episode.content,
                limit=episode_limit + 1,  # +1 to account for filtering out current episode
            )

            # Filter out the current episode
            related_episodes = [ep for ep in related_episodes if ep.episode_id != episode.episode_id][:episode_limit]

            return {
                "episode_semantics": episode_semantics,
                "related_semantics": related_semantics,
                "related_episodes": related_episodes,
                "context_summary": self._build_context_summary(episode_semantics, related_semantics, related_episodes),
            }

        except Exception as e:
            print(f"Error getting contextual knowledge: {e}")
            return {"episode_semantics": [], "related_semantics": [], "related_episodes": [], "context_summary": ""}

    def _build_context_summary(
        self,
        episode_semantics: list[SemanticNode],
        related_semantics: list[SemanticNode],
        related_episodes: list[Episode],
    ) -> str:
        """Build a text summary of the contextual knowledge."""
        summary_parts = []

        if episode_semantics:
            keys = [node.key for node in episode_semantics[:3]]  # Top 3
            summary_parts.append(f"Direct knowledge: {', '.join(keys)}")

        if related_semantics:
            keys = [node.key for node in related_semantics[:3]]  # Top 3
            summary_parts.append(f"Related knowledge: {', '.join(keys)}")

        if related_episodes:
            titles = [ep.title for ep in related_episodes[:2]]  # Top 2
            summary_parts.append(f"Related episodes: {', '.join(titles)}")

        return "; ".join(summary_parts) if summary_parts else "No contextual knowledge available"

    async def search_knowledge_by_type(
        self, owner_id: str, knowledge_type: str, query: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """
        Search for specific types of knowledge (facts, preferences, definitions, etc.).
        搜索特定类型的知识（事实、偏好、定义等）。
        """
        try:
            # Use semantic search with type-specific filtering
            semantic_query = SemanticNodeQuery(
                owner_id=owner_id, text_search=query, limit=limit, sort_by="confidence", sort_order=SortOrder.DESC
            )

            if knowledge_type == "facts":
                # Look for factual knowledge (high confidence, specific values)
                semantic_query.min_confidence = 0.8
            elif knowledge_type == "preferences":
                # Look for preference-related knowledge
                semantic_query.key_pattern = "%偏好%|%喜欢%|%preference%|%like%"
            elif knowledge_type == "definitions":
                # Look for definitional knowledge
                semantic_query.key_pattern = "%定义%|%definition%|%是什么%|%what is%"

            result = await self.semantic_storage.search_semantic_nodes(semantic_query)

            return {
                "knowledge_type": knowledge_type,
                "nodes": result.semantic_nodes,
                "total_count": result.total_nodes,
                "query_time_ms": result.query_time_ms,
            }

        except Exception as e:
            print(f"Error searching knowledge by type: {e}")
            return {"knowledge_type": knowledge_type, "nodes": [], "total_count": 0, "query_time_ms": 0.0}


class AdaptiveMemoryService:
    """
    Adaptive service that provides different memory qualities based on business needs.
    根据业务需求提供不同质量记忆的自适应服务。
    """

    def __init__(self, unified_retrieval: UnifiedRetrievalService):
        self.unified_retrieval = unified_retrieval

    async def get_memory_for_query(
        self, owner_id: str, query: str, quality_preference: str = "balanced"
    ) -> dict[str, Any]:
        """
        Get memory with specified quality preference.
        获取指定质量偏好的记忆。
        """
        if quality_preference == "factual":
            # 需要准确事实：优先语义记忆
            return {"primary": await self.unified_retrieval.search_semantic_memories(owner_id, query), "secondary": []}

        elif quality_preference == "contextual":
            # 需要丰富上下文：优先情景记忆
            return {"primary": await self.unified_retrieval.search_episodic_memories(owner_id, query), "secondary": []}

        elif quality_preference == "comprehensive":
            # 需要全面信息：组合两种记忆
            semantic_results = await self.unified_retrieval.search_semantic_memories(owner_id, query)
            episodic_results = await self.unified_retrieval.search_episodic_memories(owner_id, query)
            # 对于每个语义节点，获取其关联的情景
            enriched_results = []
            for semantic_node in semantic_results:
                associated_episodes = await self.unified_retrieval.get_semantic_episodes(semantic_node.node_id)
                enriched_results.append(
                    {
                        "semantic_knowledge": semantic_node,
                        "supporting_episodes": associated_episodes["linked_episodes"],
                        "evolution_context": associated_episodes["evolution_episodes"],
                    }
                )

            return {"primary": enriched_results, "secondary": episodic_results}

        else:  # balanced
            # 平衡模式：基于查询类型自动选择
            return await self._smart_memory_selection(owner_id, query)

    async def _smart_memory_selection(self, owner_id: str, query: str) -> dict[str, Any]:
        """
        Smart memory selection based on query characteristics.
        基于查询特征的智能记忆选择。
        """
        # Simple heuristic: if query contains question words, prioritize factual
        question_indicators = [
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "什么",
            "谁",
            "什么时候",
            "哪里",
            "为什么",
            "怎么",
        ]
        is_factual_query = any(indicator in query.lower() for indicator in question_indicators)

        if is_factual_query:
            semantic_results = await self.unified_retrieval.search_semantic_memories(owner_id, query, limit=5)
            episodic_results = await self.unified_retrieval.search_episodic_memories(owner_id, query, limit=3)

            return {
                "primary": semantic_results,
                "secondary": episodic_results,
                "selection_reason": "factual_query_detected",
            }
        else:
            # For non-factual queries, balance both types
            enhanced_results = await self.unified_retrieval.enhanced_query(owner_id, query)

            return {
                "primary": enhanced_results["episodes"],
                "secondary": enhanced_results["semantic_knowledge"],
                "selection_reason": "balanced_approach",
            }
