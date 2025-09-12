"""
Semantic evolution manager for handling knowledge updates and evolution.

This module manages the evolution of semantic knowledge over time, handling
updates, conflicts, and maintaining bidirectional episode associations.
"""

from dataclasses import replace
from datetime import datetime
from typing import Any

from ..core.data_types import SemanticNode
from ..core.episode import Episode
from .discovery import ContextAwareSemanticDiscoveryEngine
from .relationship_discovery import RelationshipDiscoveryEngine
from ..storage.repository import EpisodicMemoryRepository, SemanticMemoryRepository
from .unified_retrieval import UnifiedRetrievalService


class SemanticEvolutionManager:
    """
    Manages semantic memory evolution with bidirectional episode associations.
    管理语义记忆演变及双向情景关联。
    """

    def __init__(
        self,
        storage: SemanticMemoryRepository,
        discovery_engine: ContextAwareSemanticDiscoveryEngine,
        retrieval_service: UnifiedRetrievalService,
    ):
        self.storage = storage
        self.discovery_engine = discovery_engine
        self.retrieval_service = retrieval_service

    async def process_episode_for_semantics(self, episode: Episode, original_content: str) -> list[SemanticNode]:
        """
        Process an episode to discover and update semantic knowledge with full context.
        处理情景以发现和更新语义知识，包含完整上下文。
        """
        # Discover new semantic knowledge with context
        # 利用上下文发现新的语义知识
        discovered_nodes = await self.discovery_engine.discover_semantic_knowledge(episode, original_content)

        # Process each discovered node for evolution and linking
        # 处理每个发现的节点以进行演变和关联
        processed_nodes = []
        for new_node in discovered_nodes:
            processed_node = await self._process_semantic_node(new_node, episode)
            processed_nodes.append(processed_node)

        return processed_nodes

    async def _process_semantic_node(self, new_node: SemanticNode, episode: Episode) -> SemanticNode:
        """
        Process a semantic node for evolution, linking, and storage.
        处理语义节点以进行演变、关联和存储。
        """
        # Check if knowledge already exists
        # 检查知识是否已存在
        existing_node = await self.storage.find_semantic_node_by_key(owner_id=new_node.owner_id, key=new_node.key)

        if existing_node:
            # Handle evolution of existing knowledge
            # 处理现有知识的演变
            return await self._evolve_semantic_knowledge(existing_node, new_node, episode)
        else:
            # Store new knowledge with bidirectional linking and embedding generation
            # 存储新知识并建立双向链接，同时生成嵌入向量
            content_for_embedding = f"{new_node.key} {new_node.value} {new_node.context}"
            await self.storage.store_semantic_node_with_embedding(new_node, content_for_embedding)
            await self._establish_bidirectional_links(new_node, episode)
            return new_node

    async def _evolve_semantic_knowledge(
        self, existing: SemanticNode, new: SemanticNode, episode: Episode
    ) -> SemanticNode:
        """
        Evolve existing semantic knowledge with comprehensive evolution tracking.
        用综合演变跟踪演变现有语义知识。
        """
        from dataclasses import replace  # Import here to avoid unused import warnings
        
        if existing.value != new.value:
            # Knowledge evolution detected
            # 检测到知识演变
            evolved_node = replace(
                existing,
                value=new.value,
                context=new.context,  # Update context
                version=existing.version + 1,
                evolution_history=existing.evolution_history + [existing.value],
                evolution_episode_ids=existing.evolution_episode_ids + [episode.episode_id],
                last_updated=datetime.now(),
                confidence=(existing.confidence + new.confidence) / 2,
            )

            # Update storage with new embedding for evolved content
            # 更新存储，为演化的内容生成新的嵌入向量
            # First generate embedding for the evolved node
            evolved_content = f"{evolved_node.key} {evolved_node.value} {evolved_node.context}"
            embedding = await self._generate_embedding_if_available(evolved_content)
            if embedding:
                from dataclasses import replace
                evolved_node = replace(evolved_node, embedding_vector=embedding)
            
            await self.storage.update_semantic_node(evolved_node)

            # Link this episode as an evolution trigger
            # 将此情景链接为演变触发器
            await self._establish_bidirectional_links(evolved_node, episode)

            return evolved_node
        else:
            # Reinforce existing knowledge without evolution
            # 强化现有知识但不演变
            from dataclasses import replace  # Import here for this usage
            reinforced_node = replace(
                existing,
                linked_episode_ids=list(set(existing.linked_episode_ids + [episode.episode_id])),
                confidence=min(1.0, existing.confidence + 0.1),
                last_accessed=datetime.now(),
                access_count=existing.access_count + 1,
            )

            await self.storage.update_semantic_node(reinforced_node)
            await self._establish_bidirectional_links(reinforced_node, episode)

            return reinforced_node

    async def _establish_bidirectional_links(self, semantic_node: SemanticNode, episode: Episode) -> None:
        """
        Establish bidirectional links between semantic node and episode.
        在语义节点和情景之间建立双向链接。
        """
        # Update episode to reference this semantic node
        # 更新情景以引用此语义节点
        if "semantic_node_ids" not in episode.metadata.custom_fields:
            episode.metadata.custom_fields["semantic_node_ids"] = []

        if semantic_node.node_id not in episode.metadata.custom_fields["semantic_node_ids"]:
            episode.metadata.custom_fields["semantic_node_ids"].append(semantic_node.node_id)

        # Note: Episode storage update would happen in the calling context
        # 注意：情景存储更新会在调用上下文中发生

    async def _generate_embedding_if_available(self, content: str) -> list[float] | None:
        """Generate embedding if storage has embedding capabilities."""
        if hasattr(self.storage, '_generate_query_embedding'):
            try:
                return await self.storage._generate_query_embedding(content)
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")
        return None

    async def get_semantic_evolution_history(self, semantic_node_id: str) -> dict[str, Any]:
        """
        Get comprehensive evolution history including all related episodes.
        获取包括所有相关情景的综合演变历史。

        Returns:
            {
                "node": current_semantic_node,
                "evolution_timeline": [
                    {
                        "version": 1,
                        "value": "historical_value",
                        "episode": episode_that_caused_change,
                        "timestamp": change_timestamp
                    }
                ],
                "linked_episodes": [episodes that reference this knowledge],
                "evolution_episodes": [episodes that caused evolution]
            }
        """
        semantic_node = await self.storage.get_by_id(semantic_node_id)
        if not semantic_node:
            return {}

        # Build evolution timeline
        # 构建演变时间线
        evolution_timeline = []

        # Add historical versions
        # 添加历史版本
        for i, historical_value in enumerate(semantic_node.evolution_history):
            episode_id = (
                semantic_node.evolution_episode_ids[i] if i < len(semantic_node.evolution_episode_ids) else None
            )
            episode = None
            if episode_id:
                episodes = await self.retrieval_service.episodic_storage.get_by_ids([episode_id])
                episode = episodes[0] if episodes else None

            evolution_timeline.append(
                {
                    "version": i + 1,
                    "value": historical_value,
                    "episode": episode,
                    "timestamp": episode.temporal_info.timestamp if episode else semantic_node.created_at,
                }
            )

        # Add current version
        # 添加当前版本
        evolution_timeline.append(
            {
                "version": semantic_node.version,
                "value": semantic_node.value,
                "episode": None,  # Current version
                "timestamp": semantic_node.last_updated,
            }
        )

        # Get associated episodes
        # 获取关联情景
        associated_episodes = await self.retrieval_service.get_semantic_episodes(semantic_node_id)

        return {
            "node": semantic_node,
            "evolution_timeline": evolution_timeline,
            "linked_episodes": associated_episodes["linked_episodes"],
            "evolution_episodes": associated_episodes["evolution_episodes"],
        }
