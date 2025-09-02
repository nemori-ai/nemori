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


class SemanticEvolutionManager:
    """
    Manages semantic memory evolution with bidirectional episode associations.
    管理语义记忆演变及双向情景关联。
    """
    
    def __init__(
        self, 
        storage, # SemanticMemoryRepository
        discovery_engine: ContextAwareSemanticDiscoveryEngine,
        retrieval_service=None
    ):
        self.storage = storage
        self.discovery_engine = discovery_engine
        self.retrieval_service = retrieval_service
        
    async def process_episode_for_semantics(
        self, 
        episode: Episode, 
        original_content: str
    ) -> list[SemanticNode]:
        """
        Process an episode to discover and update semantic knowledge with full context.
        处理情景以发现和更新语义知识，包含完整上下文。
        """
        # Discover new semantic knowledge with context
        # 利用上下文发现新的语义知识
        discovered_nodes = await self.discovery_engine.discover_semantic_knowledge(
            episode, original_content
        )
        
        # Process each discovered node for evolution and linking
        # 处理每个发现的节点以进行演变和关联
        processed_nodes = []
        for new_node in discovered_nodes:
            try:
                processed_node = await self._process_semantic_node(new_node, episode)
                processed_nodes.append(processed_node)
            except Exception as e:
                print(f"Error processing semantic node: {e}")
                # Continue with other nodes even if one fails
                
        return processed_nodes
        
    async def _process_semantic_node(
        self, 
        new_node: SemanticNode, 
        episode: Episode
    ) -> SemanticNode:
        """
        Process a semantic node for evolution, linking, and storage.
        处理语义节点以进行演变、关联和存储。
        """
        # Check if knowledge already exists
        # 检查知识是否已存在
        existing_node = await self.storage.find_semantic_node_by_key(
            owner_id=new_node.owner_id, 
            key=new_node.key
        )
        
        if existing_node:
            # Handle evolution of existing knowledge
            # 处理现有知识的演变
            evolved_node = await self._evolve_semantic_knowledge(existing_node, new_node, episode)
            await self._establish_bidirectional_links(evolved_node, episode)
            return evolved_node
        else:
            # Store new knowledge with bidirectional linking
            # 存储新知识并建立双向链接
            await self.storage.store_semantic_node(new_node)
            await self._establish_bidirectional_links(new_node, episode)
            return new_node
    
    async def _evolve_semantic_knowledge(
        self, 
        existing: SemanticNode, 
        new: SemanticNode, 
        episode: Episode
    ) -> SemanticNode:
        """
        Evolve existing semantic knowledge with comprehensive evolution tracking.
        用综合演变跟踪演变现有语义知识。
        """
        if existing.value != new.value:
            # Knowledge evolution detected
            # 检测到知识演变
            evolved_node = replace(existing,
                value=new.value,
                context=new.context,  # Update context
                version=existing.version + 1,
                evolution_history=existing.evolution_history + [existing.value],
                evolution_episode_ids=existing.evolution_episode_ids + [episode.episode_id],
                last_updated=datetime.now(),
                confidence=(existing.confidence + new.confidence) / 2,  # Average confidence
                # Merge linked episodes
                linked_episode_ids=list(set(existing.linked_episode_ids + [episode.episode_id]))
            )
            
            # Update storage
            # 更新存储
            await self.storage.update_semantic_node(evolved_node)
            
            return evolved_node
        else:
            # Reinforce existing knowledge without evolution
            # 强化现有知识但不演变
            reinforced_node = replace(existing,
                linked_episode_ids=list(set(existing.linked_episode_ids + [episode.episode_id])),
                confidence=min(1.0, existing.confidence + 0.1),  # Small confidence boost
                last_accessed=datetime.now(),
                access_count=existing.access_count + 1,
                # Update context if new context is more informative
                context=new.context if len(new.context) > len(existing.context) else existing.context
            )
            
            await self.storage.update_semantic_node(reinforced_node)
            return reinforced_node
    
    async def _establish_bidirectional_links(
        self, 
        semantic_node: SemanticNode, 
        episode: Episode
    ) -> None:
        """
        Establish bidirectional links between semantic node and episode.
        在语义节点和情景之间建立双向链接。
        """
        try:
            # Update episode to reference this semantic node
            # 更新情景以引用此语义节点
            if "semantic_node_ids" not in episode.metadata.custom_fields:
                episode.metadata.custom_fields["semantic_node_ids"] = []
            
            if semantic_node.node_id not in episode.metadata.custom_fields["semantic_node_ids"]:
                episode.metadata.custom_fields["semantic_node_ids"].append(semantic_node.node_id)
                
            # Note: Episode storage update would happen in the calling context
            # 注意：情景存储更新会在调用上下文中发生
            
        except Exception as e:
            print(f"Warning: Failed to establish bidirectional links: {e}")
    
    async def get_semantic_evolution_history(
        self, 
        semantic_node_id: str
    ) -> dict[str, Any]:
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
        semantic_node = await self.storage.get_semantic_node_by_id(semantic_node_id)
        if not semantic_node:
            return {}
        
        try:
            # Build evolution timeline
            # 构建演变时间线
            evolution_timeline = []
            
            # Add historical versions
            # 添加历史版本
            for i, historical_value in enumerate(semantic_node.evolution_history):
                episode_id = (semantic_node.evolution_episode_ids[i] 
                            if i < len(semantic_node.evolution_episode_ids) else None)
                episode = None
                if episode_id and self.retrieval_service:
                    episodes = await self.retrieval_service.get_episodes_by_ids([episode_id])
                    episode = episodes[0] if episodes else None
                
                evolution_timeline.append({
                    "version": i + 1,
                    "value": historical_value,
                    "episode": episode,
                    "timestamp": episode.temporal_info.timestamp if episode else semantic_node.created_at
                })
            
            # Add current version
            # 添加当前版本
            evolution_timeline.append({
                "version": semantic_node.version,
                "value": semantic_node.value,
                "episode": None,  # Current version
                "timestamp": semantic_node.last_updated
            })
            
            # Get associated episodes
            # 获取关联情景
            linked_episodes = []
            evolution_episodes = []
            
            if self.retrieval_service:
                if semantic_node.linked_episode_ids:
                    linked_episodes = await self.retrieval_service.get_episodes_by_ids(
                        semantic_node.linked_episode_ids
                    )
                
                if semantic_node.evolution_episode_ids:
                    evolution_episodes = await self.retrieval_service.get_episodes_by_ids(
                        semantic_node.evolution_episode_ids
                    )
            
            return {
                "node": semantic_node,
                "evolution_timeline": evolution_timeline,
                "linked_episodes": linked_episodes,
                "evolution_episodes": evolution_episodes
            }
            
        except Exception as e:
            print(f"Error getting evolution history: {e}")
            return {
                "node": semantic_node,
                "evolution_timeline": [],
                "linked_episodes": [],
                "evolution_episodes": []
            }

    async def update_semantic_node_importance(
        self, 
        node_id: str, 
        importance_delta: float = 0.1
    ) -> SemanticNode | None:
        """
        Update the importance score of a semantic node.
        更新语义节点的重要性分数。
        """
        try:
            node = await self.storage.get_semantic_node_by_id(node_id)
            if not node:
                return None
            
            updated_node = replace(
                node,
                importance_score=max(0.0, min(1.0, node.importance_score + importance_delta)),
                last_accessed=datetime.now(),
                access_count=node.access_count + 1
            )
            
            await self.storage.update_semantic_node(updated_node)
            return updated_node
            
        except Exception as e:
            print(f"Error updating node importance: {e}")
            return None

    async def get_knowledge_evolution_stats(self, owner_id: str) -> dict[str, Any]:
        """
        Get evolution statistics for an owner's semantic knowledge.
        获取所有者语义知识的演变统计信息。
        """
        try:
            all_nodes = await self.storage.get_all_semantic_nodes_for_owner(owner_id)
            
            total_nodes = len(all_nodes)
            evolved_nodes = [node for node in all_nodes if node.version > 1]
            total_evolutions = sum(node.version - 1 for node in all_nodes)
            
            avg_confidence = sum(node.confidence for node in all_nodes) / total_nodes if total_nodes > 0 else 0.0
            avg_importance = sum(node.importance_score for node in all_nodes) / total_nodes if total_nodes > 0 else 0.0
            
            return {
                "total_nodes": total_nodes,
                "evolved_nodes": len(evolved_nodes),
                "evolution_rate": len(evolved_nodes) / total_nodes if total_nodes > 0 else 0.0,
                "total_evolutions": total_evolutions,
                "avg_confidence": avg_confidence,
                "avg_importance": avg_importance,
                "most_evolved": max(all_nodes, key=lambda x: x.version) if all_nodes else None,
                "highest_importance": max(all_nodes, key=lambda x: x.importance_score) if all_nodes else None
            }
            
        except Exception as e:
            print(f"Error getting evolution stats: {e}")
            return {
                "total_nodes": 0,
                "evolved_nodes": 0,
                "evolution_rate": 0.0,
                "total_evolutions": 0,
                "avg_confidence": 0.0,
                "avg_importance": 0.0,
                "most_evolved": None,
                "highest_importance": None
            }