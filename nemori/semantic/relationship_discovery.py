"""
Relationship discovery engine for identifying connections between semantic nodes.

This module provides capabilities to discover and manage relationships
between pieces of semantic knowledge based on their context and co-occurrence.
"""

from ..core.data_types import SemanticNode, SemanticRelationship, RelationshipType
from ..core.episode import Episode


class RelationshipDiscoveryEngine:
    """
    Discovers and manages relationships between semantic nodes.
    发现和管理语义节点之间的关系。
    """

    def __init__(self, storage=None):
        self.storage = storage

    async def discover_relationships(
        self, nodes: list[SemanticNode], context_episode: Episode
    ) -> list[SemanticRelationship]:
        """
        Discover relationships between semantic nodes based on context.
        基于上下文发现语义节点之间的关系。
        """
        if len(nodes) < 2:
            return []

        relationships = []

        # Simple bidirectional association discovery
        # 简单的双向关联发现
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                relationship_strength = await self._calculate_relationship_strength(node_a, node_b, context_episode)

                if relationship_strength > 0.3:  # Threshold for meaningful relationship
                    relationship = SemanticRelationship(
                        source_node_id=node_a.node_id,
                        target_node_id=node_b.node_id,
                        relationship_type=RelationshipType.RELATED,
                        strength=relationship_strength,
                        discovery_episode_id=context_episode.episode_id,
                        description=f"Co-discovered in episode: {context_episode.title}",
                    )
                    relationships.append(relationship)

        if self.storage:
            for rel in relationships:
                await self.storage.store_semantic_relationship(rel)

        return relationships

    async def _calculate_relationship_strength(
        self, node_a: SemanticNode, node_b: SemanticNode, context: Episode
    ) -> float:
        """
        Calculate relationship strength between two semantic nodes.
        计算两个语义节点之间的关系强度。

        Simple implementation - can be enhanced with NLP similarity analysis.
        简单实现 - 可通过 NLP 相似性分析增强。
        """
        # Temporal proximity (co-discovered in same batch)
        # 时间邻近性
        time_factor = 1.0 if node_a.created_at == node_b.created_at else 0.7

        # Context similarity (same episode)
        # 上下文相似性（相同情景）
        context_factor = 1.0 if node_a.discovery_episode_id == node_b.discovery_episode_id else 0.5

        # Textual similarity of their original contexts (basic implementation)
        # 文本相似性（基本实现）
        text_similarity = self._calculate_text_similarity(node_a.context, node_b.context)

        return time_factor * 0.3 + context_factor * 0.4 + text_similarity * 0.3

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate a basic Jaccard similarity for two texts."""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0
