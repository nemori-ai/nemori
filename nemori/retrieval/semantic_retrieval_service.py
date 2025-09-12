from ..storage.repository import SemanticMemoryRepository
from ..core.data_types import SemanticNode
from typing import Any


class SemanticRetrievalService:
    """
    Service for retrieving semantic knowledge.
    语义知识检索服务。
    """

    def __init__(self, storage: SemanticMemoryRepository):
        self.storage = storage

    async def query_semantic_knowledge(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """
        Query semantic knowledge with text search.
        使用文本搜索查询语义知识。
        """
        # Basic text search implementation
        # 基本文本搜索实现
        results = await self.storage.query_semantic_knowledge(owner_id, query, limit)

        # Sort by relevance and recency
        # 按相关性和时效性排序
        sorted_results = sorted(results, key=lambda x: (x.relevance_score, x.last_updated), reverse=True)

        return sorted_results[:limit]

    async def get_related_knowledge(self, node_id: str, max_depth: int = 2) -> dict[str, list[SemanticNode]]:
        """
        Get related knowledge nodes with specified depth.
        获取指定深度的相关知识节点。
        """
        visited = set()
        result = {"direct": [], "indirect": []}

        # Get direct relationships
        # 获取直接关系
        direct_related = await self.storage.find_related_nodes(node_id)
        result["direct"] = [node for node, _ in direct_related]
        visited.add(node_id)

        # Get indirect relationships if max_depth > 1
        # 如果 max_depth > 1 则获取间接关系
        if max_depth > 1:
            for node, _ in direct_related:
                if node.node_id not in visited:
                    visited.add(node.node_id)
                    indirect_related = await self.storage.find_related_nodes(node.node_id)
                    for indirect_node, _ in indirect_related:
                        if indirect_node.node_id not in visited:
                            result["indirect"].append(indirect_node)

        return result

    async def get_knowledge_evolution(self, owner_id: str, key: str) -> list[dict[str, Any]]:
        """
        Get evolution history of a specific knowledge key.
        获取特定知识键的演变历史。
        """
        node = await self.storage.find_semantic_node_by_key(owner_id, key)
        if not node:
            return []

        evolution = []

        # Add historical versions
        # 添加历史版本
        for i, historical_value in enumerate(node.evolution_history):
            evolution.append(
                {
                    "version": i + 1,
                    "value": historical_value,
                    "timestamp": node.created_at,  # Approximate - could be enhanced
                    "confidence": 1.0,  # Historical confidence unknown
                }
            )

        # Add current version
        # 添加当前版本
        evolution.append(
            {
                "version": node.version,
                "value": node.value,
                "timestamp": node.last_updated,
                "confidence": node.confidence,
            }
        )

        return evolution
