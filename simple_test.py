
import asyncio
from datetime import datetime
import os
from uuid import uuid4
from typing import Any
from nemori.core.data_types import SemanticNode
from nemori.core.episode import Episode, EpisodeLevel, EpisodeType, TemporalInfo
from nemori.retrieval.service import UnifiedRetrievalService
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.repository import EpisodicMemoryRepository, SemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig, EpisodeQuery, EpisodeSearchResult
from nemori.storage.storage_types import (
    DuplicateKeyError,
    InvalidDataError,
    NotFoundError,
    SemanticStorageError,
)
from nemori.core.data_types import RelationshipType, SemanticNode, SemanticRelationship
class MockSemanticStorage(SemanticMemoryRepository):
    """Mock implementation of SemanticStorage for testing protocol adherence."""

    def __init__(self):
        from nemori.storage.storage_types import StorageConfig

        super().__init__(StorageConfig())
        self.nodes = {}  # node_id -> SemanticNode
        self.relationships = {}  # relationship_id -> SemanticRelationship
        self.owner_keys = {}  # (owner_id, key) -> node_id

    # Base repository methods
    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True

    async def get_stats(self):
        from nemori.storage.storage_types import StorageStats

        return StorageStats()

    async def backup(self, destination: str) -> bool:
        # Mock implementation for testing
        return True

    async def restore(self, source: str) -> bool:
        # Mock implementation for testing
        return True

    # Missing abstract methods that need to be implemented
    async def search_semantic_nodes(self, query):
        from nemori.storage.storage_types import SemanticSearchResult

        # Simple implementation for testing
        return SemanticSearchResult()

    async def search_semantic_relationships(self, query):
        from nemori.storage.storage_types import SemanticSearchResult

        # Simple implementation for testing
        return SemanticSearchResult()

    async def store_semantic_node(self, node: SemanticNode) -> None:
        key = (node.owner_id, node.key)
        if key in self.owner_keys:
            raise DuplicateKeyError(f"Node with key {node.key} already exists for owner {node.owner_id}")

        self.nodes[node.node_id] = node
        self.owner_keys[key] = node.node_id

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        return self.nodes.get(node_id)

    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        node_id = self.owner_keys.get((owner_id, key))
        return self.nodes.get(node_id) if node_id else None

    async def update_semantic_node(self, node: SemanticNode) -> None:
        if node.node_id not in self.nodes:
            raise NotFoundError(f"Node {node.node_id} not found")

        # Update owner_keys mapping if key changed
        old_node = self.nodes[node.node_id]
        if old_node.key != node.key:
            old_key = (old_node.owner_id, old_node.key)
            new_key = (node.owner_id, node.key)
            del self.owner_keys[old_key]
            self.owner_keys[new_key] = node.node_id

        self.nodes[node.node_id] = node

    async def delete_semantic_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        key = (node.owner_id, node.key)
        del self.nodes[node_id]
        del self.owner_keys[key]
        return True

    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if node.discovery_episode_id == episode_id]

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if episode_id in node.linked_episode_ids]

    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        # Simple mock implementation: return nodes containing query text
        matching_nodes = []
        for node in self.nodes.values():
            if node.owner_id == owner_id and (
                query.lower() in node.key.lower()
                or query.lower() in node.value.lower()
                or query.lower() in node.context.lower()
            ):
                matching_nodes.append(node)
        return matching_nodes[:limit]

    async def store_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        self.relationships[relationship.relationship_id] = relationship

    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        return self.relationships.get(relationship_id)

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        results = []
        for rel in self.relationships.values():
            if rel.source_node_id == node_id:
                target_node = self.nodes.get(rel.target_node_id)
                if target_node:
                    results.append((target_node, rel))
            elif rel.target_node_id == node_id:
                source_node = self.nodes.get(rel.source_node_id)
                if source_node:
                    results.append((source_node, rel))
        return results

    async def update_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        if relationship.relationship_id not in self.relationships:
            raise NotFoundError(f"Relationship {relationship.relationship_id} not found")
        self.relationships[relationship.relationship_id] = relationship

    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        if relationship_id not in self.relationships:
            return False
        del self.relationships[relationship_id]
        return True

    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]

    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        return [node for node in self.nodes.values() if node.owner_id == owner_id]

    async def get_semantic_statistics(self, owner_id: str) -> dict[str, Any]:
        owner_nodes = await self.get_all_semantic_nodes_for_owner(owner_id)
        owner_relationships = [
            rel
            for rel in self.relationships.values()
            if any(
                self.nodes.get(rel.source_node_id, {}).get("owner_id") == owner_id
                or self.nodes.get(rel.target_node_id, {}).get("owner_id") == owner_id
                for rel in [rel]
            )
        ]

        return {
            "total_nodes": len(owner_nodes),
            "total_relationships": len(owner_relationships),
            "average_confidence": sum(node.confidence for node in owner_nodes) / len(owner_nodes) if owner_nodes else 0,
            "total_access_count": sum(node.access_count for node in owner_nodes),
        }

    async def cleanup_orphaned_relationships(self) -> int:
        orphaned_count = 0
        to_delete = []

        for rel_id, rel in self.relationships.items():
            if rel.source_node_id not in self.nodes or rel.target_node_id not in self.nodes:
                to_delete.append(rel_id)
                orphaned_count += 1

        for rel_id in to_delete:
            del self.relationships[rel_id]

        return orphaned_count

# --------------------------------------------------------------------------------
# Nemori 使用教程主函数 (Main Tutorial Function)
# --------------------------------------------------------------------------------

async def main():
    """一个完整的 Nemori 功能使用教程，包含中文注释"""
    owner_id = "user_tutorial"
    db_file = "nemori_tutorial.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    # --- 1. 初始化服务 (Initialization) ---
    print("--- 步骤 1: 初始化服务 ---")
    # 情节记忆使用 DuckDB, 这是一个真实的文件数据库实现
    episode_repo = DuckDBEpisodicMemoryRepository(StorageConfig(connection_string=db_file))
    # 语义记忆使用我们上面定义的内存模拟实现
    semantic_repo = MockSemanticStorage()
    await episode_repo.initialize()
    await semantic_repo.initialize()

    # UnifiedRetrievalService 是与记忆系统交互的统一入口
    retrieval_service = UnifiedRetrievalService(
        episode_repository=episode_repo, semantic_repository=semantic_repo
    )
    print("服务初始化成功。")

    # --- 2. 记忆的产生与存储 (Storing Initial Memories) ---
    print("--- 步骤 2: 记忆的产生与存储 ---")
    # 模拟一个“情节”：John 在一次会议上介绍了他的研究方向
    initial_episode = Episode(
        episode_id="ep_001",
        owner_id=owner_id,
        title="John 的研究方向介绍",
        content="在会议上, John 提到他目前的研究方向是大语言模型(LLM)。",
        episode_type=EpisodeType.CONVERSATIONAL,
        temporal_info=TemporalInfo(timestamp=datetime.now()),
    )
    await episode_repo.store_episode(initial_episode)
    print(f"  [存储情节] '{initial_episode.title}'")

    # 从这个情节中，我们提炼出一个“语义知识”
    initial_node = SemanticNode(
        node_id="node_john_research",
        owner_id=owner_id,
        key="John的研究方向",
        value="大语言模型(LLM)",
        discovery_episode_id="ep_001",  # 通过这个ID将语义知识与源头情节链接起来
    )
    await semantic_repo.store_semantic_node(initial_node)
    print(f"  [存储语义] 从 ep_001 提炼出知识: '{initial_node.key}' -> '{initial_node.value}'")

    # --- 3. 记忆的演化 (Memory Evolution) ---
    print("--- 步骤 3: 记忆的演化 ---")
    # 模拟一个新的“情节”：几周后，John 的研究方向更新了
    update_episode = Episode(
        episode_id="ep_002",
        owner_id=owner_id,
        title="研究方向更新会议",
        content="在今天的讨论中, John 宣布他的研究方向已转向 AI Agent 行为规划。",
        episode_type=EpisodeType.CONVERSATIONAL,
        temporal_info=TemporalInfo(timestamp=datetime.now()),
    )
    await episode_repo.store_episode(update_episode)
    print(f"  [存储新情节] '{update_episode.title}'")
    
    # 找到旧的语义知识并使其“演化”
    node_to_evolve = await semantic_repo.find_semantic_node_by_key(owner_id, "John的研究方向")
    if node_to_evolve:
        evolved_node = node_to_evolve.evolve(
            new_value="AI Agent 行为规划",
            new_context="在研究方向更新会议上得知",
            evolution_episode_id="ep_002" # 记录是哪个情节导致了这次演化
        )
        await semantic_repo.update_semantic_node(evolved_node)
        print(f"  [演化语义] 知识 '{evolved_node.key}' 已更新 -> '{evolved_node.value}' (版本: {evolved_node.version})")


    # --- 4. 记忆的检索与溯源 (Retrieval and Tracing) ---
    print("--- 步骤 4: 记忆的检索与溯源 ---")
    # 场景 A: 独立进行语义搜索
    query = "John"
    print(f"  (A) 搜索与 '{query}' 相关的语义知识...")
    search_results = await retrieval_service.search_semantic_memories(owner_id, query)
    for node in search_results:
        print(f"    -> 找到: '{node.key}' -> '{node.value}' (置信度: {node.confidence}, 版本: {node.version})")

    # 场景 B: 从语义知识追溯相关情节
    print(f"(B) 追溯知识 '{evolved_node.key}' 的相关情节...")
    related_episodes_info = await retrieval_service.get_semantic_episodes(evolved_node.node_id)
    discovery_ep = related_episodes_info.get("discovery_episode", [])
    evolution_eps = related_episodes_info.get("evolution_episodes", [])
    if discovery_ep:
        print(f"    -> 知识发现于情节: '{discovery_ep[0].title}'")
    if evolution_eps:
        print(f"    -> 知识演化于情节: '{evolution_eps[0].title}'")

    # --- 5. 清理 (Cleanup) ---
    await episode_repo.close()
    if os.path.exists(db_file):
        os.remove(db_file)
    print("--- 教程结束, 资源已清理 ---")


if __name__ == "__main__":
    asyncio.run(main())
