"""
Simplified Semantic Memory Demo

This script demonstrates the core functionality of the Nemori semantic memory system
with a simplified, working example.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from nemori.core.data_types import SemanticNode, SemanticRelationship, RelationshipType
from nemori.storage.storage_types import StorageConfig, SemanticNodeQuery, SortOrder
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository


async def main():
    print("🧠 Nemori Semantic Memory System - Simplified Demo")
    print("=" * 60)

    # Initialize semantic storage
    print("\n🔧 Initializing semantic memory storage...")

    storage_config = StorageConfig(connection_string="demo_semantic.duckdb", backend_type="duckdb")

    semantic_storage = DuckDBSemanticMemoryRepository(storage_config)
    await semantic_storage.initialize()
    print("✅ Semantic memory storage initialized")

    try:
        # Demo 1: Create and store semantic nodes
        print("\n📝 Demo 1: Creating and storing semantic knowledge")
        print("-" * 50)

        # Create semantic nodes representing John's research evolution
        node1 = SemanticNode(
            owner_id="john",
            key="研究方向",
            value="大语言模型提示工程",
            context="John表示最近在研究大语言模型的提示工程，特别关注如何让LLM更好地理解复杂指令",
            confidence=0.9,
            discovery_episode_id="episode_1",
            linked_episode_ids=["episode_1"],
        )

        await semantic_storage.store_semantic_node(node1)
        print(f"💾 Stored: {node1.key} -> {node1.value}")

        # Create another related node
        node2 = SemanticNode(
            owner_id="john",
            key="技术偏好",
            value="Python和PyTorch",
            context="John在讨论实现时提到更喜欢使用Python和PyTorch框架",
            confidence=0.85,
            discovery_episode_id="episode_1",
            linked_episode_ids=["episode_1"],
        )

        await semantic_storage.store_semantic_node(node2)
        print(f"💾 Stored: {node2.key} -> {node2.value}")

        # Demo 2: Create relationships
        print("\n🔗 Demo 2: Creating relationships between semantic nodes")
        print("-" * 50)

        relationship = SemanticRelationship(
            source_node_id=node1.node_id,
            target_node_id=node2.node_id,
            relationship_type=RelationshipType.RELATED,
            strength=0.8,
            description="技术栈支持研究方向",
            discovery_episode_id="episode_1",
        )

        await semantic_storage.store_semantic_relationship(relationship)
        print(f"🔗 Created relationship: {node1.key} <-> {node2.key}")

        # Demo 3: Search semantic knowledge
        print("\n🔍 Demo 3: Searching semantic knowledge")
        print("-" * 50)

        search_query = SemanticNodeQuery(
            owner_id="john", text_search="研究", limit=10, sort_by="confidence", sort_order=SortOrder.DESC
        )

        search_results = await semantic_storage.search_semantic_nodes(search_query)
        print(f"📊 Found {len(search_results.semantic_nodes)} nodes matching '研究':")

        for node in search_results.semantic_nodes:
            print(f"  • {node.key}: {node.value} (confidence: {node.confidence:.2f})")

        # Demo 4: Knowledge evolution
        print("\n🔄 Demo 4: Knowledge evolution")
        print("-" * 50)

        # Evolve John's research focus
        evolved_node = node1.evolve(
            new_value="AI Agent行为规划",
            new_context="John转向AI Agent的行为规划研究，认为Agent的决策机制更有挑战性",
            evolution_episode_id="episode_2",
        )

        await semantic_storage.update_semantic_node(evolved_node)
        print(f"🔄 Evolved knowledge: {evolved_node.key}")
        print(f"  Current value: {evolved_node.value} (Version {evolved_node.version})")
        print(f"  Evolution history: {evolved_node.evolution_history}")

        # Demo 5: Find related nodes
        print("\n🌐 Demo 5: Finding related semantic nodes")
        print("-" * 50)

        related_nodes = await semantic_storage.find_relationships_for_node(node1.node_id)
        print(f"📊 Found {len(related_nodes)} relationships for '{node1.key}':")

        for related_node, relationship in related_nodes:
            print(f"  🔗 {related_node.key}: {related_node.value}")
            print(f"     Relationship: {relationship.relationship_type.value} (strength: {relationship.strength:.2f})")

        # Demo 6: Get all nodes for owner
        print("\n📋 Demo 6: All semantic knowledge for owner")
        print("-" * 50)

        all_nodes = await semantic_storage.get_all_semantic_nodes_for_owner("john")
        print(f"📊 Total semantic nodes for 'john': {len(all_nodes)}")

        for node in all_nodes:
            print(f"  📝 {node.key}: {node.value}")
            if node.version > 1:
                print(f"     └── Evolution: v{node.version}, History: {node.evolution_history}")

        # Demo 7: Statistics
        print("\n📊 Demo 7: Semantic memory statistics")
        print("-" * 50)

        stats = await semantic_storage.get_semantic_statistics("john")
        print(f"📈 Statistics for 'john':")
        print(f"  • Total nodes: {stats['node_count']}")
        print(f"  • Total relationships: {stats['relationship_count']}")
        print(f"  • Average confidence: {stats['average_confidence']:.2f}")
        print(f"  • Version distribution: {stats['version_distribution']}")

        # Demo 8: Similarity search
        print("\n🎯 Demo 8: Similarity-based search")
        print("-" * 50)

        similar_nodes = await semantic_storage.similarity_search_semantic_nodes("john", "技术", limit=5)
        print(f"🔍 Similarity search for '技术' returned {len(similar_nodes)} results:")

        for node in similar_nodes:
            print(f"  🎯 {node.key}: {node.value}")

        print("\n🎉 Demo completed successfully!")
        print("\n" + "=" * 60)
        print("✨ Key Features Demonstrated:")
        print("  ✓ Semantic knowledge storage and retrieval")
        print("  ✓ Knowledge evolution with version tracking")
        print("  ✓ Bidirectional episode associations")
        print("  ✓ Relationship mapping between knowledge")
        print("  ✓ Full-text and similarity search")
        print("  ✓ Statistics and analytics")
        print("  ✓ Owner-based knowledge isolation")

    finally:
        # Cleanup
        await semantic_storage.close()

        # Remove demo database
        db_path = Path("demo_semantic.duckdb")
        if db_path.exists():
            db_path.unlink()
            print("\n🧹 Demo database cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
