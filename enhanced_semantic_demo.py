"""
Enhanced Semantic Memory Demo with Real Embedding Search

This script demonstrates the complete semantic memory system with real
embedding-based similarity search using OpenAI embeddings.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from nemori.core.data_types import SemanticNode, SemanticRelationship, RelationshipType
from nemori.storage.storage_types import StorageConfig, SemanticNodeQuery, SortOrder
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository


async def main():
    print("🧠 Nemori Enhanced Semantic Memory System - Embedding Demo")
    print("=" * 65)
    api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
    base_url = "https://jeniya.cn/v1"
    # Check for OpenAI API key
    openai_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
    openai_base_url = "https://jeniya.cn/v1"
    #     openai_key = "EMPTY"
    # openai_base_url = "http://localhost:6007/v1"
    # model = "qwen3-emb"
    if not openai_key and not openai_base_url:
        print("⚠️  Warning: No OPENAI_API_KEY or OPENAI_BASE_URL found in environment.")
        print("   Embedding-based similarity search will fallback to text search.")
        print("   Set OPENAI_API_KEY or OPENAI_BASE_URL to test real embeddings.\n")

    # Initialize semantic storage with embedding configuration
    print("\n🔧 Initializing semantic memory storage with embedding support...")
    
    storage_config = StorageConfig(
        connection_string="enhanced_semantic_demo.duckdb", 
        backend_type="duckdb"
    )
    # Add embedding configuration
    storage_config.openai_api_key = openai_key
    storage_config.openai_base_url = openai_base_url

    semantic_storage = DuckDBSemanticMemoryRepository(storage_config)
    await semantic_storage.initialize()
    print("✅ Enhanced semantic memory storage initialized")

    try:
        # Demo 1: Create semantic nodes with embeddings
        print("\n📝 Demo 1: Creating semantic knowledge with embeddings")
        print("-" * 55)

        # Create comprehensive semantic nodes
        nodes_data = [
            {
                "key": "研究领域",
                "value": "人工智能与机器学习",
                "context": "John专注于人工智能领域，特别是机器学习算法的研究和应用",
                "confidence": 0.95
            },
            {
                "key": "编程语言偏好", 
                "value": "Python和JavaScript",
                "context": "John在开发中主要使用Python进行AI研究，JavaScript用于前端开发",
                "confidence": 0.9
            },
            {
                "key": "技术栈",
                "value": "TensorFlow、PyTorch、React",
                "context": "John熟练使用TensorFlow和PyTorch进行深度学习，React用于构建用户界面",
                "confidence": 0.85
            },
            {
                "key": "工作经验",
                "value": "5年软件开发，3年AI研究",
                "context": "John有5年的软件开发经验，最近3年专注于AI和机器学习研究",
                "confidence": 0.9
            },
            {
                "key": "教育背景",
                "value": "计算机科学硕士学位",
                "context": "John拥有计算机科学硕士学位，专业方向为人工智能",
                "confidence": 0.95
            }
        ]

        nodes = []
        for i, node_data in enumerate(nodes_data):
            node = SemanticNode(
                owner_id="john",
                key=node_data["key"],
                value=node_data["value"],
                context=node_data["context"],
                confidence=node_data["confidence"],
                discovery_episode_id=f"episode_{i+1}",
                linked_episode_ids=[f"episode_{i+1}"],
            )
            
            # Store node with embedding generation
            embedding_content = f"{node.key}: {node.value}. {node.context}"
            await semantic_storage.store_semantic_node_with_embedding(node, embedding_content)
            nodes.append(node)
            print(f"💾 Stored with embedding: {node.key} -> {node.value}")

        # Demo 2: Create relationships
        print("\n🔗 Demo 2: Creating semantic relationships")
        print("-" * 45)

        # Create meaningful relationships between nodes
        relationships_data = [
            (0, 2, RelationshipType.RELATED, "技术栈支持研究领域"),
            (1, 2, RelationshipType.PART_OF, "编程语言是技术栈的一部分"),
            (3, 0, RelationshipType.RELATED, "工作经验与研究领域相关"),
            (4, 0, RelationshipType.RELATED, "教育背景支撑研究领域")
        ]

        for source_idx, target_idx, rel_type, description in relationships_data:
            relationship = SemanticRelationship(
                source_node_id=nodes[source_idx].node_id,
                target_node_id=nodes[target_idx].node_id,
                relationship_type=rel_type,
                strength=0.8,
                description=description,
                discovery_episode_id="relationship_discovery"
            )
            await semantic_storage.store_semantic_relationship(relationship)
            print(f"🔗 Created: {nodes[source_idx].key} <-> {nodes[target_idx].key}")

        # Demo 3: Text-based search
        print("\n🔍 Demo 3: Traditional text-based search")
        print("-" * 45)

        text_query = SemanticNodeQuery(
            owner_id="john", 
            text_search="编程", 
            limit=10, 
            sort_by="confidence", 
            sort_order=SortOrder.DESC
        )
        text_results = await semantic_storage.search_semantic_nodes(text_query)
        print(f"📊 Text search for '编程' found {len(text_results.semantic_nodes)} results:")
        for node in text_results.semantic_nodes:
            print(f"  • {node.key}: {node.value} (confidence: {node.confidence:.2f})")

        # Demo 4: Embedding-based similarity search
        print("\n🎯 Demo 4: Embedding-based similarity search")
        print("-" * 50)

        # Test various similarity queries
        similarity_queries = [
            "机器学习和深度学习技术",
            "软件开发工具和框架", 
            "学历和教育经历",
            "JavaScript前端开发",
            "人工智能研究经验"
        ]

        for query in similarity_queries:
            print(f"\n🔍 Similarity search for: '{query}'")
            similar_nodes = await semantic_storage.similarity_search_semantic_nodes("john", query, limit=3)
            
            if similar_nodes:
                print(f"   Found {len(similar_nodes)} similar nodes:")
                for node in similar_nodes:
                    print(f"   🎯 {node.key}: {node.value}")
            else:
                print("   No similar nodes found (likely no embeddings available)")

        # Demo 5: Knowledge evolution with embeddings
        print("\n🔄 Demo 5: Knowledge evolution with embedding updates")
        print("-" * 55)

        # Evolve research focus
        research_node = next(node for node in nodes if node.key == "研究领域")
        evolved_node = research_node.evolve(
            new_value="大语言模型和多模态AI",
            new_context="John的研究重点转向大语言模型（LLM）和多模态人工智能系统",
            evolution_episode_id="evolution_episode_1"
        )

        # Update with new embedding (don't store as new, just update)
        new_embedding_content = f"{evolved_node.key}: {evolved_node.value}. {evolved_node.context}"
        
        # Generate new embedding for the evolved content
        if semantic_storage.openai_client:
            try:
                new_embedding = await semantic_storage._generate_query_embedding(new_embedding_content)
                if new_embedding:
                    from dataclasses import replace
                    evolved_node = replace(evolved_node, embedding_vector=new_embedding)
            except Exception as e:
                print(f"Warning: Could not generate new embedding: {e}")
        
        await semantic_storage.update_semantic_node(evolved_node)
        
        print(f"🔄 Evolved: {evolved_node.key}")
        print(f"  New value: {evolved_node.value}")
        print(f"  Version: {evolved_node.version}")
        print(f"  History: {evolved_node.evolution_history}")

        # Demo 6: Compare similarity before and after evolution
        print("\n📊 Demo 6: Similarity comparison after evolution")
        print("-" * 50)

        evolution_queries = [
            "大语言模型GPT",
            "多模态人工智能",
            "传统机器学习算法"
        ]

        for query in evolution_queries:
            print(f"\n🔍 Testing similarity with: '{query}'")
            results = await semantic_storage.similarity_search_semantic_nodes("john", query, limit=2)
            for node in results:
                version_info = f"(v{node.version})" if node.version > 1 else ""
                print(f"   🎯 {node.key}: {node.value} {version_info}")

        # Demo 7: Comprehensive statistics
        print("\n📊 Demo 7: Enhanced semantic memory statistics")
        print("-" * 50)

        stats = await semantic_storage.get_semantic_statistics("john")
        print(f"📈 Comprehensive statistics for 'john':")
        print(f"  • Total nodes: {stats['node_count']}")
        print(f"  • Total relationships: {stats['relationship_count']}")
        print(f"  • Average confidence: {stats['average_confidence']:.2f}")
        print(f"  • Version distribution: {stats['version_distribution']}")

        # Check embedding availability
        all_nodes = await semantic_storage.get_all_semantic_nodes_for_owner("john")
        nodes_with_embeddings = sum(1 for node in all_nodes if node.embedding_vector)
        print(f"  • Nodes with embeddings: {nodes_with_embeddings}/{len(all_nodes)}")

        print("\n🎉 Enhanced demo completed successfully!")
        print("\n" + "=" * 65)
        print("✨ Enhanced Features Demonstrated:")
        print("  ✓ Real embedding generation with OpenAI API")
        print("  ✓ Vector-based semantic similarity search")
        print("  ✓ Embedding-aware knowledge storage")
        print("  ✓ Evolution with embedding updates")
        print("  ✓ Fallback to text search when embeddings unavailable")
        print("  ✓ Comprehensive relationship mapping")
        print("  ✓ Advanced similarity comparisons")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await semantic_storage.close()

        # Remove demo database
        db_path = Path("enhanced_semantic_demo.duckdb")
        if db_path.exists():
            db_path.unlink()
            print("\n🧹 Demo database cleaned up")


if __name__ == "__main__":
    asyncio.run(main())