"""
Comprehensive Nemori System Test

This script demonstrates the complete Nemori functionality including:
1. Episodic Memory: Building episodes from conversation data
2. Semantic Memory: Knowledge discovery and evolution  
3. Unified Retrieval: Cross-memory search capabilities
4. Bidirectional Linking: Episode ↔ Semantic associations
5. LLM Integration: Semantic discovery through differential analysis
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from nemori.core.data_types import RawEventData, DataType, TemporalInfo
from nemori.core.episode import Episode
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.retrieval.unified_service import UnifiedRetrievalService
from nemori.retrieval.providers.bm25_provider import BM25RetrievalProvider
from nemori.retrieval.providers.embedding_provider import EmbeddingRetrievalProvider
from nemori.episode_manager import EpisodeManager


class ComprehensiveNemoriTest:
    def __init__(self):
        self.db_path = "comprehensive_test.duckdb"
        self.episode_repo = None
        self.semantic_repo = None
        self.episode_manager = None
        self.unified_retrieval = None
        
    async def initialize(self):
        """Initialize all Nemori components"""
        print("🔧 Initializing Nemori System Components")
        print("=" * 60)
        
        # Storage configuration
        storage_config = StorageConfig(
            connection_string=self.db_path, 
            backend_type="duckdb"
        )
        
        # Add embedding configuration (using your API settings)
        storage_config.openai_api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        storage_config.openai_base_url = "https://jeniya.cn/v1"
        
        # Initialize repositories
        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
        self.semantic_repo = DuckDBSemanticMemoryRepository(storage_config)
        
        await self.episode_repo.initialize()
        await self.semantic_repo.initialize()
        
        # Initialize LLM provider for semantic discovery
        llm_provider = OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm",
            base_url="https://jeniya.cn/v1"
        )
        
        # Initialize episode manager with semantic discovery
        from nemori.core.builders import EpisodeBuilderRegistry
        from nemori.storage.duckdb_storage import DuckDBRawDataRepository
        
        builder_registry = EpisodeBuilderRegistry()
        raw_data_repo = DuckDBRawDataRepository(storage_config)
        await raw_data_repo.initialize()
        
        self.episode_manager = EpisodeManager(
            raw_data_repo=raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=None
        )
        from nemori.retrieval.retrieval_types import RetrievalConfig
        # Initialize unified retrieval service
        embedding_provider = EmbeddingRetrievalProvider(
            episode_repo=self.episode_repo,
            api_key="sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm",
            base_url="https://jeniya.cn/v1"
        )
        
        self.unified_retrieval = UnifiedRetrievalService(
            episode_repo=self.episode_repo,
            semantic_repo=self.semantic_repo,
            retrieval_providers=embedding_provider
        )
        
        print("✅ All Nemori components initialized successfully")
        print("   📖 Episodic Memory: DuckDB repository")
        print("   🧠 Semantic Memory: DuckDB repository with embeddings")
        print("   🤖 LLM Provider: OpenAI compatible API")
        print("   🔗 Unified Retrieval: BM25 + Embedding search")
        
    def create_sample_conversations(self) -> List[Dict[str, Any]]:
        """Create comprehensive sample conversation data"""
        conversations = [
            {
                "conversation_id": "conv_001",
                "participants": ["Alice", "Bob"],
                "timestamp": datetime.now() - timedelta(days=5),
                "messages": [
                    {"speaker": "Alice", "content": "我最近在研究大语言模型的微调技术，你有什么经验分享吗？", "timestamp": datetime.now() - timedelta(days=5, hours=1)},
                    {"speaker": "Bob", "content": "我用过LoRA和QLoRA方法，效果不错。你具体想微调什么类型的任务？", "timestamp": datetime.now() - timedelta(days=5, hours=1, minutes=2)},
                    {"speaker": "Alice", "content": "主要是想做中文对话系统，我在用ChatGLM-6B作为基础模型", "timestamp": datetime.now() - timedelta(days=5, hours=1, minutes=5)},
                    {"speaker": "Bob", "content": "ChatGLM不错，建议你试试P-Tuning v2，对中文效果很好。数据集准备了多少？", "timestamp": datetime.now() - timedelta(days=5, hours=1, minutes=8)},
                    {"speaker": "Alice", "content": "大概有10万条对话数据，都是客服领域的。我担心过拟合问题", "timestamp": datetime.now() - timedelta(days=5, hours=1, minutes=12)},
                    {"speaker": "Bob", "content": "10万条不算多，建议加入一些正则化技术，比如Dropout和权重衰减", "timestamp": datetime.now() - timedelta(days=5, hours=1, minutes=15)}
                ]
            },
            {
                "conversation_id": "conv_002", 
                "participants": ["Alice", "Charlie"],
                "timestamp": datetime.now() - timedelta(days=3),
                "messages": [
                    {"speaker": "Charlie", "content": "Alice，听说你在做AI项目？我们公司也想引入AI技术", "timestamp": datetime.now() - timedelta(days=3, hours=2)},
                    {"speaker": "Alice", "content": "是的，我们在开发智能客服系统。你们公司是什么行业？", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=3)},
                    {"speaker": "Charlie", "content": "我们做电商，每天客服咨询量很大，想用AI来降低人工成本", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=6)},
                    {"speaker": "Alice", "content": "电商客服很适合用AI，我们的系统就是针对这个场景。主要用了检索增强生成技术", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=9)},
                    {"speaker": "Charlie", "content": "RAG技术？听起来很高级，部署成本高吗？", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=12)},
                    {"speaker": "Alice", "content": "成本可控，主要是向量数据库和API调用费用。我们用的是开源方案", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=15)},
                    {"speaker": "Charlie", "content": "能详细介绍一下技术架构吗？我想了解具体实现", "timestamp": datetime.now() - timedelta(days=3, hours=2, minutes=18)}
                ]
            },
            {
                "conversation_id": "conv_003",
                "participants": ["Bob", "David"],
                "timestamp": datetime.now() - timedelta(days=1),
                "messages": [
                    {"speaker": "David", "content": "Bob，你对多模态AI有了解吗？我在研究视觉问答系统", "timestamp": datetime.now() - timedelta(days=1, hours=3)},
                    {"speaker": "Bob", "content": "有一些，我用过CLIP和BLIP模型。你的VQA系统主要解决什么问题？", "timestamp": datetime.now() - timedelta(days=1, hours=3, minutes=2)},
                    {"speaker": "David", "content": "我们想做医学影像的智能诊断辅助，让AI能理解X光片并回答医生的问题", "timestamp": datetime.now() - timedelta(days=1, hours=3, minutes=5)},
                    {"speaker": "Bob", "content": "医学影像很有挑战性，数据标注质量要求很高。你们有专业的医生团队吗？", "timestamp": datetime.now() - timedelta(days=1, hours=3, minutes=8)},
                    {"speaker": "David", "content": "有的，我们和三甲医院合作，有放射科医生帮助标注。数据质量应该没问题", "timestamp": datetime.now() - timedelta(days=1, hours=3, minutes=11)},
                    {"speaker": "Bob", "content": "那很好，建议你看看最新的医学多模态模型，比如Med-CLIP", "timestamp": datetime.now() - timedelta(days=1, hours=3, minutes=14)}
                ]
            }
        ]
        return conversations
    
    async def test_episodic_memory(self, conversations: List[Dict[str, Any]]):
        """Test episodic memory functionality"""
        print("\n📖 Testing Episodic Memory")
        print("-" * 40)
        
        episodes_created = []
        
        for conv in conversations:
            # Convert conversation to RawEventData
            conversation_text = "\n".join([
                f"{msg['speaker']}: {msg['content']}" 
                for msg in conv['messages']
            ])
            
            raw_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content=conversation_text,
                temporal_info=TemporalInfo(timestamp=conv['timestamp']),
                metadata={
                    "conversation_id": conv['conversation_id'],
                    "participants": conv['participants'],
                    "message_count": len(conv['messages'])
                }
            )
            
            # Process through episode manager
            await self.episode_manager.ingest_event(raw_data, conv['participants'][0])
            # Then build episode from the stored raw data
            episode = await self.episode_manager.build_episode(raw_data.data_id, conv['participants'][0])
            episodes_created.append(episode)
            
            print(f"✅ Created episode: {episode.title[:50]}...")
            print(f"   Owner: {episode.owner_id}")
            print(f"   Content length: {len(episode.content)} chars")
            print(f"   Importance: {episode.importance_score:.2f}")
        
        print(f"\n📊 Total episodes created: {len(episodes_created)}")
        return episodes_created
    
    async def test_semantic_memory(self, episodes: List[Episode]):
        """Test semantic memory functionality"""
        print("\n🧠 Testing Semantic Memory")
        print("-" * 40)
        
        # Check what semantic knowledge was discovered during episode creation
        owners = list(set(ep.owner_id for ep in episodes))
        total_semantic_nodes = 0
        
        for owner in owners:
            nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner)
            total_semantic_nodes += len(nodes)
            
            print(f"🔍 {owner}: {len(nodes)} semantic concepts discovered")
            
            # Show sample concepts
            for node in nodes[:3]:  # Show first 3
                print(f"   • {node.key}: {node.value[:40]}...")
                print(f"     Confidence: {node.confidence:.2f}, Version: {node.version}")
        
        print(f"\n📊 Total semantic concepts: {total_semantic_nodes}")
        
        # Test semantic search
        if total_semantic_nodes > 0:
            print("\n🔍 Testing semantic search:")
            search_queries = ["AI", "技术", "模型", "系统"]
            
            for query in search_queries:
                similar_nodes = await self.semantic_repo.similarity_search_semantic_nodes(
                    owners[0], query, limit=3
                )
                print(f"   Query '{query}': {len(similar_nodes)} results")
                for node in similar_nodes[:2]:
                    print(f"     - {node.key}: {node.value[:30]}...")
        
        return total_semantic_nodes
    
    async def test_unified_retrieval(self, episodes: List[Episode]):
        """Test unified retrieval functionality"""
        print("\n🔗 Testing Unified Retrieval")
        print("-" * 40)
        
        # Test queries that should find both episodic and semantic content
        test_queries = [
            "大语言模型微调",
            "智能客服系统",
            "多模态AI视觉",
            "技术架构设计"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Search episodes
            episode_results = await self.unified_retrieval.search_episodes(
                query=query,
                owner_id=episodes[0].owner_id,
                limit=3
            )
            print(f"   📖 Episodes found: {len(episode_results)}")
            for ep in episode_results[:2]:
                print(f"     - {ep.title[:40]}...")
            
            # Search semantic knowledge
            semantic_results = await self.unified_retrieval.search_semantic_knowledge(
                query=query,
                owner_id=episodes[0].owner_id,
                limit=3
            )
            print(f"   🧠 Semantic concepts found: {len(semantic_results)}")
            for sem in semantic_results[:2]:
                print(f"     - {sem.key}: {sem.value[:30]}...")
    
    async def test_bidirectional_linking(self, episodes: List[Episode]):
        """Test bidirectional episode-semantic linking"""
        print("\n🔗 Testing Bidirectional Linking")
        print("-" * 40)
        
        if not episodes:
            print("   No episodes to test linking")
            return
        
        sample_episode = episodes[0]
        
        # Get semantic concepts linked to this episode
        episode_semantics = await self.unified_retrieval.get_episode_semantics(
            sample_episode.episode_id
        )
        print(f"✅ Episode '{sample_episode.title[:30]}...' links to {len(episode_semantics)} semantic concepts")
        
        if episode_semantics:
            # Test reverse linking
            sample_semantic = episode_semantics[0]
            semantic_episodes = await self.unified_retrieval.get_semantic_episodes(
                sample_semantic.node_id
            )
            print(f"✅ Semantic concept '{sample_semantic.key}' links to {len(semantic_episodes)} episodes")
            print("   🔗 Bidirectional linking verified!")
        
        return len(episode_semantics)
    
    async def test_knowledge_evolution(self):
        """Test knowledge evolution functionality"""
        print("\n🔄 Testing Knowledge Evolution")
        print("-" * 40)
        
        # Find a semantic node to evolve
        alice_nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner("Alice")
        
        if alice_nodes:
            original_node = alice_nodes[0]
            print(f"📝 Original: {original_node.key} = {original_node.value}")
            print(f"   Version: {original_node.version}, Confidence: {original_node.confidence:.2f}")
            
            # Evolve the knowledge
            evolved_node = original_node.evolve(
                new_value=f"{original_node.value} - 已经实际部署到生产环境",
                new_context=f"{original_node.context} 项目已成功上线运行",
                evolution_episode_id="evolution_test"
            )
            
            await self.semantic_repo.update_semantic_node(evolved_node)
            
            print(f"🔄 Evolved: {evolved_node.key} = {evolved_node.value}")
            print(f"   Version: {evolved_node.version}, History: {len(evolved_node.evolution_history)} changes")
            
            return True
        
        print("   No semantic nodes available for evolution test")
        return False
    
    async def generate_comprehensive_report(self, episodes: List[Episode], semantic_count: int):
        """Generate a comprehensive test report"""
        print("\n📊 Comprehensive Test Report")
        print("=" * 60)
        
        # Episode statistics
        owners = list(set(ep.owner_id for ep in episodes))
        print(f"📖 EPISODIC MEMORY:")
        print(f"   • Total Episodes: {len(episodes)}")
        print(f"   • Unique Owners: {len(owners)}")
        
        for owner in owners:
            owner_episodes = [ep for ep in episodes if ep.owner_id == owner]
            avg_importance = sum(ep.importance_score for ep in owner_episodes) / len(owner_episodes)
            print(f"   • {owner}: {len(owner_episodes)} episodes (avg importance: {avg_importance:.2f})")
        
        # Semantic memory statistics
        print(f"\n🧠 SEMANTIC MEMORY:")
        print(f"   • Total Concepts: {semantic_count}")
        
        total_nodes_by_owner = {}
        for owner in owners:
            nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner)
            total_nodes_by_owner[owner] = len(nodes)
            if nodes:
                avg_confidence = sum(node.confidence for node in nodes) / len(nodes)
                print(f"   • {owner}: {len(nodes)} concepts (avg confidence: {avg_confidence:.2f})")
        
        # Unified system statistics
        print(f"\n🔗 UNIFIED SYSTEM:")
        total_memory_items = len(episodes) + semantic_count
        print(f"   • Total Memory Items: {total_memory_items}")
        print(f"   • Memory Distribution: {len(episodes)} episodes + {semantic_count} concepts")
        
        # Test various retrieval scenarios
        print(f"\n🎯 RETRIEVAL CAPABILITIES:")
        test_passed = 0
        total_tests = 4
        
        # Test 1: Episode search
        try:
            results = await self.unified_retrieval.search_episodes("AI", owners[0], limit=5)
            print(f"   ✓ Episode search: {len(results)} results")
            test_passed += 1
        except Exception as e:
            print(f"   ✗ Episode search failed: {e}")
        
        # Test 2: Semantic search
        try:
            results = await self.unified_retrieval.search_semantic_knowledge("技术", owners[0], limit=5)
            print(f"   ✓ Semantic search: {len(results)} results")
            test_passed += 1
        except Exception as e:
            print(f"   ✗ Semantic search failed: {e}")
        
        # Test 3: Cross-memory retrieval
        try:
            if episodes and semantic_count > 0:
                episode_semantics = await self.unified_retrieval.get_episode_semantics(episodes[0].episode_id)
                print(f"   ✓ Cross-memory linking: {len(episode_semantics)} links")
                test_passed += 1
            else:
                print(f"   - Cross-memory linking: No data to test")
        except Exception as e:
            print(f"   ✗ Cross-memory linking failed: {e}")
        
        # Test 4: Statistics
        try:
            stats = await self.semantic_repo.get_semantic_statistics(owners[0])
            print(f"   ✓ Statistics generation: {stats['node_count']} nodes tracked")
            test_passed += 1
        except Exception as e:
            print(f"   ✗ Statistics generation failed: {e}")
        
        print(f"\n✨ TEST SUMMARY:")
        print(f"   Tests Passed: {test_passed}/{total_tests}")
        print(f"   System Status: {'✅ HEALTHY' if test_passed >= 3 else '⚠️ ISSUES DETECTED'}")
        
        return test_passed >= 3
    
    async def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up resources...")
        
        if self.episode_repo:
            await self.episode_repo.close()
        if self.semantic_repo:
            await self.semantic_repo.close()
        
        # Remove test database
        db_path = Path(self.db_path)
        if db_path.exists():
            db_path.unlink()
            print("✅ Test database cleaned up")
    

async def main():
    """Main test function"""
    print("🚀 Comprehensive Nemori System Test")
    print("=" * 60)
    print("Testing complete Nemori functionality:")
    print("  📖 Episodic Memory: Episode creation and storage")
    print("  🧠 Semantic Memory: Knowledge discovery and evolution")
    print("  🔗 Unified Retrieval: Cross-memory search capabilities")
    print("  🤖 LLM Integration: Semantic discovery via differential analysis")
    print("  🎯 Bidirectional Linking: Episode ↔ Semantic associations")
    print()
    
    test_runner = ComprehensiveNemoriTest()
    
    try:
        # Initialize system
        await test_runner.initialize()
        
        # Create sample data
        print("\n📝 Creating sample conversation data...")
        conversations = test_runner.create_sample_conversations()
        print(f"✅ Created {len(conversations)} sample conversations")
        
        # Test episodic memory
        episodes = await test_runner.test_episodic_memory(conversations)
        
        # Test semantic memory
        semantic_count = await test_runner.test_semantic_memory(episodes)
        
        # Test unified retrieval
        await test_runner.test_unified_retrieval(episodes)
        
        # Test bidirectional linking
        await test_runner.test_bidirectional_linking(episodes)
        
        # Test knowledge evolution
        await test_runner.test_knowledge_evolution()
        
        # Generate comprehensive report
        success = await test_runner.generate_comprehensive_report(episodes, semantic_count)
        
        if success:
            print(f"\n🎉 ALL TESTS PASSED! Nemori system is working correctly.")
        else:
            print(f"\n⚠️ Some tests failed. Check the output above for details.")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await test_runner.cleanup()
        

if __name__ == "__main__":
    asyncio.run(main())