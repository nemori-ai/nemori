"""
Simple Nemori System Test

A simplified test script that demonstrates Nemori functionality using the actual APIs.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from nemori.core.data_types import RawEventData, DataType, TemporalInfo
from nemori.core.episode import Episode
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.episode_manager import EpisodeManager


async def test_basic_nemori():
    """Test basic Nemori functionality"""
    print("🚀 Simple Nemori Test")
    print("=" * 50)
    
    db_path = "simple_test.duckdb"
    
    try:
        # 1. Initialize storage
        print("\n🔧 Initializing storage...")
        storage_config = StorageConfig(
            connection_string=db_path,
            backend_type="duckdb"
        )
        
        raw_data_repo = DuckDBRawDataRepository(storage_config)
        episode_repo = DuckDBEpisodicMemoryRepository(storage_config) 
        semantic_repo = DuckDBSemanticMemoryRepository(storage_config)
        
        await raw_data_repo.initialize()
        await episode_repo.initialize()
        await semantic_repo.initialize()
        
        # 2. Initialize episode manager with conversation builder
        print("🏗️ Setting up episode manager...")
        from nemori.builders.conversation_builder import ConversationEpisodeBuilder
        
        builder_registry = EpisodeBuilderRegistry()
        
        # Register conversation builder
        conversation_builder = ConversationEpisodeBuilder()
        builder_registry.register(conversation_builder)
        print("✅ Registered conversation builder")
        
        episode_manager = EpisodeManager(
            raw_data_repo=raw_data_repo,
            episode_repo=episode_repo,
            builder_registry=builder_registry
        )
        
        # 3. Create sample conversation data
        print("\n📝 Creating sample conversation...")
        
        # Conversation content should be a list of message dictionaries
        conversation_messages = [
            {
                "speaker_id": "Alice",
                "content": "我最近在研究大语言模型的微调技术，你有什么经验分享吗？",
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
            },
            {
                "speaker_id": "Bob", 
                "content": "我用过LoRA和QLoRA方法，效果不错。你具体想微调什么类型的任务？",
                "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat()
            },
            {
                "speaker_id": "Alice",
                "content": "主要是想做中文对话系统，我在用ChatGLM-6B作为基础模型",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            },
            {
                "speaker_id": "Bob",
                "content": "ChatGLM不错，建议你试试P-Tuning v2，对中文效果很好。数据集准备了多少？", 
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        raw_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=conversation_messages,  # List of message dicts
            temporal_info=TemporalInfo(timestamp=datetime.now()),
            metadata={
                "participants": ["Alice", "Bob"],
                "topic": "AI模型微调"
            }
        )
        
        # 4. Process the event
        print("🔄 Processing event through episode manager...")
        owner_id = "Alice"
        
        # Process raw data directly into episode
        episode = await episode_manager.process_raw_data(raw_data, owner_id)
        
        if episode:
            print(f"✅ Episode created:")
            print(f"   ID: {episode.episode_id}")
            print(f"   Owner: {episode.owner_id}")
            print(f"   Title: {episode.title}")
            print(f"   Content length: {len(episode.content)} chars")
            print(f"   Summary: {episode.summary[:100]}...")
        else:
            print("⚠️ No episode was created (no suitable builder found)")
            return
        
        # 5. Test episode storage and retrieval
        print("\n🔍 Testing episode retrieval...")
        stored_episodes = await episode_repo.get_episodes_by_owner(owner_id)
        print(f"✅ Found {len(stored_episodes.episodes)} episodes for owner '{owner_id}'")
        
        if stored_episodes.episodes:
            ep = stored_episodes.episodes[0]
            print(f"   First episode: {ep.title}")
            print(f"   Type: {ep.episode_type.value}")
            print(f"   Timestamp: {ep.temporal_info.timestamp}")
        
        # 6. Test semantic storage (if available)
        print("\n🧠 Testing semantic memory...")
        semantic_nodes = await semantic_repo.get_all_semantic_nodes_for_owner(owner_id)
        print(f"📊 Found {len(semantic_nodes)} semantic concepts for '{owner_id}'")
        
        if semantic_nodes:
            for i, node in enumerate(semantic_nodes[:3]):  # Show first 3
                print(f"   {i+1}. {node.key}: {node.value}")
                print(f"      Confidence: {node.confidence:.2f}")
        else:
            print("   No semantic concepts found (this is expected without LLM integration)")
        
        # 7. Test episode statistics
        print("\n📊 Episode statistics...")
        # Use get_episodes_by_owner for all episodes
        all_episodes_result = await episode_repo.get_episodes_by_owner(owner_id)
        print(f"   Episodes for '{owner_id}': {len(all_episodes_result.episodes)}")
        
        for ep in all_episodes_result.episodes:
            print(f"   - {ep.owner_id}: {ep.title[:30]}... (importance: {ep.importance_score:.2f})")
        
        print("\n🎉 Basic Nemori test completed successfully!")
        print("✨ Demonstrated:")
        print("  ✓ Raw data ingestion")
        print("  ✓ Episode building")
        print("  ✓ Storage and retrieval")
        print("  ✓ Basic episode management")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await raw_data_repo.close()
        await episode_repo.close()
        await semantic_repo.close()
        
        # Remove test database
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
            print("✅ Test database cleaned up")


if __name__ == "__main__":
    asyncio.run(test_basic_nemori())