"""
Unified Episodic and Semantic Memory Demo

This script demonstrates the complete Nemori system combining episodic memory 
construction with semantic memory discovery, following the ingestion_emb.py pattern.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from nemori.core.data_types import (
    DataType,
    RawEventData, 
    TemporalInfo,
    ConversationMessage,
    SemanticNode,
    SemanticRelationship,
    RelationshipType
)
from nemori.core.episode import Episode, EpisodeType, EpisodeLevel, EpisodeMetadata
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.retrieval.service import RetrievalService
from nemori.retrieval.retrieval_types import RetrievalConfig, RetrievalStrategy, RetrievalStorageType
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder


class UnifiedMemoryExperiment:
    """
    Unified experiment class that demonstrates both episodic and semantic memory.
    """
    
    def __init__(self, db_dir: Path = Path("unified_demo_db")):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.llm_provider = None
        self.episodic_storage = None
        self.semantic_storage = None
        self.retrieval_service = None
        self.unified_retrieval = None
        self.episode_manager = None
        self.semantic_discovery = None
        self.semantic_evolution = None
        
        # Data
        self.episodes = []
        self.semantic_nodes = []
        
    async def setup_llm_provider(self, model: str = "gpt-4o-mini", api_key: str = "", base_url: str = ""):
        """Setup OpenAI LLM provider."""
        print("\n🤖 Setting up LLM Provider")
        print("=" * 50)
        
        self.llm_provider = OpenAIProvider(
            model=model,
            temperature=0.1,
            max_tokens=16 * 1024,
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=base_url or os.getenv('OPENAI_BASE_URL')
        )
        
        try:
            if await self.llm_provider.test_connection():
                print("✅ LLM connection successful!")
                print(f"🎯 Model: {self.llm_provider.model}")
                return True
            else:
                print("❌ LLM connection failed!")
                return False
        except Exception as e:
            print(f"❌ LLM connection error: {e}")
            return False
            
    async def setup_storage_and_retrieval(self, emb_api_key: str = "", emb_base_url: str = "", embed_model: str = ""):
        """Setup unified storage system with both episodic and semantic memories."""
        print("\n🗄️ Setting up Unified Memory Storage")
        print("=" * 50)
        
        # Clean up existing data
        db_path = self.db_dir / "unified_memory.duckdb"
        if db_path.exists():
            db_path.unlink()
            print("🧹 Cleaned existing database")
            
        for index_file in self.db_dir.glob("*_index_*.pkl"):
            index_file.unlink()
            print(f"🧹 Cleaned existing index: {index_file.name}")
            
        # Create storage configurations
        storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(db_path),
            batch_size=100,
            cache_size=1000
        )
        
        # Add embedding configuration to semantic storage
        semantic_config = StorageConfig(
            backend_type="duckdb", 
            connection_string=str(db_path),
            batch_size=100,
            cache_size=1000
        )
        semantic_config.openai_api_key = emb_api_key or os.getenv('OPENAI_API_KEY')
        semantic_config.openai_base_url = emb_base_url or os.getenv('OPENAI_BASE_URL')
        
        # Initialize storage repositories
        self.episodic_storage = DuckDBEpisodicMemoryRepository(storage_config)
        self.semantic_storage = DuckDBSemanticMemoryRepository(semantic_config)
        
        await self.episodic_storage.initialize()
        await self.semantic_storage.initialize()
        print(f"✅ Unified storage initialized: {db_path}")
        
        # Setup retrieval services
        self.retrieval_service = RetrievalService(self.episodic_storage)
        self.unified_retrieval = UnifiedRetrievalService(self.episodic_storage, self.semantic_storage)
        
        # Setup embedding retrieval
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.db_dir)},
            api_key=emb_api_key or os.getenv('OPENAI_API_KEY'),
            base_url=emb_base_url or os.getenv('OPENAI_BASE_URL'),
            embed_model=embed_model or "text-embedding-ada-002"
        )
        
        self.retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
        await self.retrieval_service.initialize()
        print("✅ Retrieval services initialized")
        
        # Setup semantic processing components
        self.semantic_discovery = ContextAwareSemanticDiscoveryEngine(
            llm_provider=self.llm_provider,
            retrieval_service=self.unified_retrieval
        )
        
        self.semantic_evolution = SemanticEvolutionManager(
            storage=self.semantic_storage,
            discovery_engine=self.semantic_discovery,
            retrieval_service=self.unified_retrieval
        )
        print("✅ Semantic processing components initialized")
        
    async def setup_episode_manager(self):
        """Setup episode manager with enhanced semantic-aware builder."""
        print("\n🏗️ Setting up Episode Manager")
        print("=" * 40)
        
        # Create enhanced conversation builder with semantic capabilities
        enhanced_builder = EnhancedConversationEpisodeBuilder(
            llm_provider=self.llm_provider,
            semantic_manager=self.semantic_evolution
        )
        
        # Create episode manager
        self.episode_manager = EpisodeManager(
            storage_repo=self.episodic_storage,
            retrieval_service=self.retrieval_service
        )
        
        # Register the enhanced builder
        self.episode_manager.register_builder(DataType.CONVERSATION, enhanced_builder)
        print("✅ Enhanced episode manager configured")
        
    async def create_sample_conversations(self):
        """Create sample conversation data for demonstration."""
        print("\n📝 Creating Sample Conversations")
        print("=" * 40)
        
        conversations = [
            {
                "conversation_id": "conv_1",
                "participants": ["Alice", "Bob"],
                "messages": [
                    {
                        "speaker_id": "Alice",
                        "content": "Hi Bob! I've been learning machine learning recently, focusing on deep neural networks.",
                        "timestamp": "2024-01-15T10:00:00"
                    },
                    {
                        "speaker_id": "Bob", 
                        "content": "That's awesome Alice! I'm also interested in AI. I've been working with PyTorch and TensorFlow.",
                        "timestamp": "2024-01-15T10:01:00"
                    },
                    {
                        "speaker_id": "Alice",
                        "content": "Great! I prefer PyTorch for research. Are you familiar with transformer architectures?",
                        "timestamp": "2024-01-15T10:02:00"
                    },
                    {
                        "speaker_id": "Bob",
                        "content": "Yes! I've implemented BERT and GPT models. Currently exploring multimodal transformers.",
                        "timestamp": "2024-01-15T10:03:00"
                    }
                ]
            },
            {
                "conversation_id": "conv_2", 
                "participants": ["Alice", "Carol"],
                "messages": [
                    {
                        "speaker_id": "Alice",
                        "content": "Carol, do you know any good restaurants in the downtown area?",
                        "timestamp": "2024-01-16T12:00:00"
                    },
                    {
                        "speaker_id": "Carol",
                        "content": "Yes! I love Italian food. There's this amazing place called 'Luigi's' on Main Street.",
                        "timestamp": "2024-01-16T12:01:00"
                    },
                    {
                        "speaker_id": "Alice",
                        "content": "That sounds perfect! I'm also a fan of Italian cuisine. Do they have good pasta?",
                        "timestamp": "2024-01-16T12:02:00"
                    },
                    {
                        "speaker_id": "Carol",
                        "content": "The best! Their carbonara is my favorite. They also have excellent wine selection.",
                        "timestamp": "2024-01-16T12:03:00"
                    }
                ]
            },
            {
                "conversation_id": "conv_3",
                "participants": ["Alice", "Bob"],
                "messages": [
                    {
                        "speaker_id": "Alice",
                        "content": "Bob, I've made progress on my machine learning project. Now I'm working on reinforcement learning!",
                        "timestamp": "2024-02-01T14:00:00"
                    },
                    {
                        "speaker_id": "Bob",
                        "content": "That's exciting Alice! RL is challenging but rewarding. Are you using Q-learning or policy gradients?",
                        "timestamp": "2024-02-01T14:01:00"
                    },
                    {
                        "speaker_id": "Alice",
                        "content": "I started with Q-learning, but now I'm exploring actor-critic methods. The results are promising!",
                        "timestamp": "2024-02-01T14:02:00"
                    },
                    {
                        "speaker_id": "Bob",
                        "content": "Actor-critic is powerful! Have you tried it with neural networks? Deep RL could be your next step.",
                        "timestamp": "2024-02-01T14:03:00"
                    }
                ]
            }
        ]
        
        return conversations
        
    def convert_conversation_to_raw_data(self, conv_data: Dict[str, Any]) -> RawEventData:
        """Convert conversation data to RawEventData format."""
        messages = []
        
        for msg in conv_data["messages"]:
            conversation_msg = ConversationMessage(
                speaker_id=msg["speaker_id"],
                user_name=msg["speaker_id"],  # Use speaker_id as display name
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"])
            )
            messages.append(conversation_msg)
            
        # Get temporal info from first and last messages
        first_time = datetime.fromisoformat(conv_data["messages"][0]["timestamp"])
        last_time = datetime.fromisoformat(conv_data["messages"][-1]["timestamp"])
        duration = (last_time - first_time).total_seconds()
        
        temporal_info = TemporalInfo(
            timestamp=first_time,
            duration=duration,
            timezone="UTC",
            precision="second"
        )
        
        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=messages,
            source=f"demo_conversation_{conv_data['conversation_id']}",
            temporal_info=temporal_info,
            metadata={"conversation_id": conv_data["conversation_id"], "participants": conv_data["participants"]}
        )
        
    async def process_conversations_unified(self):
        """Process conversations with unified episodic and semantic memory."""
        print("\n🔄 Processing Conversations with Unified Memory")
        print("=" * 60)
        
        conversations = await self.create_sample_conversations()
        
        for i, conv_data in enumerate(conversations):
            conv_id = conv_data["conversation_id"]
            print(f"\n📖 Processing conversation {i+1}/3: {conv_id}")
            print("-" * 50)
            
            # Convert to RawEventData
            raw_data = self.convert_conversation_to_raw_data(conv_data)
            
            # Build episodes with automatic semantic discovery
            try:
                episodes = await self.episode_manager.process_event(
                    raw_data, 
                    for_owner=conv_data["participants"][0]  # Use first participant as owner
                )
                
                print(f"✅ Created {len(episodes)} episodes from {conv_id}")
                
                # Display episode details
                for j, episode in enumerate(episodes):
                    print(f"  📝 Episode {j+1}: {episode.title}")
                    
                    # Check for semantic discoveries
                    if "semantic_node_ids" in episode.metadata.custom_fields:
                        semantic_count = episode.metadata.custom_fields.get("discovered_semantics", 0)
                        print(f"    🧠 Discovered {semantic_count} semantic concepts")
                        
                self.episodes.extend(episodes)
                    
            except Exception as e:
                print(f"❌ Error processing {conv_id}: {e}")
                
        print(f"\n🎯 Total episodes created: {len(self.episodes)}")
        
    async def analyze_semantic_discoveries(self):
        """Analyze the semantic knowledge discovered from conversations."""
        print("\n🧠 Analyzing Semantic Memory Discoveries")
        print("=" * 60)
        
        # Get all semantic nodes for each participant
        participants = ["Alice", "Bob", "Carol"]
        
        for participant in participants:
            print(f"\n👤 Semantic Knowledge for {participant}:")
            print("-" * 40)
            
            nodes = await self.semantic_storage.get_all_semantic_nodes_for_owner(participant)
            
            if not nodes:
                print("   No semantic knowledge discovered")
                continue
                
            for node in nodes:
                print(f"  🔹 {node.key}: {node.value}")
                print(f"    📊 Confidence: {node.confidence:.2f} | Version: {node.version}")
                if node.evolution_history:
                    print(f"    🔄 Evolution: {' → '.join(node.evolution_history)} → {node.value}")
                print()
                
            # Get statistics
            stats = await self.semantic_storage.get_semantic_statistics(participant)
            print(f"  📈 Statistics:")
            print(f"    • Total nodes: {stats['node_count']}")
            print(f"    • Average confidence: {stats['average_confidence']:.2f}")
            print(f"    • Relationships: {stats['relationship_count']}")
            
    async def demonstrate_unified_retrieval(self):
        """Demonstrate unified retrieval across both memory types."""
        print("\n🔍 Unified Memory Retrieval Demonstration")
        print("=" * 60)
        
        queries = [
            "machine learning and neural networks",
            "Italian food restaurants", 
            "reinforcement learning progress",
            "PyTorch and TensorFlow frameworks"
        ]
        
        for query in queries:
            print(f"\n🔎 Query: '{query}'")
            print("-" * 30)
            
            # Search episodic memories
            episodic_results = await self.unified_retrieval.search_episodic_memories("Alice", query, limit=2)
            print(f"📚 Episodic Results ({len(episodic_results)}):")
            for episode in episodic_results:
                print(f"  📖 {episode.title}")
                
            # Search semantic memories
            semantic_results = await self.unified_retrieval.search_semantic_memories("Alice", query, limit=3)
            print(f"🧠 Semantic Results ({len(semantic_results)}):")
            for node in semantic_results:
                print(f"  🔹 {node.key}: {node.value}")
                
            # Enhanced query combining both
            enhanced_results = await self.unified_retrieval.enhanced_query("Alice", query, episode_limit=1, semantic_limit=2)
            print(f"🔥 Enhanced Combined Results:")
            print(f"  📚 {len(enhanced_results['episodes'])} episodes + 🧠 {len(enhanced_results['semantic_knowledge'])} concepts")
            
    async def run_complete_demo(self):
        """Run the complete unified memory demonstration."""
        print("🚀 Unified Episodic & Semantic Memory System")
        print("=" * 70)
        print("🎯 Demonstrating complete memory integration following ingestion pattern")
        
        try:
            # Step 1: Setup LLM
            api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
            base_url = "https://jeniya.cn/v1"
            model = "gpt-4o-mini"
            
            llm_success = await self.setup_llm_provider(model, api_key, base_url)
            if not llm_success:
                print("⚠️ Continuing with limited functionality")
                
            # Step 2: Setup storage and retrieval
            emb_api_key = api_key  # Use same API key for embeddings
            emb_base_url = base_url
            embed_model = "text-embedding-ada-002"
            
            await self.setup_storage_and_retrieval(emb_api_key, emb_base_url, embed_model)
            
            # Step 3: Setup episode manager
            await self.setup_episode_manager()
            
            # Step 4: Process conversations (builds episodes + discovers semantics)
            await self.process_conversations_unified()
            
            # Step 5: Analyze semantic discoveries
            await self.analyze_semantic_discoveries()
            
            # Step 6: Demonstrate unified retrieval
            await self.demonstrate_unified_retrieval()
            
            print("\n🎉 Unified Memory Demo Complete!")
            print("=" * 70)
            print("✨ Successfully demonstrated:")
            print("  ✓ Episodic memory construction with boundary detection")
            print("  ✓ Automatic semantic knowledge discovery")
            print("  ✓ Knowledge evolution and relationship tracking")  
            print("  ✓ Embedding-based similarity search")
            print("  ✓ Unified retrieval across both memory types")
            print("  ✓ Complete ingestion pipeline integration")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if self.episodic_storage:
                await self.episodic_storage.close()
            if self.semantic_storage:
                await self.semantic_storage.close()
            if self.retrieval_service:
                await self.retrieval_service.close()
                
            # Clean up demo database
            db_path = self.db_dir / "unified_memory.duckdb"
            if db_path.exists():
                db_path.unlink()
                print("\n🧹 Demo database cleaned up")


async def main():
    """Main function following the ingestion_emb.py pattern."""
    experiment = UnifiedMemoryExperiment()
    await experiment.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())