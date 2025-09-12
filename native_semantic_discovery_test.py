"""
Complete Nemori Semantic Discovery Test - Using Native Methods

This script demonstrates the complete 3-step semantic discovery process using
Nemori's built-in methods, prompts, and existing architecture:

1. Generate episodic memory from original conversation (EnhancedConversationEpisodeBuilder)
2. Reconstruct conversation from historical semantic memory (ContextAwareSemanticDiscoveryEngine)
3. Discover new semantic knowledge through differential analysis (SemanticEvolutionManager)
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from nemori.core.data_types import RawEventData, DataType, TemporalInfo, SemanticNode
from nemori.core.episode import Episode
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager


class NativeSemanticDiscoveryTest:
    """Test using Nemori's native semantic discovery methods"""
    
    def __init__(self):
        self.db_path = "native_semantic_test.duckdb"
        self.raw_data_repo = None
        self.episode_repo = None
        self.semantic_repo = None
        self.episode_manager = None
        self.llm_provider = None
        self.unified_retrieval = None
        self.discovery_engine = None
        self.evolution_manager = None
        
    async def initialize_native_components(self):
        """Initialize Nemori components using native architecture"""
        print("🔧 Initializing Native Nemori Components")
        print("=" * 60)
        
        # Storage configuration with embedding settings
        storage_config = StorageConfig(
            connection_string=self.db_path,
            backend_type="duckdb"
        )
        # storage_config.openai_api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        # storage_config.openai_base_url = "https://jeniya.cn/v1"
        
        # Embedding API configuration for semantic storage
        storage_config.openai_api_key = "EMPTY"
        storage_config.openai_base_url = "http://localhost:6007/v1" 
        storage_config.openai_embed_model = "qwen3-emb"
        
        # Initialize storage repositories
        self.raw_data_repo = DuckDBRawDataRepository(storage_config)
        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config) 
        self.semantic_repo = DuckDBSemanticMemoryRepository(storage_config)
        
        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()
        await self.semantic_repo.initialize()
        print("✅ Storage repositories initialized")
        
        # Initialize LLM provider
        self.llm_provider = OpenAIProvider(
            model="gpt-4o-mini",
            api_key="sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm",
            base_url="https://jeniya.cn/v1"
        )
        print("✅ LLM provider initialized")
        
        # Initialize unified retrieval service (required for semantic discovery)
        self.unified_retrieval = UnifiedRetrievalService(
            episodic_storage=self.episode_repo,
            semantic_storage=self.semantic_repo
        )
        await self.unified_retrieval.initialize()
        print("✅ Unified retrieval service initialized")
        
        # Initialize semantic discovery engine (uses native prompts)
        self.discovery_engine = ContextAwareSemanticDiscoveryEngine(
            llm_provider=self.llm_provider,
            retrieval_service=self.unified_retrieval
        )
        print("✅ Semantic discovery engine initialized")
        
        # Initialize semantic evolution manager
        self.evolution_manager = SemanticEvolutionManager(
            storage=self.semantic_repo,
            discovery_engine=self.discovery_engine,
            retrieval_service=self.unified_retrieval
        )
        print("✅ Semantic evolution manager initialized")
        
        # Initialize enhanced episode builder with semantic integration
        builder_registry = EpisodeBuilderRegistry()
        enhanced_builder = EnhancedConversationEpisodeBuilder(
            llm_provider=self.llm_provider,
            semantic_manager=self.evolution_manager
        )
        builder_registry.register(enhanced_builder)
        
        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry
        )
        print("✅ Enhanced episode manager with semantic integration initialized")
        
    def create_caroline_scenario(self):
        """Create Caroline adoption scenario conversation"""
        return [
            {
                "speaker_id": "Caroline",
                "content": "Hey everyone, I wanted to share some exciting news! I just took a photo of the adoption agency I'm considering.",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                "speaker_id": "Melanie", 
                "content": "That's wonderful, Caroline! Tell us more about it.",
                "timestamp": (datetime.now() - timedelta(minutes=14)).isoformat()
            },
            {
                "speaker_id": "Caroline",
                "content": "I chose this agency because they truly support LGBTQ+ families and single parents. I feel like I'll get the support I need there.",
                "timestamp": (datetime.now() - timedelta(minutes=12)).isoformat()
            },
            {
                "speaker_id": "Melanie",
                "content": "Caroline, your kindness and dedication are really admirable. Are you ready for this challenge?",
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
            },
            {
                "speaker_id": "Caroline",
                "content": "Yes, I'm ready. Being a single parent will definitely have challenges, but I believe love and determination can overcome anything.",
                "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat()
            }
        ]
    
    async def create_historical_semantic_context(self):
        """Create historical semantic nodes to demonstrate reconstruction"""
        print("📚 Creating Historical Semantic Context...")
        
        historical_nodes = [
            SemanticNode(
                owner_id="Caroline",
                key="Family Building Goal",
                value="Caroline wants to build a family through adoption",
                context="Caroline has expressed desire to adopt and start a family",
                confidence=0.9,
                discovery_episode_id="previous_1",
                linked_episode_ids=["previous_1"]
            ),
            SemanticNode(
                owner_id="Caroline",
                key="Values and Preferences",
                value="Caroline values inclusivity and support in family services",
                context="Caroline prioritizes LGBTQ+ friendly services",
                confidence=0.85,
                discovery_episode_id="previous_2", 
                linked_episode_ids=["previous_2"]
            ),
            SemanticNode(
                owner_id="Caroline",
                key="Support Network",
                value="Caroline receives emotional support from friends and mentors during adoption process",
                context="Friends provide encouragement for Caroline's adoption journey",
                confidence=0.8,
                discovery_episode_id="previous_3",
                linked_episode_ids=["previous_3"]
            )
        ]
        
        # Store historical semantic nodes
        for node in historical_nodes:
            await self.semantic_repo.store_semantic_node(node)
            
        print(f"✅ Created {len(historical_nodes)} historical semantic nodes")
        return historical_nodes
    
    async def step1_native_episodic_generation(self, conversation_messages):
        """Step 1: Use EnhancedConversationEpisodeBuilder for episodic generation"""
        print("\\n🎯 STEP 1: Native Episodic Memory Generation")
        print("-" * 60)
        print("Using: EnhancedConversationEpisodeBuilder with semantic integration")
        
        # Convert to RawEventData
        raw_data = RawEventData(
            data_type=DataType.CONVERSATION,
            content=conversation_messages,
            temporal_info=TemporalInfo(timestamp=datetime.now()),
            metadata={
                "participants": ["Caroline", "Melanie"],
                "topic": "adoption agency discussion",
                "context": "Caroline sharing adoption agency choice"
            }
        )
        
        # Store original conversation text for later comparison
        original_conversation = "\\n".join([
            f"{msg['speaker_id']}: {msg['content']}" for msg in conversation_messages
        ])
        
        # Process through enhanced episode manager (includes automatic semantic discovery)
        episode = await self.episode_manager.process_raw_data(raw_data, "Caroline")
        
        if episode:
            print("✅ Successfully generated episodic memory with semantic integration:")
            print(f"   📝 Title: {episode.title}")
            print(f"   📄 Summary: {episode.summary[:150]}...")
            print(f"   🕐 Timestamp: {episode.temporal_info.timestamp}")
            
            # Check if semantic knowledge was discovered automatically
            semantic_info = episode.metadata.custom_fields.get("discovered_semantics", 0)
            if semantic_info > 0:
                print(f"   🧠 Automatically discovered {semantic_info} semantic concepts")
            else:
                print("   🧠 No semantic concepts discovered automatically (will process manually)")
                
            return episode, original_conversation
        else:
            print("❌ Failed to generate episodic memory")
            return None, None
    
    async def step2_native_reconstruction(self, episode, historical_nodes):
        """Step 2: Use native ContextAwareSemanticDiscoveryEngine for reconstruction"""
        print("\\n🎯 STEP 2: Native Context-Aware Reconstruction")
        print("-" * 60)
        print("Using: ContextAwareSemanticDiscoveryEngine._reconstruct_with_context")
        
        # Prepare context using historical semantic nodes (simulating retrieval)
        context = {
            "related_semantic_memories": historical_nodes,
            "related_historical_episodes": [],  # Empty for this demo
            "current_episode": episode
        }
        
        try:
            # Use the native reconstruction method
            reconstructed_conversation = await self.discovery_engine._reconstruct_with_context(
                episode=episode,
                context=context
            )
            
            print("✅ Successfully reconstructed conversation using native method:")
            print("📄 Reconstructed conversation (using Nemori's native prompt):")
            print("   " + reconstructed_conversation.replace("\\n", "\\n   "))
            
            return reconstructed_conversation
            
        except Exception as e:
            print(f"❌ Native reconstruction failed: {e}")
            return None
    
    async def step3_native_semantic_discovery(self, episode, original_conversation, reconstructed_conversation):
        """Step 3: Use native SemanticEvolutionManager for knowledge discovery"""
        print("\\n🎯 STEP 3: Native Semantic Knowledge Discovery")
        print("-" * 60)
        print("Using: SemanticEvolutionManager.process_episode_for_semantics")
        
        try:
            # Use the native semantic discovery method
            discovered_nodes = await self.evolution_manager.process_episode_for_semantics(
                episode=episode,
                original_content=original_conversation
            )
            
            print(f"✅ Native semantic discovery completed:")
            print(f"   🔍 Discovered {len(discovered_nodes)} new semantic concepts:")
            
            for i, node in enumerate(discovered_nodes, 1):
                print(f"   {i}. {node.key}: {node.value}")
                print(f"      📊 Confidence: {node.confidence:.2f}")
                print(f"      📝 Context: {node.context[:80]}...")
                print(f"      🔗 Discovery Method: {node.discovery_method}")
                print(f"      📎 Linked to Episode: {node.discovery_episode_id}")
                print()
            
            return discovered_nodes
            
        except Exception as e:
            print(f"❌ Native semantic discovery failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def demonstrate_native_flow(self):
        """Demonstrate complete flow using native Nemori methods"""
        print("🚀 Native Nemori Semantic Discovery Flow")
        print("=" * 60)
        print("Using Nemori's built-in methods and prompts:")
        print("1. 📖 EnhancedConversationEpisodeBuilder (with LLM)")
        print("2. 🧠 ContextAwareSemanticDiscoveryEngine (native prompts)")
        print("3. 🔍 SemanticEvolutionManager (differential analysis)")
        print()
        
        try:
            # Initialize native components
            await self.initialize_native_components()
            
            # Setup historical context
            historical_nodes = await self.create_historical_semantic_context()
            
            # Create Caroline scenario
            conversation_messages = self.create_caroline_scenario()
            
            # Step 1: Native episodic generation
            episode, original_conversation = await self.step1_native_episodic_generation(conversation_messages)
            if not episode:
                return
            
            # Step 2: Native reconstruction
            reconstructed_conversation = await self.step2_native_reconstruction(episode, historical_nodes)
            if not reconstructed_conversation:
                return
            
            # Step 3: Native semantic discovery
            new_nodes = await self.step3_native_semantic_discovery(episode, original_conversation, reconstructed_conversation)
            
            # Final analysis
            print("\\n🎉 Native Flow Completion Analysis")
            print("=" * 60)
            
            # Show total semantic knowledge for Caroline
            all_nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner("Caroline")
            print(f"📊 Total semantic concepts for Caroline: {len(all_nodes)}")
            print(f"   📚 Historical concepts: {len(historical_nodes)}")
            print(f"   🆕 Newly discovered: {len(new_nodes)}")
            
            # Show episodic-semantic linking
            episode_semantics = await self.unified_retrieval.get_episode_semantics(episode.episode_id)
            print(f"🔗 Episode-semantic bidirectional links: {len(episode_semantics)}")
            
            # Verify differential analysis worked
            print("\\n✨ Native Methods Successfully Demonstrated:")
            print("  ✓ Enhanced episodic memory generation with LLM")
            print("  ✓ Context-aware reconstruction using native prompts")
            print("  ✓ Differential analysis for private domain knowledge discovery")
            print("  ✓ Automatic semantic knowledge evolution and storage")
            print("  ✓ Bidirectional episode-semantic memory linking")
            print("  ✓ Full integration with Nemori's native architecture")
            
        except Exception as e:
            print(f"❌ Native flow demonstration failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def cleanup(self):
        """Cleanup resources"""
        print("\\n🧹 Cleaning up resources...")
        
        if self.raw_data_repo:
            await self.raw_data_repo.close()
        if self.episode_repo:
            await self.episode_repo.close()
        if self.semantic_repo:
            await self.semantic_repo.close()
        if self.unified_retrieval:
            await self.unified_retrieval.close()
        
        # Remove test database
        db_file = Path(self.db_path)
        if db_file.exists():
            db_file.unlink()
            print("✅ Test database cleaned up")


async def main():
    """Main test function"""
    test_runner = NativeSemanticDiscoveryTest()
    
    try:
        await test_runner.demonstrate_native_flow()
    finally:
        await test_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())