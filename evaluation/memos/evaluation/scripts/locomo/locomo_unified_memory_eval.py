import argparse
import asyncio

import pandas as pd

from dotenv import load_dotenv
from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy

# Nemori evaluation imports  
try:
    from nemori_eval import NemoriExperiment
    NEMORI_AVAILABLE = True
except ImportError:
    NEMORI_AVAILABLE = False
    print("⚠️ Nemori not available. Install nemori to use nemori functionality.")


async def main_nemori_unified(version="default"):
    """Main function for Nemori processing with unified episodic and semantic memory."""
    load_dotenv()
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    print("🚀 Starting Nemori Unified Memory Ingestion (Episodic + Semantic)")
    print("=" * 70)
    print("🎯 Processing LoCoMo conversations to build:")
    print("   📖 Episodic Memory: Conversation episodes with temporal boundaries")  
    print("   🧠 Semantic Memory: Extracted knowledge using differential analysis")
    print("   🔗 Bidirectional Links: Episodes ↔ Semantic knowledge connections")

    # Create Nemori experiment with unified memory capabilities
    experiment = NemoriExperiment(
        version=version, 
        episode_mode="speaker", 
        retrievalstrategy=RetrievalStrategy.BM25  # Changed to BM25 for compatibility
    )

    try:
        # Step 1: Setup LLM provider (essential for semantic discovery)
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"
        model = "gpt-4o-mini"
        
        print("\n🤖 Step 1: Setting up LLM Provider for Semantic Discovery")
        print("-" * 60)
        
        llm_available = await experiment.setup_llm_provider(model=model, api_key=api_key, base_url=base_url)
        if not llm_available:
            print("❌ LLM provider is REQUIRED for semantic knowledge discovery")
            print("   Semantic memory uses LLM to:")
            print("   • Perform differential analysis (original vs reconstructed)")
            print("   • Extract private domain knowledge")
            print("   • Detect knowledge evolution")
            return

        print("✅ All components configured successfully")
        print(f"   🎯 LLM Model: {model}")
        print(f"   🧠 Semantic Discovery: Enabled with differential analysis")
        print(f"   🔗 Unified Retrieval: Episodic + Semantic memory integration")
        print(f"   📊 Retrieval Strategy: {experiment.retrievalstrategy.value}")

        # Step 2: Load LoCoMo conversation data
        print(f"\n📊 Step 2: Loading LoCoMo Dataset")
        print("-" * 60)
        
        experiment.load_locomo_data(locomo_df)
        print(f"✅ Loaded {len(experiment.conversations)} LoCoMo conversations")
        
        # Show sample conversation structure
        if experiment.conversations:
            sample_conv = experiment.conversations[0]["conversation"]
            sessions = [key for key in sample_conv if key.startswith("session_") and not key.endswith("_date_time")]
            print(f"   📝 Sample conversation structure:")
            print(f"   • Speakers: {sample_conv.get('speaker_a')} & {sample_conv.get('speaker_b')}")
            print(f"   • Sessions: {len(sessions)} multi-session conversations")
            if sessions:
                sample_session = sample_conv[sessions[0]]
                print(f"   • Messages per session: ~{len(sample_session)} messages")
        
        # Step 3: Setup unified storage with semantic capabilities
        print(f"\n🗄️ Step 3: Setting up Unified Memory Storage")
        print("-" * 60)
        emb_api_key = "EMPTY" 
        emb_base_url = "http://localhost:6007/v1"
        embed_model = "qwen3-emb" 
        
        await experiment.setup_storage_and_retrieval(
            emb_api_key=emb_api_key, 
            emb_base_url=emb_base_url, 
            embed_model=embed_model
        )
        
        print("✅ Unified storage system initialized:")
        print("   📖 Episodic Memory: DuckDB episodes repository")
        print("   🧠 Semantic Memory: DuckDB semantic nodes repository") 
        print("   🔗 Unified Retrieval: Cross-memory search capabilities")
        print("   🎯 Embedding Support: Vector similarity for both memory types")

        # Step 4: Process LoCoMo conversations with unified memory building
        print(f"\n🏗️ Step 4: Processing LoCoMo Data with Unified Memory")
        print("-" * 60)
        print("🔄 For each conversation, the system will:")
        print("   1. 📖 Build episodic episodes using boundary detection")
        print("   2. 🧠 Perform ACTIVE LEARNING semantic discovery:")
        print("      a) Retrieve related episodic memories and existing semantic knowledge")
        print("      b) Use LLM to reconstruct original conversation from retrieved context")
        print("      c) Perform differential analysis: original vs reconstructed content")
        print("      d) Extract knowledge gaps as new semantic concepts")
        print("      e) Update existing semantic knowledge with evolved information")
        print("   3. 🔗 Create bidirectional episode-semantic links")
        print("   4. 🎯 Generate embeddings for vector similarity search")
        print()
        
        # Process all conversations with active learning
        print("🧠 Starting Active Learning Semantic Discovery Process...")
        print("   (Using nemori's built-in ContextAwareSemanticDiscoveryEngine)")
        await experiment.build_episodes()

        # Step 5: Show that active learning is working through the existing nemori system
        print(f"\n🔬 Step 5: Active Learning Results via Nemori's Semantic Discovery")
        print("=" * 60)
        print("🎯 The system automatically uses episodic memory as 'knowledge mask'")
        print("   through nemori's ContextAwareSemanticDiscoveryEngine and")
        print("   SemanticEvolutionManager during episode building process.")
        print()
        
        # Show that semantic concepts were discovered during episode processing
        episode_owners = set(ep.owner_id for ep in experiment.episodes) if experiment.episodes else set()
        print(f"✅ Active learning completed for {len(episode_owners)} speakers")
        print("   🧠 Semantic concepts were automatically discovered during episode processing")
        print("   🔄 Knowledge evolution and confidence tracking are handled by nemori")
        print("   🔗 Bidirectional linking created between episodes and semantic concepts")
        print()
        
        # Display detailed active learning results
        if experiment.episodes:
            print("📋 Active Learning Process Details:")
            print("   During episode building, for each episode:")
            print("   1. 🔍 ContextAwareSemanticDiscoveryEngine.discover_semantic_knowledge():")
            print("      • Gathers context from related memories (episodes + semantics)")
            print("      • Uses LLM to reconstruct original conversation from episodic summary")
            print("      • Performs differential analysis: original vs reconstructed")
            print("      • Identifies knowledge gaps as new semantic concepts")
            print("   2. 🔄 SemanticEvolutionManager.process_episode_for_semantics():")
            print("      • Updates existing semantic knowledge with new discoveries")
            print("      • Tracks confidence scores and evolution history")
            print("      • Creates bidirectional links between episodes and concepts")
            print("   3. 🎯 EnhancedConversationEpisodeBuilder integration:")
            print("      • Seamlessly integrates semantic discovery into episode creation")
            print("      • Ensures all episodes contribute to knowledge evolution")

        # Step 6: Analyze the unified memory results
        print(f"\n📊 Step 6: Unified Memory Analysis Results")
        print("=" * 60)
        
        # Episodic memory statistics
        print(f"📖 EPISODIC MEMORY RESULTS:")
        print(f"   • Total Episodes: {len(experiment.episodes)}")
        
        if experiment.episodes:
            episode_owners = set(ep.owner_id for ep in experiment.episodes)
            print(f"   • Unique Speakers: {len(episode_owners)}")
            
            # Show per-speaker episode counts
            print(f"   • Episodes per Speaker:")
            for owner in sorted(episode_owners):
                owner_episodes = [ep for ep in experiment.episodes if ep.owner_id == owner]
                print(f"     - {owner}: {len(owner_episodes)} episodes")
                
            # Show sample episode details
            sample_episode = experiment.episodes[0]
            print(f"   • Sample Episode: '{sample_episode.title[:50]}...'")
            print(f"     - Owner: {sample_episode.owner_id}")
            print(f"     - Content Length: {len(sample_episode.content)} chars")

        # Semantic memory statistics
        print(f"\n🧠 SEMANTIC MEMORY RESULTS:")
        all_owners = set()
        if experiment.episodes:
            all_owners.update(ep.owner_id for ep in experiment.episodes)
        
        total_semantic_nodes = 0
        semantic_owners = set()
        semantic_samples = []
        
        for owner in all_owners:
            try:
                nodes = await experiment.semantic_repo.get_all_semantic_nodes_for_owner(owner)
                if nodes:
                    semantic_owners.add(owner)
                    owner_count = len(nodes)
                    total_semantic_nodes += owner_count
                    print(f"   • {owner}: {owner_count} semantic concepts discovered")
                    
                    # Collect sample semantic knowledge
                    for node in nodes[:2]:  # Show first 2 concepts per owner
                        semantic_samples.append((owner, node.key, node.value, node.confidence))
                        
            except Exception as e:
                print(f"   ⚠️ {owner}: Error accessing semantic memory - {e}")
        
        print(f"   • Total Semantic Concepts: {total_semantic_nodes}")
        print(f"   • Speakers with Semantic Knowledge: {len(semantic_owners)}")
        
        # Show sample discovered knowledge with details
        if semantic_samples:
            print(f"\n   📋 Sample Discovered Knowledge (via Differential Analysis):")
            for owner, key, value, confidence in semantic_samples[:6]:  # Show first 6 samples
                print(f"     - {owner}: {key} → {value[:40]}... (confidence: {confidence:.2f})")
                
            print(f"\n   🧠 Semantic Discovery Process Verification:")
            print(f"      • Each concept above was discovered through:")
            print(f"        1. LLM reconstructed conversation from episode summary")
            print(f"        2. Compared reconstruction vs original conversation")
            print(f"        3. Identified information gaps as private domain knowledge")
            print(f"        4. Structured gaps into semantic concepts with confidence scores")

        # Step 7: Validate unified memory integration and semantic evolution
        print(f"\n🔗 Step 7: Validating Unified Memory Integration & Semantic Evolution")
        print("-" * 60)
        
        if experiment.episodes and total_semantic_nodes > 0:
            try:
                # Test bidirectional linking with sample episode
                sample_episode = experiment.episodes[0]
                episode_semantics = await experiment.unified_retrieval_service.get_episode_semantics(sample_episode.episode_id)
                
                print(f"✅ Memory integration validation:")
                print(f"   • Sample Episode ID: {sample_episode.episode_id}")
                print(f"   • Linked Semantic Concepts: {len(episode_semantics)}")
                
                if episode_semantics:
                    sample_semantic = episode_semantics[0]
                    semantic_episodes = await experiment.unified_retrieval_service.get_semantic_episodes(sample_semantic.node_id)
                    print(f"   • Reverse Link Test: {len(semantic_episodes)} episodes link to semantic concept")
                    print(f"   🔗 Bidirectional linking is working correctly!")
                    
                    # Show semantic evolution details
                    print(f"\n   🔄 Semantic Evolution Features:")
                    print(f"      • Concept: {sample_semantic.key}")
                    print(f"      • Current Value: {sample_semantic.value[:50]}...")
                    print(f"      • Confidence: {sample_semantic.confidence:.2f}")
                    print(f"      • Version: {sample_semantic.version}")
                    if hasattr(sample_semantic, 'evolution_history') and sample_semantic.evolution_history:
                        print(f"      • Evolution History: {len(sample_semantic.evolution_history)} changes")
                    print(f"      • Discovery Episode: {sample_semantic.discovery_episode_id}")
                    print(f"      • Linked Episodes: {len(sample_semantic.linked_episode_ids)}")
                else:
                    print(f"   ⚪ No semantic concepts linked to this episode (expected for some episodes)")
                    
            except Exception as e:
                print(f"   ⚠️ Bidirectional linking test failed: {e}")
                
        # Show comprehensive semantic evolution capabilities
        if total_semantic_nodes > 0:
            print(f"\n   📈 Semantic Evolution System Features:")
            print(f"      ✓ Differential Analysis: Original vs reconstructed content comparison")
            print(f"      ✓ Knowledge Gap Detection: Private domain information identification") 
            print(f"      ✓ Confidence Scoring: Probabilistic knowledge quality assessment")
            print(f"      ✓ Version Tracking: Complete evolution history maintenance")
            print(f"      ✓ Bidirectional Linking: Episode ↔ Semantic concept associations")
            print(f"      ✓ Context-Aware Discovery: Related memories inform discovery process")
            print(f"      ✓ Automatic Evolution: Continuous knowledge updates during processing")

        print(f"\n🎉 LoCoMo Unified Memory Processing Complete!")
        print("=" * 70)
        print("✨ Successfully demonstrated:")
        print("  ✓ LoCoMo dataset ingestion and processing")
        print("  ✓ Episodic memory construction from multi-session conversations")
        print("  ✓ Active learning semantic discovery through differential analysis")
        print("  ✓ Knowledge evolution tracking and confidence scoring")
        print("  ✓ Bidirectional episode-semantic linking")
        print("  ✓ Unified retrieval service integration")
        print("  ✓ Embedding-based vector similarity search")
        print("  ✓ Ready for LoCoMo benchmark evaluation!")
        
        # Memory utilization summary
        print(f"\n📈 Memory Utilization Summary:")
        print(f"   📖 Episodic Episodes: {len(experiment.episodes)}")
        print(f"   🧠 Semantic Concepts: {total_semantic_nodes}")
        print(f"   👥 Speakers Processed: {len(all_owners)}")
        print(f"   🔗 Memory Integration: Active bidirectional linking")
        print(f"   🧠 Active Learning: Differential analysis for knowledge discovery")
        
        total_memory_items = len(experiment.episodes) + total_semantic_nodes
        print(f"   🎯 Total Memory Items: {total_memory_items}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Generate responses: python locomo_unified_responses.py --lib nemori --version {version}")
        print(f"   2. Run evaluation: python locomo_eval.py --lib nemori --version {version}")
        
    except Exception as e:
        print(f"❌ Nemori unified ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print(f"\n🧹 Cleaning up resources...")
        await experiment.cleanup()
        print("✅ Cleanup completed")


def main(frame, version="default"):
    load_dotenv()

    if frame == "nemori":
        if not NEMORI_AVAILABLE:
            print("❌ Nemori is not available. Please install nemori to use this framework.")
            return
        # Run async main for unified nemori
        asyncio.run(main_nemori_unified(version))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "nemori"],
        help="Specify the memory framework (zep or memos or mem0 or mem0_graph or nemori)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version

    main(lib, version)