#!/usr/bin/env python3
"""
Test script for unified episodic and semantic memory evaluation.

This script tests the complete pipeline following the locomo_ingestion.py pattern
but integrating both episodic and semantic memory systems.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
import pandas as pd


async def test_unified_memory_system():
    """Test the unified episodic and semantic memory system."""
    print("🧪 Testing Unified Episodic + Semantic Memory System")
    print("=" * 60)
    
    # Create a simple test conversation dataset
    test_conversations = [
        {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob", 
                "session_0": [
                    {
                        "speaker": "Alice",
                        "text": "Hi Bob! I've been learning machine learning recently, focusing on neural networks and deep learning.",
                        "timestamp": "2024-01-15T10:00:00Z"
                    },
                    {
                        "speaker": "Bob",
                        "text": "That's awesome Alice! I'm also interested in AI. I've been working with PyTorch for computer vision projects.",
                        "timestamp": "2024-01-15T10:01:00Z"
                    },
                    {
                        "speaker": "Alice", 
                        "text": "Great! I prefer PyTorch too. Are you familiar with transformer architectures like BERT and GPT?",
                        "timestamp": "2024-01-15T10:02:00Z"
                    },
                    {
                        "speaker": "Bob",
                        "text": "Yes! I've implemented both BERT and GPT models. Currently exploring multimodal transformers for vision-language tasks.",
                        "timestamp": "2024-01-15T10:03:00Z"
                    }
                ],
                "session_0_date_time": "10:00 AM on 15 January, 2024"
            }
        }
    ]
    
    # Convert to DataFrame format expected by NemoriExperiment
    test_df = pd.DataFrame(test_conversations)
    
    print(f"📊 Created test dataset with {len(test_conversations)} conversations")
    
    # Initialize experiment
    experiment = NemoriExperiment(
        version="test_unified",
        episode_mode="speaker",
        retrievalstrategy=RetrievalStrategy.EMBEDDING
    )
    
    try:
        print("\n🤖 Step 1: Setting up LLM Provider")
        print("-" * 40)
        
        # Use test API credentials
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"
        model = "gpt-4o-mini"
        
        llm_available = await experiment.setup_llm_provider(
            model=model, 
            api_key=api_key, 
            base_url=base_url
        )
        
        if not llm_available:
            print("❌ LLM setup failed - continuing with limitations")
        else:
            print("✅ LLM provider configured successfully")
            
        print("\n🗄️ Step 2: Loading Test Data") 
        print("-" * 40)
        
        experiment.load_locomo_data(test_df)
        print(f"✅ Loaded {len(experiment.conversations)} test conversations")
        
        print("\n🔧 Step 3: Setting up Unified Storage")
        print("-" * 40)
        
        # Setup with embedding support
        await experiment.setup_storage_and_retrieval(
            emb_api_key=api_key,
            emb_base_url=base_url,
            embed_model="text-embedding-ada-002"
        )
        print("✅ Unified storage (episodic + semantic) initialized")
        
        print("\n🏗️ Step 4: Building Episodes with Semantic Discovery")
        print("-" * 40)
        
        await experiment.build_episodes()
        print(f"✅ Created {len(experiment.episodes)} episodes")
        
        # Display episode details
        if experiment.episodes:
            print("\n📖 Episode Details:")
            for i, episode in enumerate(experiment.episodes):
                print(f"  {i+1}. {episode.title} (Owner: {episode.owner_id})")
                
        print("\n🧠 Step 5: Checking Semantic Knowledge Discovery")
        print("-" * 40)
        
        # Get all unique owners
        if experiment.episodes:
            owners = set(ep.owner_id for ep in experiment.episodes)
            print(f"👥 Found {len(owners)} owners: {list(owners)}")
            
            total_semantic_nodes = 0
            for owner in owners:
                try:
                    nodes = await experiment.semantic_repo.get_all_semantic_nodes_for_owner(owner)
                    if nodes:
                        total_semantic_nodes += len(nodes)
                        print(f"  🔹 {owner}: {len(nodes)} semantic concepts")
                        
                        # Show first few concepts
                        for node in nodes[:3]:  # Show first 3
                            print(f"    • {node.key}: {node.value} (confidence: {node.confidence:.2f})")
                    else:
                        print(f"  ⚪ {owner}: No semantic concepts discovered")
                        
                except Exception as e:
                    print(f"  ❌ {owner}: Error accessing semantic memory - {e}")
                    
            print(f"🎯 Total semantic concepts discovered: {total_semantic_nodes}")
            
        print("\n🔍 Step 6: Testing Unified Retrieval")
        print("-" * 40)
        
        if experiment.unified_retrieval_service and owners:
            test_owner = next(iter(owners))
            test_queries = ["machine learning", "PyTorch", "transformers"]
            
            print(f"🎯 Testing queries with owner: {test_owner}")
            
            for query in test_queries:
                print(f"\n  🔎 Query: '{query}'")
                
                try:
                    # Test unified retrieval
                    results = await experiment.unified_retrieval_service.enhanced_query(
                        test_owner, query, episode_limit=2, semantic_limit=2
                    )
                    
                    episodes = results.get('episodes', [])
                    semantic_knowledge = results.get('semantic_knowledge', [])
                    
                    print(f"    📚 Found {len(episodes)} relevant episodes")
                    print(f"    🧠 Found {len(semantic_knowledge)} semantic concepts")
                    
                    if episodes or semantic_knowledge:
                        print(f"    ✅ Unified retrieval working!")
                    else:
                        print(f"    ⚪ No results (may need more data)")
                        
                except Exception as e:
                    print(f"    ❌ Query failed: {e}")
        
        print("\n🎉 Test Results Summary")
        print("=" * 60)
        
        success_indicators = []
        
        # Check episodic memory
        if experiment.episodes:
            success_indicators.append("✅ Episodic memory: Episodes created successfully")
        else:
            success_indicators.append("❌ Episodic memory: No episodes created")
            
        # Check semantic memory
        if total_semantic_nodes > 0:
            success_indicators.append("✅ Semantic memory: Knowledge concepts discovered")
        else:
            success_indicators.append("❌ Semantic memory: No concepts discovered")
            
        # Check LLM integration
        if llm_available:
            success_indicators.append("✅ LLM integration: Working correctly")
        else:
            success_indicators.append("❌ LLM integration: Failed to connect")
            
        # Check unified retrieval
        if experiment.unified_retrieval_service:
            success_indicators.append("✅ Unified retrieval: Service initialized")
        else:
            success_indicators.append("❌ Unified retrieval: Service not available")
            
        for indicator in success_indicators:
            print(indicator)
            
        # Overall assessment
        successful_components = sum(1 for ind in success_indicators if ind.startswith("✅"))
        total_components = len(success_indicators)
        
        print(f"\n🎯 Overall Success Rate: {successful_components}/{total_components} components working")
        
        if successful_components == total_components:
            print("🌟 All systems working perfectly! Ready for LoCoMo benchmark.")
        elif successful_components >= total_components * 0.75:
            print("🟡 Most systems working. May proceed with benchmark testing.")
        else:
            print("🔴 Multiple issues detected. Please check configuration.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test resources")
        await experiment.cleanup()
        print("✅ Cleanup completed")


if __name__ == "__main__":
    print("🚀 Starting Unified Memory System Test")
    print("=" * 60)
    
    # Run the test
    asyncio.run(test_unified_memory_system())
    
    print("\n✨ Test completed!")