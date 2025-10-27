#!/usr/bin/env python3
"""
Debug John & Maria conversation data issue.
"""

import asyncio
import os
from pathlib import Path
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
from nemori.retrieval import RetrievalStrategy

async def debug_john_maria():
    """Debug why John & Maria conversation returns no data."""
    
    print("🔍 Debugging John & Maria data issue...")
    
    # Use the correct database path 
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-default/storages/nemori_memory.duckdb"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Using database: {db_path}")
    
    # Get unified client
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        user_id="test_user_2",
        version="episode_semantic",  # This should use default db
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb"
    )
    
    # Check what user IDs exist in the database
    print("\n📊 Checking user IDs in database...")
    
    try:
        # Query all owner_ids from episodic memories
        if episode_repo and hasattr(episode_repo, 'get_all_episodes'):
            episodes = await episode_repo.get_all_episodes()
            owner_ids = set()
            for episode in episodes:
                if hasattr(episode, 'owner_id'):
                    owner_ids.add(episode.owner_id)
            
            print(f"📋 Found {len(owner_ids)} unique owner IDs:")
            for owner_id in sorted(owner_ids):
                print(f"   - {owner_id}")
                
            # Check specifically for john and maria related IDs
            john_ids = [oid for oid in owner_ids if 'john' in oid.lower()]
            maria_ids = [oid for oid in owner_ids if 'maria' in oid.lower()]
            
            print(f"\n🔍 John-related IDs: {john_ids}")
            print(f"🔍 Maria-related IDs: {maria_ids}")
            
        else:
            print("❌ Cannot access episode repository")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
    
    # Test searches with different user ID combinations
    test_combinations = [
        ("john_2", "maria_2"),   # Current mapping
        ("John", "Maria"),       # Original names
        ("locomo_exp_user_2", "locomo_exp_user_2"),  # Conversation ID
    ]
    
    test_query = "Who did Maria have dinner with on May 3, 2023?"
    
    for speaker_a_id, speaker_b_id in test_combinations:
        print(f"\n🧪 Testing with IDs: '{speaker_a_id}' & '{speaker_b_id}'")
        
        try:
            context, duration = await nemori_unified_search(
                unified_retrieval_service=unified_retrieval,
                retrieval_service=retrieval_service,
                query=test_query,
                speaker_a_user_id=speaker_a_id,
                speaker_b_user_id=speaker_b_id,
                top_k=5
            )
            
            print(f"⏱️ Duration: {duration:.2f}ms")
            
            if "No relevant" in context or len(context.strip()) < 100:
                print(f"❌ No content found")
                print(f"📝 Context: {context[:100]}...")
            else:
                print(f"✅ Content found!")
                print(f"📝 Context preview: {context[:200]}...")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # Cleanup
    if retrieval_service:
        await retrieval_service.close()
    if episode_repo:
        await episode_repo.close()
    if semantic_repo:
        await semantic_repo.close()

if __name__ == "__main__":
    asyncio.run(debug_john_maria())