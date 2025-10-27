#!/usr/bin/env python3
"""
Test user 3 (Joanna & Nate) specifically with the fixed database path.
"""

import asyncio
import pandas as pd
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
from nemori.retrieval import RetrievalStrategy

# Global cache for user mapping
_user_mapping_cache = {}

def get_database_user_mapping(db_path: str) -> dict:
    """Get mapping from conversation speakers to database user IDs."""
    global _user_mapping_cache
    
    if db_path in _user_mapping_cache:
        return _user_mapping_cache[db_path]
    
    import duckdb
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        episode_users = conn.execute("SELECT DISTINCT owner_id FROM episodes").fetchall()
        episode_users = [user[0] for user in episode_users]
        
        semantic_users = conn.execute("SELECT DISTINCT owner_id FROM semantic_nodes").fetchall()
        semantic_users = [user[0] for user in semantic_users]
        
        all_users = set(episode_users + semantic_users)
        
        mapping = {}
        for db_user in all_users:
            base_name = db_user.split('_')[0]
            mapping[base_name.lower()] = db_user
            mapping[base_name.capitalize()] = db_user
            mapping[base_name.upper()] = db_user
        
        conn.close()
        _user_mapping_cache[db_path] = mapping
        
        print(f"📋 User mapping created: {len(mapping)} entries")
        return mapping
        
    except Exception as e:
        print(f"⚠️ Failed to create user mapping: {e}")
        return {}


async def test_user_3():
    """Test search functionality for user 3 (Joanna & Nate)."""
    # Load test data
    data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    locomo_df = pd.read_json(data_path)
    
    # Test with user 3 (Joanna & Nate)
    conversation = locomo_df["conversation"].iloc[3]
    qa_set = locomo_df["qa"].iloc[3]
    
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    
    print(f"🗣️ Testing user 3: {speaker_a} & {speaker_b}")
    print(f"❓ Total questions: {len(qa_set)}")
    
    # Get unified client with fixed database path
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        user_id="test_user_3",
        version="episode_semantic",  # This will now use the correct database
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb"
    )
    
    # The user mapping should now work with the correct database
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-default/storages/nemori_memory.duckdb"
    user_mapping = get_database_user_mapping(db_path)
    
    speaker_a_db = user_mapping.get(speaker_a, speaker_a)
    speaker_b_db = user_mapping.get(speaker_b, speaker_b)
    
    print(f"🆔 Database IDs: {speaker_a_db} & {speaker_b_db}")
    
    # Test with first 3 questions from user 3
    test_questions = [
        "Is it likely that Nate has friends besides Joanna?",
        "What kind of interests do Joanna and Nate share?", 
        "When did Joanna first watch \"Eternal Sunshine of the Spotless Mind?\""
    ]
    
    for i, query in enumerate(test_questions):
        print(f"\n💬 Question {i+1}: {query}")
        
        try:
            context, duration = await nemori_unified_search(
                unified_retrieval_service=unified_retrieval,
                retrieval_service=retrieval_service,
                query=query,
                speaker_a_user_id=speaker_a_db,
                speaker_b_user_id=speaker_b_db,
                top_k=3
            )
            
            print(f"✅ Search completed in {duration:.2f}ms")
            
            # Check if we actually found content
            if "No relevant" in context or len(context.strip()) < 100:
                print(f"⚠️ Limited content found")
                print(f"📄 Context: {context[:200]}...")
            else:
                print(f"✅ Rich content found!")
                print(f"📄 Context preview: {context[:300]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    if retrieval_service:
        await retrieval_service.close()
    if episode_repo:
        await episode_repo.close()
    if semantic_repo:
        await semantic_repo.close()


if __name__ == "__main__":
    asyncio.run(test_user_3())