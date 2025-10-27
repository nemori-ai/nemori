#!/usr/bin/env python3
"""
Simplified test script to verify locomo_search.py functionality.
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


async def test_search():
    """Test search functionality with a few queries."""
    # Database path
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    # Get user mapping
    user_mapping = get_database_user_mapping(db_path)
    
    # Load test data
    data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    locomo_df = pd.read_json(data_path)
    
    # Test with first conversation
    conversation = locomo_df["conversation"].iloc[0]
    qa_set = locomo_df["qa"].iloc[0]
    
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_db = user_mapping.get(speaker_a, speaker_a)
    speaker_b_db = user_mapping.get(speaker_b, speaker_b)
    
    print(f"\n🗣️ Testing conversation: {speaker_a} & {speaker_b}")
    print(f"🆔 Database IDs: {speaker_a_db} & {speaker_b_db}")
    
    # Get unified client
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        user_id="test",
        version="episode_semantic",
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb"
    )
    
    # Test with first 3 questions
    for i, qa in enumerate(qa_set[:3]):
        if qa.get("category") == 5:
            continue
            
        query = qa.get("question", "")
        if not query:
            continue
        
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
            print(f"📄 Context preview:")
            print(context[:300] + "..." if len(context) > 300 else context)
            
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
    asyncio.run(test_search())