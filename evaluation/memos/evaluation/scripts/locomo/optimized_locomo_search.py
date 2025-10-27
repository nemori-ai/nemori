#!/usr/bin/env python3
"""
Optimized locomo_search.py for testing with fewer users and proper result saving.
"""

import argparse
import asyncio
import json
import os

from collections import defaultdict
from time import time

import pandas as pd

from dotenv import load_dotenv

NEMORI_AVAILABLE = True

# Nemori evaluation imports
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
from nemori.retrieval import RetrievalStrategy

# Global cache for user mapping to avoid repeated database connections
_user_mapping_cache = {}

def get_database_user_mapping(db_path: str) -> dict:
    """Get mapping from conversation speakers to database user IDs."""
    global _user_mapping_cache
    
    # Use cache if available
    if db_path in _user_mapping_cache:
        print(f"📋 Using cached user mapping: {len(_user_mapping_cache[db_path])} entries")
        return _user_mapping_cache[db_path]
    
    import duckdb
    
    try:
        # Use read-only connection to avoid conflicts
        conn = duckdb.connect(db_path, read_only=True)
        
        # Get all unique owner_ids from episodes
        episode_users = conn.execute("SELECT DISTINCT owner_id FROM episodes").fetchall()
        episode_users = [user[0] for user in episode_users]
        
        # Get all unique owner_ids from semantic nodes 
        semantic_users = conn.execute("SELECT DISTINCT owner_id FROM semantic_nodes").fetchall()
        semantic_users = [user[0] for user in semantic_users]
        
        # Combine and create mapping
        all_users = set(episode_users + semantic_users)
        
        mapping = {}
        for db_user in all_users:
            # Extract the base name (e.g., "jon_1" -> "jon")
            base_name = db_user.split('_')[0]
            # Create mapping for both cases
            mapping[base_name.lower()] = db_user
            mapping[base_name.capitalize()] = db_user
            mapping[base_name.upper()] = db_user
        
        conn.close()
        
        # Cache the mapping
        _user_mapping_cache[db_path] = mapping
        
        print(f"📋 User mapping created: {len(mapping)} entries")
        return mapping
        
    except Exception as e:
        print(f"⚠️ Failed to create user mapping: {e}")
        return {}


# Template for unified memory search results
TEMPLATE_NEMORI_UNIFIED = """Unified memories for conversation between {speaker_1} and {speaker_2}:

Episodic Memories:
{episodic_memories}

Semantic Knowledge:
{semantic_knowledge}
"""

async def search_query_async(client, query, metadata, frame, reversed_client=None, top_k=10):
    """Async version of search_query for nemori with unified memory support."""
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    if not NEMORI_AVAILABLE:
        raise ImportError("Nemori is not available. Please install nemori.")
    
    # Use unified search
    context, duration_ms = await nemori_unified_search(
        unified_retrieval_service=client[0],
        retrieval_service=client[1], 
        query=query,
        speaker_a_user_id=speaker_a_user_id,
        speaker_b_user_id=speaker_b_user_id,
        top_k=top_k
    )
    
    return context, duration_ms


def load_existing_results(frame, version, group_idx):
    result_path = f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"Error loading existing results for group {group_idx}: {e}")
    return {}, False


async def process_user_nemori(group_idx, locomo_df, frame, version, top_k=20, max_questions=5):
    """Process user for Nemori framework with unified memory support."""
    if not NEMORI_AVAILABLE:
        raise ImportError("Nemori is not available. Please install nemori.")

    print(f"\n🚀 [NEMORI PROCESS] Starting processing for user {group_idx}")
    print(f"   🎯 Version: {version}")

    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[group_idx]
    conversation = locomo_df["conversation"].iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    
    # Get database path for user mapping
    from pathlib import Path
    
    # Handle special case for episode_semantic version
    if version == "episode_semantic":
        # Use the absolute path we know exists
        db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
        if not Path(db_path).exists():
            print(f"⚠️ Episode semantic database not found at {db_path}")
            db_path = None
    else:
        # Original logic for other versions
        storage_dir = Path(f"results/locomo/nemori-{version}/storages")
        
        # Try different database file names based on version
        possible_db_names = [
            "nemori_memory.duckdb",         # Standard database name from ingestion
            "nemori_full_semantic.duckdb",  # Full semantic database name
            f"nemori_{version}.duckdb",     # Version-specific name
            "nemori.duckdb",                # Generic name
        ]
        
        db_path = None
        for db_name in possible_db_names:
            potential_path = storage_dir / db_name
            if potential_path.exists():
                db_path = str(potential_path)
                break
    
    # Get user mapping from database
    user_mapping = {}
    if db_path:
        user_mapping = get_database_user_mapping(db_path)
    
    # Map speakers to database user IDs
    speaker_a_user_id = user_mapping.get(speaker_a, speaker_a)
    speaker_b_user_id = user_mapping.get(speaker_b, speaker_b)
    
    conv_id = f"locomo_exp_user_{group_idx}"

    print(f"   👥 Original speakers: '{speaker_a}' & '{speaker_b}'")
    print(f"   🆔 Database IDs: '{speaker_a_user_id}' & '{speaker_b_user_id}'")
    print(f"   📝 Conversation ID: '{conv_id}'")
    print(f"   ❓ QA set size: {len(qa_set)} (processing first {max_questions})")

    existing_results, loaded = load_existing_results(frame, version, group_idx)
    if loaded:
        print(f"Loaded existing results for group {group_idx}")
        return existing_results

    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": group_idx,
        "conv_id": conv_id,
        "version": version,  # Add version to metadata
    }
    
    # Get appropriate client based on version
    print(f"   🧠 Using unified semantic search client for version: {version}")
    # Use unified client for semantic versions
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        user_id=conv_id, 
        version=version,
        retrievalstrategy=RetrievalStrategy.EMBEDDING,  # 默认使用embedding搜索
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1", 
        embed_model="qwen3-emb"
    )
    client = (unified_retrieval, retrieval_service, episode_repo, semantic_repo)

    async def process_qa(qa):
        query = qa.get("question")
        if qa.get("category") == 5:
            return None

        context, duration_ms = await search_query_async(client, query, metadata, frame, top_k=top_k)

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms}

    # Process limited QAs for testing
    test_qa_set = qa_set[:max_questions]
    
    # Process QAs one by one to avoid overwhelming the system
    for qa_idx, qa in enumerate(test_qa_set):
        try:
            result = await process_qa(qa)
            if result:
                context_preview = result["context"][:50] + "..." if result["context"] else "No context"
                print(f"   🔍 Query {qa_idx+1}: {result['query'][:30]}...")
                print(f"   📄 Context: {context_preview}")
                print(f"   ⏱️ Duration: {result['duration_ms']:.2f}ms")
                search_results[conv_id].append(result)
        except Exception as e:
            print(f"   ⚠️ Error processing query {qa_idx+1}: {qa.get('question', 'Unknown')[:30]}... - {e}")

    # client is now a tuple of (unified_retrieval, retrieval_service, episode_repo, semantic_repo)
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = client
    
    if hasattr(unified_retrieval, 'close'):
        await unified_retrieval.close()
    if retrieval_service:
        await retrieval_service.close()
    if episode_repo:
        await episode_repo.close() 
    if semantic_repo:
        await semantic_repo.close()

    # Save individual user results
    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    result_file = f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    with open(result_file, "w") as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"   💾 Saved search results to {result_file}")

    return search_results


async def main_nemori(version="default", top_k=20, max_users=3, max_questions=5):
    """Main function for Nemori search with limited processing for testing."""
    load_dotenv()
    
    # Fix the data path to use the correct location
    data_path = "../data/locomo/locomo10.json"
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found. Tried: {data_path}")
        return
        
    locomo_df = pd.read_json(data_path)
    print(f"✅ Loaded {len(locomo_df)} conversations from {data_path}")
    print(f"📊 Processing first {max_users} users with {max_questions} questions each")

    frame = "nemori"
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    # Process limited users sequentially for testing
    for idx in range(min(max_users, len(locomo_df))):
        try:
            print(f"\n{'='*60}")
            print(f"Processing user {idx}...")
            user_results = await process_user_nemori(idx, locomo_df, frame, version, top_k, max_questions)
            
            if user_results:
                for conv_id, results in user_results.items():
                    all_search_results[conv_id].extend(results)
                    
        except Exception as e:
            print(f"❌ User {idx} generated an exception: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    final_result_file = f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json"
    with open(final_result_file, "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print(f"\n💾 Saved all search results to {final_result_file}")


def main(version="default", top_k=10, max_users=3, max_questions=5):
    load_dotenv()

    # Run async main for nemori
    asyncio.run(main_nemori(version, top_k, max_users, max_questions))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["nemori"],
        default="nemori",
        help="Specify the memory framework"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="episode_semantic",
        help="Version identifier for saving results"
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve in search queries")
    parser.add_argument("--max_users", type=int, default=3, help="Maximum number of users to process (for testing)")
    parser.add_argument("--max_questions", type=int, default=5, help="Maximum questions per user (for testing)")
    
    args = parser.parse_args()
    
    main(args.version, args.top_k, args.max_users, args.max_questions)