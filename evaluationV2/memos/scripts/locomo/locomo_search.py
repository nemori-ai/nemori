import argparse
import asyncio
import json
import os

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import pandas as pd

from dotenv import load_dotenv
from mem0 import MemoryClient
from utils import filter_memory_data
from zep_cloud.client import Zep

MEMOS_AVAILABLE = True
NEMORI_AVAILABLE = True

# Nemori evaluation imports

from nemori_eval.search import get_nemori_unified_client, nemori_unified_search, NemoriUnifiedSearchClient
from nemori.retrieval import RetrievalStrategy

# Global cache for user mapping to avoid repeated database connections (not used anymore)
_user_mapping_cache = {}

# Static user mapping function - no longer used with dynamic ID generation
# def get_static_user_mapping() -> dict:
#     """Get static user mapping to avoid database connection conflicts."""
#     # Static mapping based on known database users
#     ...
# (Function commented out since we now use dynamic ID generation)



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
    
    # Check if this is a unified semantic version
    version = metadata.get("version", "default")
    
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


async def process_user_nemori(group_idx, locomo_df, frame, version, top_k=20):
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
        # The episode_semantic database is empty, use the default database which has data
        db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-default/storages/nemori_memory.duckdb"
        if not Path(db_path).exists():
            print(f"⚠️ Default database not found at {db_path}")
            db_path = None
        else:
            print(f"🔄 Using default database with actual data: {db_path}")
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
    
    # Generate dynamic user IDs like in locomo_search_emb2.py
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{group_idx}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{group_idx}"
    
    conv_id = f"locomo_exp_user_{group_idx}"

    print(f"   👥 Original speakers: '{speaker_a}' & '{speaker_b}'")
    print(f"   🆔 Generated IDs: '{speaker_a_user_id}' & '{speaker_b_user_id}'")
    print(f"   📝 Conversation ID: '{conv_id}'")
    print(f"   ❓ QA set size: {len(qa_set)}")

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

    # Process QAs concurrently with asyncio.gather
    semaphore = asyncio.Semaphore(10)  # Limit concurrent operations to 10
    
    async def process_qa_with_semaphore(qa):
        async with semaphore:
            return await process_qa(qa)
    
    # Create tasks for all QAs
    tasks = [process_qa_with_semaphore(qa) for qa in qa_set]
    
    # Process all QAs concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for qa, result in zip(qa_set, results):
        if isinstance(result, Exception):
            print(f"   ⚠️ Error processing query: {qa.get('question', 'Unknown')[:30]}... - {result}")
            continue
        if result:
            context_preview = result["context"][:50] + "..." if result["context"] else "No context"
            print(f"   🔍 Query: {result['query'][:30]}...")
            print(f"   📄 Context: {context_preview}")
            print(f"   ⏱️ Duration: {result['duration_ms']:.2f}ms")
            search_results[conv_id].append(result)

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


    # Save individual user results immediately after processing
    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    result_file = f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    with open(result_file, "w") as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"   💾 Saved search results to {result_file}")

    return search_results


async def main_nemori(version="default", top_k=20):
    """Main function for Nemori search with concurrent user processing."""
    load_dotenv()
    
    # Fix the data path to use the correct location
    data_path = "../data/locomo/locomo10.json"
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = "data/locomo/locomo10.json"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found. Tried: {data_path}")
        return
        
    locomo_df = pd.read_json(data_path)
    print(f"✅ Loaded {len(locomo_df)} conversations from {data_path}")

    num_conv = 10
    frame = "nemori"
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    # Create semaphore for concurrent user processing
    user_semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent users
    
    async def process_user_with_semaphore(idx):
        async with user_semaphore:
            try:
                print(f"Processing user {idx}...")
                user_results = await process_user_nemori(idx, locomo_df, frame, version, top_k)
                return idx, user_results, None
            except Exception as e:
                print(f"User {idx} generated an exception: {e}")
                return idx, None, e
    
    # Create tasks for all users
    user_tasks = [process_user_with_semaphore(idx) for idx in range(num_conv)]
    
    # Process all users concurrently
    user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
    
    # Collect results
    for result in user_results:
        if isinstance(result, Exception):
            print(f"Unexpected error: {result}")
            continue
        idx, user_result, error = result
        if error is None and user_result:
            for conv_id, results in user_result.items():
                all_search_results[conv_id].extend(results)

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


def main(version="default", top_k=10):
    load_dotenv()

    # Run async main for nemori
    asyncio.run(main_nemori(version, top_k))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "langmem", "nemori"],
        help="Specify the memory framework (zep or memos or mem0 or mem0_graph or nemori)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers to process users")
    parser.add_argument("--top_k", type=int, default=20, help="Number of results to retrieve in search queries")
    args = parser.parse_args()
    version = args.version
    top_k = args.top_k

    main(version, top_k)
