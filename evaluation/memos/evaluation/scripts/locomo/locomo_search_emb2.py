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
from tqdm import tqdm
from utils import filter_memory_data
from zep_cloud.client import Zep

from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy, RetrievalConfig

# Nemori evaluation imports
from nemori_eval import get_nemori_client, nemori_search

NEMORI_AVAILABLE = True


async def search_query_async(client, query, metadata, frame, top_k=20, retrievalstrategy=RetrievalStrategy.EMBEDDING):
    """Async version of search_query for nemori."""
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    if frame == "nemori":
        context, duration_ms = await nemori_search(
            client, query, speaker_a_user_id, speaker_b_user_id, top_k, retrievalstrategy=retrievalstrategy
        )
        return context, duration_ms


async def process_user_nemori(group_idx, locomo_df, frame, version, top_k=20):
    """Process user for Nemori framework."""
    print(f"\n🚀 [NEMORI PROCESS] Starting processing for user {group_idx}")

    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[group_idx]
    conversation = locomo_df["conversation"].iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{group_idx}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{group_idx}"
    conv_id = f"locomo_exp_user_{group_idx}"

    print(f"   👥 Original speakers: '{speaker_a}' & '{speaker_b}'")
    print(f"   🆔 Generated IDs: '{speaker_a_user_id}' & '{speaker_b_user_id}'")
    print(f"   📝 Conversation ID: '{conv_id}'")
    print(f"   ❓ QA set size: {len(qa_set)}")

    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": group_idx,
        "conv_id": conv_id,
    }
    retrievalstrategy = RetrievalStrategy.EMBEDDING
    api_key = "EMPTY"
    base_url = "http://localhost:6007/v1"
    model = "qwen3-emb"
    # Get nemori client
    client = await get_nemori_client(
        conv_id,
        version,
        retrievalstrategy=retrievalstrategy,
        emb_api_key=api_key,
        emb_base_url=base_url,
        embed_model=model,
    )

    async def process_qa(qa):
        query = qa.get("question")
        if qa.get("category") == 5:
            return None

        context, duration_ms = await search_query_async(
            client, query, metadata, frame, top_k=top_k, retrievalstrategy=RetrievalStrategy.EMBEDDING
        )

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms}

    # Process QAs sequentially for nemori (since it's async)
    for qa in tqdm(qa_set, desc=f"Processing user {group_idx}"):
        result = await process_qa(qa)
        if result:
            context_preview = result["context"][:20] + "..." if result["context"] else "No context"
            print(
                {
                    "query": result["query"],
                    "context": context_preview,
                    "duration_ms": result["duration_ms"],
                }
            )
            search_results[conv_id].append(result)

    # Cleanup
    try:
        await client.close()
    except Exception as e:
        print(f"Warning: Error closing client: {e}")

    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    with open(f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json", "w") as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Save search results {group_idx}")

    return search_results


async def main_nemori(version="default", top_k=20):
    """Main function for Nemori search."""
    load_dotenv()
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    num_conv = 10
    frame = "nemori"
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    for idx in range(num_conv):
        print(f"Processing user {idx}...")
        user_results = await process_user_nemori(idx, locomo_df, frame, version, top_k)
        for conv_id, results in user_results.items():
            all_search_results[conv_id].extend(results)

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


def main(version="default", top_k=20):
    load_dotenv()

    asyncio.run(main_nemori(version, top_k))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Number of results to retrieve in search queries")
    args = parser.parse_args()
    version = args.version
    top_k = args.top_k

    main(version, top_k)
