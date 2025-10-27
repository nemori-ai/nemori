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


from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy, RetrievalConfig

# Nemori evaluation imports
from nemori_eval.search import get_nemori_client, nemori_search

NEMORI_AVAILABLE = True


TEMPLATE_ZEP = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

TEMPLATE_MEM0 = """Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}
"""

TEMPLATE_MEM0_GRAPH = """Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Relations for user {speaker_1_user_id}:

    {speaker_1_graph_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}

    Relations for user {speaker_2_user_id}:

    {speaker_2_graph_memories}
"""

TEMPLATE_MEMOS = """Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}
"""


async def search_query_async(
    client, query, metadata, frame, reversed_client=None, top_k=20, retrievalstrategy=RetrievalStrategy.EMBEDDING
):
    """Async version of search_query for nemori."""
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    if frame == "nemori":
        context, duration_ms = await nemori_search(
            client, query, speaker_a_user_id, speaker_b_user_id, top_k, retrievalstrategy=retrievalstrategy
        )
        return context, duration_ms


# def load_existing_results(frame, version, group_idx):
#     result_path = (
#         f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
#     )
#     if os.path.exists(result_path):
#         try:
#             with open(result_path) as f:
#                 return json.load(f), True
#         except Exception as e:
#             print(f"Error loading existing results for group {group_idx}: {e}")
#     return {}, False


async def process_user_nemori(group_idx, locomo_df, frame, version, top_k=20):
    """Process user for Nemori framework."""
    if not NEMORI_AVAILABLE:
        raise ImportError("Nemori is not available. Please install nemori.")

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

    # existing_results, loaded = load_existing_results(frame, version, group_idx)
    # if loaded:
    #     print(f"Loaded existing results for group {group_idx}")
    #     return existing_results

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
        try:
            print(f"Processing user {idx}...")
            user_results = await process_user_nemori(idx, locomo_df, frame, version, top_k)
            for conv_id, results in user_results.items():
                all_search_results[conv_id].extend(results)
        except Exception as e:
            print(f"User {idx} generated an exception: {e}")

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


def main(frame, version="default", num_workers=1, top_k=20):
    load_dotenv()

    if frame == "nemori":
        if not NEMORI_AVAILABLE:
            print("❌ Nemori is not available. Please install nemori to use this framework.")
            return
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
    lib = args.lib
    version = args.version
    workers = args.workers
    top_k = args.top_k

    main(lib, version, workers, top_k)
