"""
Nemori search client implementation extracted from wait_for_refactor.

This module contains the search functionality for Nemori in the LoCoMo evaluation.
"""

import asyncio
from pathlib import Path
from time import time

from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig


TEMPLATE_NEMORI = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""


async def get_nemori_client(user_id: str, version: str = "default", retrievalstrategy: RetrievalStrategy = RetrievalStrategy.EMBEDDING, emb_api_key="",emb_base_url="",embed_model=""):
    """Get Nemori client for search."""
    # Setup storage
    storage_dir = Path(f"results/locomo/nemori-{version}/storages")
    db_path = storage_dir / "nemori_memory.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Nemori database not found at {db_path}. Please run ingestion first.")
    
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )
    
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()
    
    # Setup retrieval
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key=emb_api_key,  # 与ingestion保持一致
        base_url=emb_base_url,  # 与ingestion保持一致
        embed_model=embed_model,  # 与ingestion保持一致
    )
    retrieval_service.register_provider(retrievalstrategy, retrieval_config)
    await retrieval_service.initialize()
    
    return retrieval_service


async def nemori_search(retrieval_service, query: str, speaker_a_user_id: str, speaker_b_user_id: str, top_k: int = 20, retrievalstrategy=RetrievalStrategy.EMBEDDING) -> tuple[str, float]:
    """Search using Nemori."""
    start = time()
    
    print(f"\n🔍 [NEMORI SEARCH] Starting search for query: '{query}'")
    print(f"   👤 Speaker A ID: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B ID: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")
    
    # Search for speaker A
    # From the perspective of episodic memory construction in the current MVP version, 
    # no specialized processing is done for any individual's memories, 
    # so searching any one person's memories is sufficient
    print(f"\n🔎 [SPEAKER A] Searching for owner_id: '{speaker_a_user_id}'")
    query_a = RetrievalQuery(text=query, owner_id=speaker_a_user_id, limit=top_k, strategy=retrievalstrategy)
    print(f"   📝 Query object: text='{query_a.text}', owner_id='{query_a.owner_id}', limit={query_a.limit}")

    result_a = await retrieval_service.search(query_a)
    print(f"   ✅ Search completed. Found {len(result_a.episodes)} episodes")
    
    if len(result_a.episodes) > 0:
        print("   📋 Sample episodes for speaker A:")
        for i, episode in enumerate(result_a.episodes[:2]):
            title = episode["title"]
            content = episode["content"][:100]
            summary = episode["summary"]
            print(f"     {i+1}. Title: '{title}'")
            print(f"        Content: '{content}...'")
            print(f"        Summary: '{summary}'")
    else:
        print("⚠️ No episodes found for speaker A")
    #print("result_a.episodes:",len(result_a.episodes))
    # Format results for speaker A
    speaker_memories = []
    for episode in result_a.episodes:
        title = episode["title"]
        content = episode["content"]
        memory_text = f"{title}: {content}"
        speaker_memories.append(memory_text)
    
    print(f"\n📊 [FORMATTING] Speaker memories: {len(speaker_memories)}")
    
    # Format context
    context = TEMPLATE_NEMORI.format(
        speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
        speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
        speaker_memories="\n".join(speaker_memories) if speaker_memories else "No relevant memories found",
    )
    
    print("\n📄 [CONTEXT] Generated context preview:")
    print(f"   {context[:200]}...")
    
    duration_ms = (time() - start) * 1000
    print(f"\n⏱️ [TIMING] Search completed in {duration_ms:.2f}ms")
    
    return context, duration_ms

