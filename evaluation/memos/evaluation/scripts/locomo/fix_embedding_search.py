#!/usr/bin/env python3
"""Fix embedding provider to properly load episodes from storage."""

import asyncio
import json
from pathlib import Path
from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig


async def fix_embedding_search():
    print("🔧 Fixing Embedding Search for evan_8")
    print("=" * 50)

    # Setup storage
    storage_dir = Path("results/locomo/nemori-default/storages")
    db_path = storage_dir / "nemori_memory.duckdb"

    # Setup storage config
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )

    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()

    # Setup retrieval service
    retrieval_service = RetrievalService(episode_repo)

    # Setup embedding provider
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key="EMPTY",
        base_url="http://localhost:6003/v1",
        embed_model="bce-emb",
    )

    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()

    print("✅ Services initialized")

    # Get the embedding provider directly
    embedding_provider = retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)

    print("🔧 Manually fixing evan_8 index...")

    # Get evan_8 episodes from database
    result = await episode_repo.get_episodes_by_owner("evan_8")
    episodes = result.episodes if hasattr(result, "episodes") else result
    print(f"📊 Found {len(episodes)} episodes in database for evan_8")

    # Get the user index
    user_index = embedding_provider._get_user_index("evan_8")
    print(f"🔍 Current index state: {len(user_index['embeddings'])} embeddings, {len(user_index['episodes'])} episodes")

    # Fix: Populate the episodes array with actual episode objects
    if len(user_index["embeddings"]) > 0 and len(user_index["episodes"]) == 0:
        print("🔧 Populating episodes array...")

        # Map episode IDs to episodes
        episode_map = {ep.id: ep for ep in episodes}

        # Populate episodes in the same order as embeddings
        for episode_id, index_pos in user_index["episode_id_to_index"].items():
            if episode_id in episode_map and index_pos < len(user_index["embeddings"]):
                # Ensure episodes array is big enough
                while len(user_index["episodes"]) <= index_pos:
                    user_index["episodes"].append(None)
                user_index["episodes"][index_pos] = episode_map[episode_id]

        # Remove None entries
        user_index["episodes"] = [ep for ep in user_index["episodes"] if ep is not None]

        print(f"✅ Fixed! Now have {len(user_index['episodes'])} episodes in index")

    # Test search
    print("🔎 Testing search...")
    search_query = RetrievalQuery(
        text="How does Evan describe being out on the water while kayaking and watching the sunset?",
        owner_id="evan_8",
        limit=5,
        strategy=RetrievalStrategy.EMBEDDING,
    )

    try:
        result = await retrieval_service.search(search_query)
        print(f"🎯 Search result: {len(result.episodes)} episodes found")

        for i, episode in enumerate(result.episodes[:3]):
            score = result.scores[i] if result.scores and i < len(result.scores) else "N/A"
            print(f"   {i+1}. '{episode.title}' (score: {score})")
            print(f"      {episode.content[:100]}...")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    await retrieval_service.close()
    await episode_repo.close()
    print("\n🏁 Complete")


if __name__ == "__main__":
    asyncio.run(fix_embedding_search())
