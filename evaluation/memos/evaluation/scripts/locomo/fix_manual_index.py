#!/usr/bin/env python3
"""Fix embedding index by manually building it with correct episode_id usage."""

import asyncio
import json
from datetime import datetime
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


async def fix_embedding_index_manually():
    print("🔧 Manually Fix Embedding Index for evan_8")
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

    print("✅ Episode repository initialized")

    # Get evan_8 episodes
    result = await episode_repo.get_episodes_by_owner("evan_8")
    episodes = result.episodes if hasattr(result, "episodes") else result
    print(f"📊 Found {len(episodes)} episodes for evan_8")

    if not episodes:
        print("❌ No episodes found!")
        await episode_repo.close()
        return

    # Setup retrieval service to get embedding provider
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key="EMPTY",
        base_url="http://localhost:6003/v1",
        embed_model="bce-emb",
    )

    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()

    # Get embedding provider
    embedding_provider = retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)
    print("✅ Embedding provider initialized")

    # Manually build embedding index
    print("🔄 Manually building embedding index...")

    embeddings = []
    episode_objects = []
    episode_id_to_index = {}

    for i, episode in enumerate(episodes):
        print(f"   Processing episode {i+1}/{len(episodes)}: {episode.title[:50]}...")

        # Build searchable text using the provider's method
        try:
            searchable_text = embedding_provider._build_searchable_text(episode)
            print(f"      Searchable text: {searchable_text[:100]}...")

            # Generate embedding
            embedding = embedding_provider._generate_embedding(searchable_text)

            if embedding and not all(x == 0 for x in embedding):
                embeddings.append(embedding)
                episode_objects.append(episode)
                # Use episode.episode_id instead of episode.id
                episode_id_to_index[episode.episode_id] = len(embeddings) - 1
                print(f"      ✅ Generated embedding (dim: {len(embedding)})")
            else:
                print(f"      ❌ Failed to generate embedding")

        except Exception as e:
            print(f"      ❌ Error: {e}")
            continue

    print(f"🎯 Successfully built {len(embeddings)} embeddings")

    if not embeddings:
        print("❌ No embeddings were generated!")
        await retrieval_service.close()
        await episode_repo.close()
        return

    # Create index file manually
    index_data = {
        "embeddings": embeddings,
        "episode_id_to_index": episode_id_to_index,
        "last_updated": datetime.now().isoformat(),
        "embedding_dimension": len(embeddings[0]) if embeddings else 768,
        "episodes": [],  # This will be populated by the provider when loading
    }

    # Save index file
    index_file = storage_dir / "embedding_index_evan_8.json"
    print(f"💾 Saving index to {index_file}")

    with open(index_file, "w") as f:
        json.dump(index_data, f)

    print(f"✅ Index saved: {index_file.stat().st_size} bytes")

    # Test by reloading the provider
    print("🔄 Testing by reinitializing provider...")
    await retrieval_service.close()

    # Create new retrieval service
    new_retrieval_service = RetrievalService(episode_repo)
    new_retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await new_retrieval_service.initialize()

    # Get new provider and check if it loaded the index
    new_provider = new_retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)
    user_index = new_provider._get_user_index("evan_8")

    # Manually populate episodes array to fix the bug
    print("🔧 Fixing episodes array in loaded index...")
    if len(user_index["embeddings"]) > 0 and len(user_index["episodes"]) == 0:
        episode_map = {ep.episode_id: ep for ep in episodes}

        # Populate episodes in correct order
        user_index["episodes"] = [None] * len(user_index["embeddings"])
        for episode_id, index_pos in user_index["episode_id_to_index"].items():
            if episode_id in episode_map and index_pos < len(user_index["embeddings"]):
                user_index["episodes"][index_pos] = episode_map[episode_id]

        # Remove None entries
        user_index["episodes"] = [ep for ep in user_index["episodes"] if ep is not None]
        print(f"✅ Fixed! Episodes array now has {len(user_index['episodes'])} entries")

    # Test search
    print("🔎 Testing search...")
    search_query = RetrievalQuery(
        text="How does Evan describe being out on the water while kayaking and watching the sunset?",
        owner_id="evan_8",
        limit=5,
        strategy=RetrievalStrategy.EMBEDDING,
    )

    try:
        result = await new_retrieval_service.search(search_query)
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
    await new_retrieval_service.close()
    await episode_repo.close()
    print("\n🏁 Complete")


if __name__ == "__main__":
    asyncio.run(fix_embedding_index_manually())
