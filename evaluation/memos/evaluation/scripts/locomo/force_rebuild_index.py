#!/usr/bin/env python3
"""Force rebuild embedding index for evan_8 from scratch."""

import asyncio
import json
import os
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


async def force_rebuild_embedding_index():
    print("🔥 Force Rebuild Embedding Index for evan_8")
    print("=" * 50)

    # Setup storage
    storage_dir = Path("results/locomo/nemori-default/storages")
    db_path = storage_dir / "nemori_memory.duckdb"

    # # Delete existing corrupted index file
    # index_file = storage_dir / "embedding_index_evan_8.json"
    # if index_file.exists():
    #     print(f"🗑️ Removing corrupted index file: {index_file}")
    #     index_file.unlink()

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

    # Get evan_8 episodes to verify they exist
    result = await episode_repo.get_episodes_by_owner("evan_8")
    episodes = result.episodes if hasattr(result, "episodes") else result
    print(f"📊 Found {len(episodes)} episodes in database for evan_8")

    if not episodes:
        print("❌ No episodes found for evan_8!")
        await episode_repo.close()
        return

    # Sample episode
    sample_episode = episodes[0]
    print(f"📝 Sample episode: '{sample_episode.title}'")
    print(f"   Content: {sample_episode.content[:100]}...")

    # Setup retrieval service
    retrieval_service = RetrievalService(episode_repo)

    # Setup embedding provider
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key="EMPTY",
        base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb",
    )

    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()

    print("✅ Retrieval service initialized")

    # Get embedding provider
    embedding_provider = retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)
    print(f"🔧 Embedding provider: {embedding_provider.__class__.__name__}")

    # Force rebuild by manually building embeddings
    print("🔄 Force building embeddings for evan_8...")

    # Get user index (should be empty/new)
    user_index = embedding_provider._get_user_index("evan_8")
    print(f"📊 Initial index state: {len(user_index['embeddings'])} embeddings, {len(user_index['episodes'])} episodes")

    # Manually build embeddings for each episode
    print("⚡ Building embeddings...")

    for i, episode in enumerate(episodes):
        print(f"   Processing episode {i+1}/{len(episodes)}: {episode.title[:50]}...")

        # Build searchable text
        searchable_text = embedding_provider._build_searchable_text(episode)

        # Generate embedding
        try:
            embedding = embedding_provider._generate_embedding(searchable_text)

            if embedding and not all(x == 0 for x in embedding):
                # Add to index
                user_index["episodes"].append(episode)
                user_index["embeddings"].append(embedding)
                user_index["episode_id_to_index"][episode.id] = len(user_index["episodes"]) - 1

                print(f"   ✅ Added embedding (dim: {len(embedding)})")
            else:
                print(f"   ❌ Failed to generate embedding")

        except Exception as e:
            print(f"   ❌ Error generating embedding: {e}")

    print(f"🎯 Built {len(user_index['embeddings'])} embeddings for {len(user_index['episodes'])} episodes")

    # Save index to disk
    print("💾 Saving index to disk...")
    if embedding_provider.persistence_enabled:
        embedding_provider._save_index_to_disk("evan_8", user_index)
        print("✅ Index saved")
    else:
        print("⚠️ Persistence not enabled")

    # Cleanup
    await retrieval_service.close()
    await episode_repo.close()
    print("\n🏁 Complete")


if __name__ == "__main__":
    asyncio.run(force_rebuild_embedding_index())
