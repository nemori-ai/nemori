#!/usr/bin/env python3
"""Rebuild embedding index for evan_8."""

import asyncio
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


async def rebuild_embedding_index():
    print("🔧 Rebuilding Embedding Index for evan_8")
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

    # Force rebuild by making a search query for evan_8
    # This will trigger the embedding index creation
    print("🔄 Triggering embedding index rebuild for evan_8...")

    search_query = RetrievalQuery(
        text="test",
        owner_id="evan_8",
        limit=1,
        strategy=RetrievalStrategy.EMBEDDING,
    )

    try:
        result = await retrieval_service.search(search_query)
        print(f"✅ Index rebuilt! Search returned {len(result.episodes)} episodes")

        # Test with actual query
        real_query = RetrievalQuery(
            text="How does Evan describe being out on the water while kayaking and watching the sunset?",
            owner_id="evan_8",
            limit=5,
            strategy=RetrievalStrategy.EMBEDDING,
        )

        result = await retrieval_service.search(real_query)
        print(f"🎯 Real search test: {len(result.episodes)} episodes found")

        for i, episode in enumerate(result.episodes[:2]):
            print(f"   {i+1}. '{episode.title}'")
            print(f"      {episode.content[:100]}...")

    except Exception as e:
        print(f"❌ Failed to rebuild index: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    await retrieval_service.close()
    await episode_repo.close()
    print("\n🏁 Complete")


if __name__ == "__main__":
    asyncio.run(rebuild_embedding_index())
