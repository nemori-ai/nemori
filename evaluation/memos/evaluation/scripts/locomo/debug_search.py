#!/usr/bin/env python3
"""Debug script for nemori search issues."""

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


async def debug_search():
    print("🔍 Debug Nemori Search")
    print("=" * 50)

    # Setup storage
    storage_dir = Path("results/locomo/nemori-default/storages")
    db_path = storage_dir / "nemori_memory.duckdb"

    print(f"📁 Storage dir: {storage_dir.absolute()}")
    print(f"🗄️ DB path: {db_path.absolute()}")
    print(f"✅ DB exists: {db_path.exists()}")

    if not db_path.exists():
        print("❌ Database not found!")
        return

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

    # Check direct database query
    result = await episode_repo.get_episodes_by_owner("evan_8")
    episodes = result.episodes if hasattr(result, "episodes") else result
    print(f"📊 Direct DB query for evan_8: {len(episodes)} episodes")

    if episodes:
        print(f"   Sample episode: '{episodes[0].title}'")

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

    print("🔧 Registering embedding provider...")
    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()
    print("✅ Retrieval service initialized")

    # Test query
    query = "How does Evan describe being out on the water while kayaking and watching the sunset?"
    print(f"\n🔎 Testing search query: '{query}'")

    search_query = RetrievalQuery(
        text=query,
        owner_id="evan_8",
        limit=5,
        strategy=RetrievalStrategy.EMBEDDING,
    )

    print(f"📝 Query object: {search_query}")

    try:
        print("🔍 Executing search...")
        result = await retrieval_service.search(search_query)
        print(f"✅ Search completed: {len(result.episodes)} episodes found")

        for i, episode in enumerate(result.episodes[:3]):
            print(f"   {i+1}. '{episode.title}' (score: {getattr(episode, 'score', 'N/A')})")
            print(f"      Content: {episode.content[:100]}...")

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    await retrieval_service.close()
    await episode_repo.close()
    print("\n🧹 Cleanup complete")


if __name__ == "__main__":
    asyncio.run(debug_search())
