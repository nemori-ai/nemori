#!/usr/bin/env python3
"""Fix all embedding indices by populating episodes arrays."""

import asyncio
from pathlib import Path
from nemori.retrieval import (
    RetrievalConfig,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig


async def fix_all_embedding_indices():
    print("🔧 Fix All Embedding Indices")
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

    # Get embedding provider
    embedding_provider = retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)

    # Get all owner IDs from episode repository instead of direct DuckDB connection
    print("🔍 Getting all owner IDs from database...")

    # Use a simple approach - get all owners from existing embedding indices
    embedding_files = list(storage_dir.glob("embedding_index_*.json"))
    owner_ids = []
    for file in embedding_files:
        owner_id = file.stem.replace("embedding_index_", "")
        owner_ids.append(owner_id)

    owner_ids.sort()

    print(f"🎯 Found {len(owner_ids)} unique owners: {owner_ids}")

    fixed_count = 0

    for owner_id in owner_ids:
        print(f"\\n🔍 Checking {owner_id}...")

        # Get user index
        user_index = embedding_provider._get_user_index(owner_id)

        embeddings_count = len(user_index["embeddings"])
        episodes_count = len(user_index["episodes"])
        mappings_count = len(user_index["episode_id_to_index"])

        print(f"   📊 Status: {embeddings_count} embeddings, {episodes_count} episodes, {mappings_count} mappings")

        # Check if needs fixing (has embeddings but no episodes)
        if embeddings_count > 0 and episodes_count == 0:
            print(f"   🔧 Fixing {owner_id}...")

            # Get episodes from database
            result = await episode_repo.get_episodes_by_owner(owner_id)
            db_episodes = result.episodes if hasattr(result, "episodes") else result

            if db_episodes:
                # Create episode mapping
                episode_map = {ep.episode_id: ep for ep in db_episodes}

                # Populate episodes array in correct order
                user_index["episodes"] = [None] * len(user_index["embeddings"])
                for episode_id, index_pos in user_index["episode_id_to_index"].items():
                    if episode_id in episode_map and index_pos < len(user_index["embeddings"]):
                        user_index["episodes"][index_pos] = episode_map[episode_id]

                # Remove None entries
                user_index["episodes"] = [ep for ep in user_index["episodes"] if ep is not None]

                print(f"   ✅ Fixed! Now has {len(user_index['episodes'])} episodes")
                fixed_count += 1
            else:
                print(f"   ❌ No episodes found in database for {owner_id}")
        else:
            print(f"   ✅ Already OK")

    print(f"\\n🎯 Fixed {fixed_count} out of {len(owner_ids)} owners")

    # Test a few searches
    test_cases = [
        ("audrey_5", "What did Audrey set up in the backyard for their dogs?"),
        ("evan_8", "How does Evan describe being out on the water while kayaking?"),
        ("caroline_0", "What does Caroline like to do?"),
    ]

    print("\\n🔎 Testing searches...")
    for owner_id, query_text in test_cases:
        try:
            from nemori.retrieval import RetrievalQuery

            query = RetrievalQuery(
                text=query_text,
                owner_id=owner_id,
                limit=3,
                strategy=RetrievalStrategy.EMBEDDING,
            )

            result = await retrieval_service.search(query)
            print(f"   {owner_id}: {len(result.episodes)} episodes found for '{query_text[:50]}...'")

            if result.episodes:
                top_episode = result.episodes[0]
                print(f"      Top result: '{top_episode.title[:60]}...'")

        except Exception as e:
            print(f"   {owner_id}: Search failed - {e}")

    # Cleanup
    await retrieval_service.close()
    await episode_repo.close()
    print("\\n🏁 Complete")


if __name__ == "__main__":
    asyncio.run(fix_all_embedding_indices())
