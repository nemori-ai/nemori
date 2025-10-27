#!/usr/bin/env python3
"""
检查DuckDB数据库内容的脚本
"""

import asyncio
from pathlib import Path

from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig


async def check_database():
    """检查数据库内容"""
    print("🔍 Checking database content...")

    # 数据库路径
    current_dir = Path(__file__).parent
    storage_dir = current_dir.parent.parent / "results" / "locomo" / "nemori-default" / "storages"
    db_path = storage_dir / "nemori_memory.duckdb"

    print(f"📁 Database path: {db_path}")

    if not db_path.exists():
        print("❌ Database file does not exist!")
        return

    # 配置存储
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )

    # 初始化存储库
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()

    try:
        # 获取所有episodes - 使用search_episodes方法
        print("\n📚 Fetching episodes from database...")

        from nemori.storage.storage_types import EpisodeQuery

        # 创建一个查询所有episodes的query
        query = EpisodeQuery(
            limit=1000,  # 获取前1000个episodes
        )

        # 搜索episodes
        search_result = await episode_repo.search_episodes(query)
        episodes = search_result.episodes

        print(f"📊 Total episodes found: {len(episodes)}")

        if len(episodes) > 0:
            # 统计按owner_id分组的数据
            owner_counts = {}
            for episode in episodes:
                owner_id = episode.owner_id
                if owner_id not in owner_counts:
                    owner_counts[owner_id] = 0
                owner_counts[owner_id] += 1

            print(f"\n👥 Episodes by owner:")
            for owner_id, count in owner_counts.items():
                print(f"   {owner_id}: {count} episodes")

            # 显示前几个episodes的示例
            print(f"\n📋 Sample episodes (first 3):")
            for i, episode in enumerate(episodes[:3]):
                print(f"\n   Episode {i+1}:")
                print(f"     ID: {episode.episode_id}")
                print(f"     Owner: {episode.owner_id}")
                print(f"     Title: {episode.title}")
                print(f"     Content: {episode.content[:200]}...")
                print(f"     Summary: {episode.summary}")
                if hasattr(episode, "timestamp"):
                    print(f"     Timestamp: {episode.timestamp}")
        else:
            print("⚠️ No episodes found in the database")

    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(check_database())
