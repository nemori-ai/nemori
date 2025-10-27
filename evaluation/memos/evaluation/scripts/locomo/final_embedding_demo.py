#!/usr/bin/env python3
"""
最终的embedding检索演示脚本
基于locomo_ingestion_emb.py保存的数据库进行embedding检索并输出详细结果
"""

import asyncio
import json
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
from nemori.storage.storage_types import StorageConfig, EpisodeQuery


async def create_retrieval_demo(version="default"):
    """创建完整的embedding检索演示"""
    print("🚀 Nemori Embedding Retrieval Demonstration")
    print("=" * 60)

    # 数据库路径设置
    current_dir = Path(__file__).parent
    storage_dir = current_dir.parent.parent / "results" / "locomo" / f"nemori-{version}" / "storages"
    db_path = storage_dir / "nemori_memory.duckdb"

    print(f"📁 Database: {db_path}")
    print(f"📂 Storage: {storage_dir}")

    if not db_path.exists():
        print("❌ Database not found! Please run locomo_ingestion_emb.py first.")
        return

    # 初始化存储
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )

    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()

    try:
        # 第一步：检查数据库内容
        print("\n📊 Database Statistics:")
        print("-" * 30)

        query = EpisodeQuery(limit=1000)
        search_result = await episode_repo.search_episodes(query)
        all_episodes = search_result.episodes

        # 统计信息
        owner_stats = {}
        for episode in all_episodes:
            owner_id = episode.owner_id
            if owner_id not in owner_stats:
                owner_stats[owner_id] = 0
            owner_stats[owner_id] += 1

        print(f"📚 Total episodes: {len(all_episodes)}")
        print(f"👥 Number of users: {len(owner_stats)}")
        print(f"🔍 Top users by episode count:")

        # 显示前5个用户
        sorted_users = sorted(owner_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for user, count in sorted_users:
            print(f"   {user}: {count} episodes")

        # 第二步：设置embedding检索
        print(f"\n🔧 Setting up embedding retrieval...")
        retrieval_service = RetrievalService(episode_repo)

        # 配置embedding
        emb_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(storage_dir)},
            api_key="EMPTY",
            base_url="http://localhost:6003/v1",
            embed_model="bce-emb",
        )

        retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, emb_config)
        await retrieval_service.initialize()
        print("✅ Embedding retrieval service initialized")

        # 第三步：执行检索测试
        print(f"\n🔍 Performing Embedding Retrieval Tests:")
        print("=" * 60)

        # 测试查询和用户
        test_cases = [
            ("john_4", "basketball achievement"),
            ("tim_4", "travel planning"),
            ("caroline_0", "adoption advice"),
            ("melanie_0", "October setback"),
            ("john_2", "project work"),
        ]

        all_results = []

        for user_id, query_text in test_cases:
            print(f"\n🎯 Testing: User '{user_id}' | Query: '{query_text}'")
            print("-" * 50)

            # 检查用户是否有episodes
            user_episodes = [ep for ep in all_episodes if ep.owner_id == user_id]
            print(f"👤 User has {len(user_episodes)} episodes in database")

            if user_episodes:
                # 显示一个示例episode
                sample_episode = user_episodes[0]
                print(f"📄 Sample episode: {sample_episode.title[:60]}...")

            # 创建检索查询
            retrieval_query = RetrievalQuery(
                text=query_text, owner_id=user_id, limit=3, strategy=RetrievalStrategy.EMBEDDING
            )

            start_time = time()

            try:
                # 执行检索
                print(f"⚡ Executing retrieval query...")
                result = await retrieval_service.search(retrieval_query)
                duration = (time() - start_time) * 1000

                print(f"⏱️ Search completed in {duration:.2f}ms")
                print(f"📊 Found {len(result.episodes)} matching episodes")

                if result.episodes:
                    print(f"🎯 Retrieved Episodes:")

                    episode_results = []
                    for i, episode in enumerate(result.episodes, 1):
                        score = getattr(episode, "score", "N/A")
                        print(f"\n   {i}. Score: {score}")
                        print(f"      Title: {episode.title}")
                        print(f"      Content: {episode.content[:200]}...")
                        print(
                            f"      Summary: {episode.summary[:150]}..." if episode.summary else "      Summary: None"
                        )

                        episode_results.append(
                            {
                                "episode_id": episode.episode_id,
                                "title": episode.title,
                                "content": episode.content[:500],  # Truncate for JSON
                                "summary": episode.summary,
                                "score": score,
                            }
                        )

                    all_results.append(
                        {
                            "user_id": user_id,
                            "query": query_text,
                            "found": len(result.episodes),
                            "duration_ms": duration,
                            "episodes": episode_results,
                        }
                    )
                else:
                    print(f"⚠️ No matching episodes found")
                    all_results.append(
                        {"user_id": user_id, "query": query_text, "found": 0, "duration_ms": duration, "episodes": []}
                    )

            except Exception as e:
                print(f"❌ Retrieval failed: {e}")
                import traceback

                traceback.print_exc()
                all_results.append(
                    {
                        "user_id": user_id,
                        "query": query_text,
                        "found": 0,
                        "duration_ms": 0,
                        "error": str(e),
                        "episodes": [],
                    }
                )

        # 第四步：保存和总结结果
        output_dir = Path(f"results/locomo/nemori-{version}/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "final_embedding_retrieval_demo.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n📈 Final Summary:")
        print("=" * 40)

        total_tests = len(test_cases)
        successful_tests = len([r for r in all_results if "error" not in r])
        tests_with_results = len([r for r in all_results if r["found"] > 0])
        total_episodes_found = sum(r["found"] for r in all_results)

        print(f"🧪 Total tests: {total_tests}")
        print(f"✅ Successful tests: {successful_tests}")
        print(f"🎯 Tests with results: {tests_with_results}")
        print(f"📚 Total episodes retrieved: {total_episodes_found}")

        if successful_tests > 0:
            avg_duration = sum(r["duration_ms"] for r in all_results if "error" not in r) / successful_tests
            print(f"⏱️ Average search time: {avg_duration:.2f}ms")

        print(f"💾 Detailed results saved to: {output_file}")
        print(f"\n🎉 Embedding Retrieval Demo Complete!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "retrieval_service" in locals():
            await retrieval_service.close()
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(create_retrieval_demo())
