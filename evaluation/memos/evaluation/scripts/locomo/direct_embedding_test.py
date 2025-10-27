#!/usr/bin/env python3
"""
直接测试embedding检索功能的脚本
基于数据库内容进行embedding检索并输出检索结果
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


async def setup_direct_retrieval_service(version="default"):
    """设置直接检索服务"""
    print("🔧 Setting up direct retrieval service...")

    # 数据库路径
    current_dir = Path(__file__).parent
    storage_dir = current_dir.parent.parent / "results" / "locomo" / f"nemori-{version}" / "storages"
    db_path = storage_dir / "nemori_memory.duckdb"

    print(f"📁 Database path: {db_path}")
    print(f"📂 Storage directory: {storage_dir}")

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Please run ingestion first.")

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

    # 设置检索服务
    retrieval_service = RetrievalService(episode_repo)

    # 配置embedding检索 - 保持与ingestion一致
    emb_api_key = "EMPTY"
    emb_base_url = "http://localhost:6003/v1"
    embed_model = "bce-emb"

    try:
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(storage_dir)},
            api_key=emb_api_key,
            base_url=emb_base_url,
            embed_model=embed_model,
        )

        print(f"🔧 Registering EMBEDDING provider with config:")
        print(f"   API Key: {emb_api_key}")
        print(f"   Base URL: {emb_base_url}")
        print(f"   Embed Model: {embed_model}")
        print(f"   Storage Directory: {storage_dir}")

        retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
        await retrieval_service.initialize()

        print("✅ Retrieval service initialized successfully")
        return retrieval_service

    except Exception as e:
        print(f"❌ Error setting up retrieval service: {e}")
        import traceback

        traceback.print_exc()
        raise


async def test_direct_embedding_retrieval(version="default"):
    """测试直接embedding检索"""
    print("🚀 Starting Direct Embedding Retrieval Test")
    print("=" * 60)

    try:
        # 设置检索服务
        retrieval_service = await setup_direct_retrieval_service(version)

        # 测试查询
        test_queries = [
            "basketball achievement",
            "travel planning",
            "visa requirements",
            "Barcelona recommendation",
            "John and Tim conversation",
        ]

        # 测试用户
        test_users = ["john_4", "tim_4", "caroline_0", "melanie_0"]

        all_results = []

        for i, query_text in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🔎 Query {i}/{len(test_queries)}: '{query_text}'")
            print(f"{'='*60}")

            query_results = []

            for user_id in test_users:
                print(f"\n👤 Searching for user: {user_id}")

                # 创建检索查询
                query = RetrievalQuery(
                    text=query_text, owner_id=user_id, limit=3, strategy=RetrievalStrategy.EMBEDDING  # 限制返回结果数量
                )

                start_time = time()

                try:
                    # 执行检索
                    print(f"   📝 Executing search with query: {query}")
                    result = await retrieval_service.search(query)
                    duration = (time() - start_time) * 1000

                    print(f"   ⏱️ Search completed in {duration:.2f}ms")
                    print(f"   📊 Found {len(result.episodes)} episodes")

                    if len(result.episodes) > 0:
                        print("   📋 Retrieved episodes:")
                        for j, episode in enumerate(result.episodes, 1):
                            score = getattr(episode, "score", "N/A")
                            print(f"      {j}. [Score: {score}] {episode.title}")
                            print(f"         Content: {episode.content[:150]}...")
                            if episode.summary:
                                print(f"         Summary: {episode.summary[:150]}...")
                            print()

                        # 保存结果
                        episode_data = []
                        for episode in result.episodes:
                            episode_data.append(
                                {
                                    "episode_id": episode.episode_id,
                                    "title": episode.title,
                                    "content": episode.content,
                                    "summary": episode.summary,
                                    "score": getattr(episode, "score", None),
                                    "timestamp": str(episode.timestamp) if hasattr(episode, "timestamp") else None,
                                }
                            )

                        query_results.append(
                            {
                                "user_id": user_id,
                                "episodes_found": len(result.episodes),
                                "duration_ms": duration,
                                "episodes": episode_data,
                            }
                        )
                    else:
                        print("   ⚠️ No episodes found")
                        query_results.append(
                            {"user_id": user_id, "episodes_found": 0, "duration_ms": duration, "episodes": []}
                        )

                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    import traceback

                    traceback.print_exc()
                    query_results.append(
                        {"user_id": user_id, "episodes_found": 0, "duration_ms": 0, "error": str(e), "episodes": []}
                    )

            all_results.append({"query": query_text, "results": query_results})

        # 保存结果
        output_dir = Path(f"results/locomo/nemori-{version}/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "direct_embedding_retrieval_test.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

        # 输出统计信息
        print(f"\n📈 Test Summary:")
        print(f"{'='*40}")
        total_searches = len(test_queries) * len(test_users)
        successful_searches = 0
        total_episodes_found = 0

        for query_result in all_results:
            for user_result in query_result["results"]:
                if "error" not in user_result:
                    successful_searches += 1
                total_episodes_found += user_result["episodes_found"]

        print(f"📊 Total searches: {total_searches}")
        print(f"✅ Successful searches: {successful_searches}")
        print(f"❌ Failed searches: {total_searches - successful_searches}")
        print(f"📚 Total episodes found: {total_episodes_found}")
        if successful_searches > 0:
            print(f"📊 Average episodes per successful search: {total_episodes_found / successful_searches:.2f}")

        print("\n🎉 Direct embedding retrieval test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        if "retrieval_service" in locals():
            await retrieval_service.close()


if __name__ == "__main__":
    asyncio.run(test_direct_embedding_retrieval())
