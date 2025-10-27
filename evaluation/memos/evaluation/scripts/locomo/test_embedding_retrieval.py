#!/usr/bin/env python3
"""
测试embedding检索功能的脚本
对locomo_ingestion_emb.py保存的数据库进行embedding检索并输出检索结果
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
from nemori.storage.storage_types import StorageConfig


async def setup_retrieval_service(version="default"):
    """设置检索服务"""
    print("🔧 Setting up retrieval service...")

    # 数据库路径 - 使用相对于当前脚本的路径
    current_dir = Path(__file__).parent
    storage_dir = current_dir.parent.parent / "results" / "locomo" / f"nemori-{version}" / "storages"
    db_path = storage_dir / "nemori_memory.duckdb"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Please run ingestion first.")

    print(f"📁 Database path: {db_path}")
    print(f"📂 Storage directory: {storage_dir}")

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

    # 配置embedding检索
    emb_api_key = "EMPTY"
    emb_base_url = "http://localhost:6003/v1"
    embed_model = "bce-emb"

    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key=emb_api_key,
        base_url=emb_base_url,
        embed_model=embed_model,
    )

    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()

    print("✅ Retrieval service initialized successfully")
    return retrieval_service


async def get_all_users(retrieval_service):
    """获取数据库中所有用户ID"""
    print("\n📊 Getting all available users...")

    # 通过查询数据库获取所有不同的owner_id
    try:
        # 这里我们尝试从数据库直接获取用户信息
        result = await retrieval_service.episode_repo.get_all_episodes()

        # 提取所有唯一的owner_id
        owner_ids = set()
        for episode in result:
            if hasattr(episode, "owner_id") and episode.owner_id:
                owner_ids.add(episode.owner_id)

        print(f"🔍 Found {len(owner_ids)} unique users: {list(owner_ids)}")
        return list(owner_ids)

    except Exception as e:
        print(f"❌ Error getting users: {e}")
        # 如果直接查询失败，使用默认的用户ID模式
        default_users = [
            "caroline_0",
            "melanie_0",
            "jon_1",
            "gina_1",
            "john_2",
            "maria_2",
            "joanna_3",
            "nate_3",
            "tim_4",
            "john_4",
            "audrey_5",
            "andrew_5",
            "james_6",
            "john_6",
            "deborah_7",
            "jolene_7",
            "evan_8",
            "sam_8",
            "calvin_9",
            "dave_9",
        ]
        print(f"🔄 Using default user list: {default_users}")
        return default_users


async def test_embedding_search(retrieval_service, queries, users, top_k=5):
    """测试embedding检索功能"""
    print(f"\n🔍 Testing embedding search with {len(queries)} queries and {len(users)} users")
    print(f"📊 Retrieving top {top_k} results for each query")

    all_results = []

    for i, query_text in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"🔎 Query {i}/{len(queries)}: '{query_text}'")
        print(f"{'='*60}")

        query_results = []

        for user_id in users:
            print(f"\n👤 Searching for user: {user_id}")

            # 创建检索查询
            query = RetrievalQuery(text=query_text, owner_id=user_id, limit=top_k, strategy=RetrievalStrategy.EMBEDDING)

            start_time = time()

            try:
                # 执行检索
                result = await retrieval_service.search(query)
                duration = (time() - start_time) * 1000

                print(f"   ⏱️ Search completed in {duration:.2f}ms")
                print(f"   📊 Found {len(result.episodes)} episodes")

                if len(result.episodes) > 0:
                    print("   📋 Retrieved episodes:")
                    for j, episode in enumerate(result.episodes, 1):
                        print(f"      {j}. [Score: {getattr(episode, 'score', 'N/A'):.4f}] {episode.title}")
                        print(f"         Content: {episode.content[:100]}...")
                        print(f"         Summary: {episode.summary}")
                        print()

                    # 保存结果
                    episode_data = []
                    for episode in result.episodes:
                        episode_data.append(
                            {
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
                query_results.append(
                    {"user_id": user_id, "episodes_found": 0, "duration_ms": 0, "error": str(e), "episodes": []}
                )

        all_results.append({"query": query_text, "results": query_results})

    return all_results


async def save_results(results, version="default"):
    """保存检索结果到文件"""
    output_dir = Path(f"results/locomo/nemori-{version}/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "embedding_retrieval_test.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")


async def main():
    """主函数"""
    print("🚀 Starting Embedding Retrieval Test")
    print("=" * 60)

    version = "default"

    try:
        # 设置检索服务
        retrieval_service = await setup_retrieval_service(version)

        # 获取所有用户
        users = await get_all_users(retrieval_service)

        if not users:
            print("❌ No users found in database")
            return

        # 测试查询
        test_queries = [
            "What advice does Caroline give for getting started with adoption?",
            "What setback did Melanie face in October 2023?",
            "How does Jon feel about his work situation?",
            "What plans does Gina have for the weekend?",
            "What project is John working on?",
            "Tell me about Maria's hobbies",
            "What is Joanna's opinion about the new policy?",
            "Describe Nate's travel experiences",
            "What challenges is Tim facing?",
            "How does Audrey spend her free time?",
        ]

        # 执行检索测试
        results = await test_embedding_search(
            retrieval_service, test_queries, users[:5], top_k=3
        )  # 限制用户数量以避免输出过多

        # 保存结果
        await save_results(results, version)

        # 输出统计信息
        print(f"\n📈 Test Summary:")
        print(f"{'='*40}")
        total_searches = len(test_queries) * len(users[:5])
        successful_searches = 0
        total_episodes_found = 0

        for query_result in results:
            for user_result in query_result["results"]:
                if "error" not in user_result:
                    successful_searches += 1
                total_episodes_found += user_result["episodes_found"]

        print(f"📊 Total searches: {total_searches}")
        print(f"✅ Successful searches: {successful_searches}")
        print(f"❌ Failed searches: {total_searches - successful_searches}")
        print(f"📚 Total episodes found: {total_episodes_found}")
        print(f"📊 Average episodes per successful search: {total_episodes_found / max(successful_searches, 1):.2f}")

        print("\n🎉 Embedding retrieval test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        if "retrieval_service" in locals():
            await retrieval_service.close()


if __name__ == "__main__":
    asyncio.run(main())
