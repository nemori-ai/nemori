#!/usr/bin/env python3
"""
深度调试embedding检索问题 - 修复版
"""

import asyncio
import json
import numpy as np
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


async def deep_debug_embedding():
    """深度调试embedding检索问题"""
    print("🔬 Deep Debugging Embedding Retrieval")
    print("=" * 60)

    version = "default"
    current_dir = Path(__file__).parent
    storage_dir = current_dir.parent.parent / "results" / "locomo" / f"nemori-{version}" / "storages"
    db_path = storage_dir / "nemori_memory.duckdb"

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
        # 设置retrieval service
        retrieval_service = RetrievalService(episode_repo)

        emb_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(storage_dir)},
            api_key="EMPTY",
            base_url="http://localhost:6003/v1",
            embed_model="bce-emb",
        )

        retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, emb_config)
        await retrieval_service.initialize()

        embedding_provider = retrieval_service.providers.get(RetrievalStrategy.EMBEDDING)

        print(f"✅ Embedding provider: {type(embedding_provider).__name__}")

        # 检查user_indices
        if hasattr(embedding_provider, "user_indices"):
            user_indices = embedding_provider.user_indices
            print(f"📊 User indices loaded: {list(user_indices.keys())}")

            # 检查john_4的索引
            if "john_4" in user_indices:
                john_index = user_indices["john_4"]
                print(f"\n🔍 Analyzing john_4 index:")
                print(f"   Index type: {type(john_index)}")

                # 检查索引的属性
                index_attrs = [attr for attr in dir(john_index) if not attr.startswith("_")]
                print(f"   Index attributes: {index_attrs}")

                # 检查常见属性
                if hasattr(john_index, "embeddings"):
                    embeddings = john_index.embeddings
                    print(
                        f"   Embeddings: {len(embeddings)} items, dimension {len(embeddings[0]) if embeddings else 0}"
                    )

                if hasattr(john_index, "episode_ids"):
                    episode_ids = john_index.episode_ids
                    print(f"   Episode IDs: {len(episode_ids)} items")
                    print(f"   Sample episode IDs: {episode_ids[:3] if episode_ids else 'None'}")

                # 测试搜索
                print(f"\n🔍 Testing manual search on john_4 index:")
                test_query = "basketball achievement"
                print(f"   Query: '{test_query}'")

                # 获取查询embedding
                if hasattr(embedding_provider, "openai_client"):
                    client = embedding_provider.openai_client
                    print(f"   ✅ Found OpenAI client")

                    try:
                        # 直接调用embedding
                        response = await client.embeddings.create(
                            model=embedding_provider.embed_model, input=[test_query]
                        )
                        query_embedding = response.data[0].embedding
                        print(f"   ✅ Got query embedding, dimension: {len(query_embedding)}")

                        # 手动计算相似度
                        if hasattr(john_index, "embeddings") and john_index.embeddings:
                            stored_embeddings = np.array(john_index.embeddings)
                            query_embedding_np = np.array(query_embedding)

                            # 计算余弦相似度
                            similarities = np.dot(stored_embeddings, query_embedding_np) / (
                                np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(query_embedding_np)
                            )

                            print(f"   📊 Similarity scores:")
                            print(f"     Count: {len(similarities)}")
                            print(f"     Max: {np.max(similarities):.4f}")
                            print(f"     Min: {np.min(similarities):.4f}")
                            print(f"     Mean: {np.mean(similarities):.4f}")

                            # 找出最相似的3个
                            top_indices = np.argsort(similarities)[-3:][::-1]
                            print(f"   🎯 Top 3 matches:")
                            for i, idx in enumerate(top_indices):
                                sim_score = similarities[idx]
                                episode_id = (
                                    john_index.episode_ids[idx]
                                    if hasattr(john_index, "episode_ids")
                                    else f"index_{idx}"
                                )
                                print(f"     {i+1}. Score: {sim_score:.4f}, Episode: {episode_id}")

                        else:
                            print(f"   ❌ No embeddings in john_4 index")

                    except Exception as e:
                        print(f"   ❌ Error getting embedding: {e}")

                else:
                    print(f"   ❌ No openai_client found")
            else:
                print(f"⚠️ john_4 not in user_indices")
        else:
            print(f"❌ No user_indices found in provider")

        # 测试provider的search方法
        print(f"\n🔍 Testing Provider Search Method Directly:")
        print("-" * 50)

        test_query = "basketball achievement"
        test_user = "john_4"

        try:
            # 直接调用provider的search方法
            provider_result = await embedding_provider.search(test_query, test_user, limit=3)
            print(f"Provider search result: {type(provider_result)}")

            if hasattr(provider_result, "episodes"):
                print(f"Episodes from provider: {len(provider_result.episodes)}")
                for i, episode in enumerate(provider_result.episodes):
                    print(f"  {i+1}. {episode.title[:50]}...")
            else:
                print(f"Provider result attributes: {dir(provider_result)}")

        except Exception as e:
            print(f"❌ Provider search failed: {e}")
            import traceback

            traceback.print_exc()

        # 最终测试：完整的retrieval service search
        print(f"\n🔍 Final Test: Full Retrieval Service Search:")
        print("-" * 50)

        query = RetrievalQuery(text=test_query, owner_id=test_user, limit=3, strategy=RetrievalStrategy.EMBEDDING)

        result = await retrieval_service.search(query)
        print(f"Final result: {len(result.episodes)} episodes found")

        if result.episodes:
            for i, episode in enumerate(result.episodes):
                print(f"  {i+1}. {episode.title}")
        else:
            print(f"No episodes found - investigating why...")

            # 检查query是否正确传递
            print(f"Query details:")
            print(f"  Text: {query.text}")
            print(f"  Owner ID: {query.owner_id}")
            print(f"  Strategy: {query.strategy}")
            print(f"  Limit: {query.limit}")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "retrieval_service" in locals():
            await retrieval_service.close()
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(deep_debug_embedding())
