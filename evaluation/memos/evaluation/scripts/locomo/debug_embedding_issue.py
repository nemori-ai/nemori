#!/usr/bin/env python3
"""
调试embedding检索问题的脚本
深入分析为什么所有查询都返回0结果
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
from nemori.storage.storage_types import StorageConfig, EpisodeQuery


async def debug_embedding_retrieval():
    """调试embedding检索问题"""
    print("🔍 Debugging Embedding Retrieval Issue")
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
        # 步骤1：检查embedding索引文件内容
        print("\n📁 Step 1: Analyzing Embedding Index Files")
        print("-" * 50)

        index_files = list(storage_dir.glob("embedding_index_*.json"))
        print(f"Found {len(index_files)} embedding index files")

        for index_file in index_files[:3]:  # 检查前3个文件
            owner_id = index_file.stem.replace("embedding_index_", "")
            print(f"\n📄 Checking {owner_id}:")

            try:
                with open(index_file, "r") as f:
                    # 只读取文件的开头部分来分析结构
                    content = f.read(1000)  # 读取前1000字符
                    print(f"   File size: {index_file.stat().st_size / 1024:.1f}KB")
                    print(f"   Content preview: {content[:200]}...")

                    # 尝试解析JSON结构
                    f.seek(0)  # 回到文件开头
                    try:
                        data = json.load(f)
                        if "embeddings" in data:
                            embeddings = data["embeddings"]
                            print(f"   Embeddings count: {len(embeddings)}")
                            if embeddings:
                                print(f"   Embedding dimension: {len(embeddings[0])}")

                        if "episode_ids" in data:
                            episode_ids = data["episode_ids"]
                            print(f"   Episode IDs count: {len(episode_ids)}")
                            print(f"   Sample episode ID: {episode_ids[0] if episode_ids else 'None'}")

                    except json.JSONDecodeError as je:
                        print(f"   ❌ JSON parsing error: {je}")

            except Exception as e:
                print(f"   ❌ Error reading file: {e}")

        # 步骤2：直接检查embedding provider的状态
        print(f"\n🔧 Step 2: Testing Embedding Provider Directly")
        print("-" * 50)

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

        # 获取embedding provider并检查其状态
        embedding_provider = retrieval_service.providers.get(RetrievalStrategy.EMBEDDING)
        if embedding_provider:
            print(f"✅ Embedding provider found: {type(embedding_provider).__name__}")

            # 检查provider的内部状态
            if hasattr(embedding_provider, "indices"):
                indices = embedding_provider.indices
                print(f"📊 Loaded indices: {list(indices.keys())}")

                # 检查一个具体的索引
                if "john_4" in indices:
                    john_index = indices["john_4"]
                    print(f"📋 John_4 index type: {type(john_index)}")
                    if hasattr(john_index, "embeddings"):
                        print(f"📊 John_4 embeddings shape: {np.array(john_index.embeddings).shape}")
                        print(f"📊 John_4 episode_ids count: {len(john_index.episode_ids)}")
                else:
                    print("⚠️ john_4 not in loaded indices")
                    print(f"📊 Available indices: {list(indices.keys())}")
            else:
                print("⚠️ No 'indices' attribute found in provider")
                # 尝试其他可能的属性名
                provider_attrs = [attr for attr in dir(embedding_provider) if not attr.startswith("_")]
                print(f"📋 Provider attributes: {provider_attrs}")
        else:
            print("❌ Embedding provider not found!")

        # 步骤3：手动测试相似度计算
        print(f"\n🧮 Step 3: Manual Similarity Calculation Test")
        print("-" * 50)

        test_query = "basketball achievement"
        test_user = "john_4"

        print(f"Query: '{test_query}'")
        print(f"User: {test_user}")

        # 获取查询的embedding
        if hasattr(embedding_provider, "embedding_client"):
            client = embedding_provider.embedding_client
            try:
                query_embedding = await client.get_embedding(test_query)
                print(f"✅ Got query embedding, dimension: {len(query_embedding)}")

                # 检查是否有对应用户的索引
                if hasattr(embedding_provider, "indices") and test_user in embedding_provider.indices:
                    user_index = embedding_provider.indices[test_user]

                    if hasattr(user_index, "embeddings") and user_index.embeddings:
                        stored_embeddings = np.array(user_index.embeddings)
                        query_embedding_np = np.array(query_embedding)

                        # 计算余弦相似度
                        similarities = np.dot(stored_embeddings, query_embedding_np) / (
                            np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(query_embedding_np)
                        )

                        print(f"📊 Calculated similarities:")
                        print(f"   Count: {len(similarities)}")
                        print(f"   Max similarity: {np.max(similarities):.4f}")
                        print(f"   Min similarity: {np.min(similarities):.4f}")
                        print(f"   Mean similarity: {np.mean(similarities):.4f}")

                        # 找出最相似的前3个
                        top_indices = np.argsort(similarities)[-3:][::-1]
                        print(f"📋 Top 3 similarities:")
                        for i, idx in enumerate(top_indices):
                            sim_score = similarities[idx]
                            episode_id = (
                                user_index.episode_ids[idx] if hasattr(user_index, "episode_ids") else f"index_{idx}"
                            )
                            print(f"   {i+1}. Similarity: {sim_score:.4f}, Episode: {episode_id}")
                    else:
                        print("⚠️ No embeddings found in user index")
                else:
                    print(f"⚠️ No index found for user {test_user}")
                    if hasattr(embedding_provider, "indices"):
                        print(f"   Available users: {list(embedding_provider.indices.keys())}")

            except Exception as e:
                print(f"❌ Error getting query embedding: {e}")
                import traceback

                traceback.print_exc()
        elif embedding_provider:
            print("⚠️ No embedding_client found in provider")
            provider_attrs = [attr for attr in dir(embedding_provider) if not attr.startswith("_")]
            print(f"📋 Provider attributes: {provider_attrs}")

        # 步骤4：检查retrieval service的search方法
        print(f"\n🔍 Step 4: Debugging Search Method")
        print("-" * 50)

        query = RetrievalQuery(text=test_query, owner_id=test_user, limit=3, strategy=RetrievalStrategy.EMBEDDING)

        print(f"Executing search with detailed logging...")

        # 添加一些调试信息到search过程中
        try:
            result = await retrieval_service.search(query)
            print(f"Search result type: {type(result)}")
            print(f"Episodes found: {len(result.episodes)}")

            if hasattr(result, "metadata"):
                print(f"Result metadata: {result.metadata}")

        except Exception as e:
            print(f"❌ Search failed: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if "retrieval_service" in locals():
            await retrieval_service.close()
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(debug_embedding_retrieval())
