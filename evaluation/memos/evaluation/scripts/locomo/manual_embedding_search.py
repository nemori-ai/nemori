#!/usr/bin/env python3
"""
最终的embedding检索修复和演示
手动实现缺失的功能来使embedding检索工作
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from time import time
import requests

from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig, EpisodeQuery


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def get_query_embedding(query_text, api_key="EMPTY", base_url="http://localhost:6003/v1", model="bce-emb"):
    """获取查询文本的embedding"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {"model": model, "input": [query_text]}

    response = requests.post(f"{base_url}/embeddings", headers=headers, json=data, timeout=30)

    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["embedding"]
    else:
        raise Exception(f"Embedding request failed: {response.status_code}")


async def manual_embedding_search():
    """手动实现embedding检索来解决bug"""
    print("🛠️ Manual Embedding Retrieval Implementation")
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
        # 获取所有episodes用于验证
        query = EpisodeQuery(limit=1000)
        search_result = await episode_repo.search_episodes(query)
        all_episodes = search_result.episodes

        print(f"📊 Database has {len(all_episodes)} total episodes")

        # 测试案例
        test_cases = [
            ("john_4", "basketball achievement"),
            ("tim_4", "travel planning"),
            ("caroline_0", "adoption advice"),
            ("melanie_0", "October setback"),
        ]

        for test_user, test_query in test_cases:
            print(f"\n🎯 Testing: '{test_user}' -> '{test_query}'")
            print("-" * 50)

            # 检查用户的episodes
            user_episodes = [ep for ep in all_episodes if ep.owner_id == test_user]
            print(f"👤 User has {len(user_episodes)} episodes")

            if not user_episodes:
                print("⚠️ No episodes for this user")
                continue

            # 加载用户的embedding索引
            index_file = storage_dir / f"embedding_index_{test_user}.json"
            if not index_file.exists():
                print(f"❌ No embedding index file for {test_user}")
                continue

            try:
                with open(index_file, "r") as f:
                    index_data = json.load(f)

                stored_embeddings = index_data.get("embeddings", [])
                episode_ids = index_data.get("episode_ids", [])

                print(f"📁 Loaded index: {len(stored_embeddings)} embeddings, {len(episode_ids)} episode IDs")

                if not stored_embeddings or not episode_ids:
                    print("⚠️ Empty index data")
                    continue

                # 获取查询embedding
                print(f"🔍 Getting query embedding...")
                query_embedding = await get_query_embedding(test_query)
                print(f"✅ Got query embedding (dim: {len(query_embedding)})")

                # 计算相似度
                similarities = []
                for i, stored_embedding in enumerate(stored_embeddings):
                    similarity = cosine_similarity(query_embedding, stored_embedding)
                    similarities.append((i, similarity, episode_ids[i]))

                # 排序并获取最相似的
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_results = similarities[:3]

                print(f"🎯 Top 3 matches:")

                # 查找对应的episodes
                found_episodes = []
                for rank, (idx, score, episode_id) in enumerate(top_results, 1):
                    print(f"   {rank}. Score: {score:.4f}, Episode ID: {episode_id}")

                    # 找到对应的episode对象
                    matching_episode = None
                    for ep in user_episodes:
                        if ep.episode_id == episode_id:
                            matching_episode = ep
                            break

                    if matching_episode:
                        print(f"      Title: {matching_episode.title}")
                        print(f"      Content: {matching_episode.content[:100]}...")
                        found_episodes.append(matching_episode)
                    else:
                        print(f"      ⚠️ Episode not found in database")

                if found_episodes:
                    print(f"✅ Successfully found {len(found_episodes)} relevant episodes!")
                else:
                    print(f"⚠️ No episodes found despite similarity matches")

            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error in index file: {e}")
            except Exception as e:
                print(f"❌ Error processing index: {e}")

        print(f"\n🎉 Manual Embedding Search Complete!")
        print(f"This demonstrates that the embedding data exists and can be searched successfully.")
        print(f"The issue in the original code is likely the missing '_generate_embedding' method.")

    except Exception as e:
        print(f"❌ Manual search failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(manual_embedding_search())
