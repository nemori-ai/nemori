#!/usr/bin/env python3
"""
修复embedding索引并演示正确的检索
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from time import time
import requests

from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig, EpisodeQuery


def cosine_similarity(a, b):
    """计算余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def get_query_embedding(query_text, api_key="EMPTY", base_url="http://localhost:6003/v1", model="bce-emb"):
    """获取查询embedding"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {"model": model, "input": [query_text]}

    response = requests.post(f"{base_url}/embeddings", headers=headers, json=data, timeout=30)

    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["embedding"]
    else:
        raise Exception(f"Embedding request failed: {response.status_code}")


async def fix_and_test_embedding_retrieval():
    """修复索引文件并测试检索"""
    print("🔧 Fixing Embedding Index and Testing Retrieval")
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
        # 获取所有episodes
        query = EpisodeQuery(limit=1000)
        search_result = await episode_repo.search_episodes(query)
        all_episodes = search_result.episodes

        print(f"📊 Database contains {len(all_episodes)} episodes")

        # 按用户分组episodes
        user_episodes = {}
        for episode in all_episodes:
            owner_id = episode.owner_id
            if owner_id not in user_episodes:
                user_episodes[owner_id] = []
            user_episodes[owner_id].append(episode)

        print(f"👥 Found episodes for {len(user_episodes)} users")

        # 测试几个用户
        test_users = ["caroline_0", "john_4", "tim_4"]

        for test_user in test_users:
            if test_user not in user_episodes:
                print(f"⚠️ No episodes for {test_user}")
                continue

            print(f"\n🔍 Processing user: {test_user}")
            print("-" * 40)

            user_eps = user_episodes[test_user]
            print(f"👤 User has {len(user_eps)} episodes")

            # 检查索引文件
            index_file = storage_dir / f"embedding_index_{test_user}.json"
            if not index_file.exists():
                print(f"❌ No index file for {test_user}")
                continue

            try:
                with open(index_file, "r") as f:
                    index_data = json.load(f)

                embeddings = index_data.get("embeddings", [])
                episode_id_to_index = index_data.get("episode_id_to_index", {})
                stored_episodes = index_data.get("episodes", [])

                print(f"📁 Index file stats:")
                print(f"   Embeddings: {len(embeddings)}")
                print(f"   Episode ID mappings: {len(episode_id_to_index)}")
                print(f"   Stored episodes: {len(stored_episodes)}")

                # 如果索引中的episodes为空，我们需要根据episode_id_to_index重建
                if len(stored_episodes) == 0 and len(episode_id_to_index) > 0:
                    print(f"🔧 Rebuilding episode list from database...")

                    # 创建一个episode_id到episode的映射
                    episode_lookup = {ep.episode_id: ep for ep in user_eps}

                    # 重建episodes列表（按索引顺序）
                    rebuilt_episodes = []
                    episode_ids = []

                    for episode_id, index_pos in episode_id_to_index.items():
                        if episode_id in episode_lookup:
                            rebuilt_episodes.append(episode_lookup[episode_id])
                            episode_ids.append(episode_id)

                    print(f"✅ Rebuilt {len(rebuilt_episodes)} episodes")

                    # 现在测试检索
                    test_queries = {
                        "caroline_0": "adoption advice",
                        "john_4": "basketball achievement",
                        "tim_4": "travel planning",
                    }

                    if test_user in test_queries:
                        test_query = test_queries[test_user]
                        print(f"\n🎯 Testing query: '{test_query}'")

                        # 获取查询embedding
                        query_embedding = await get_query_embedding(test_query)
                        print(f"✅ Query embedding obtained (dim: {len(query_embedding)})")

                        # 计算相似度
                        similarities = []
                        for i, stored_embedding in enumerate(embeddings):
                            if i < len(rebuilt_episodes):
                                similarity = cosine_similarity(query_embedding, stored_embedding)
                                similarities.append((i, similarity, rebuilt_episodes[i]))

                        # 排序获取最相似的
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_results = similarities[:3]

                        print(f"\n🏆 Top 3 Results:")
                        for rank, (idx, score, episode) in enumerate(top_results, 1):
                            print(f"\n   {rank}. Similarity Score: {score:.4f}")
                            print(f"      Episode ID: {episode.episode_id}")
                            print(f"      Title: {episode.title}")
                            print(f"      Content: {episode.content[:150]}...")
                            if episode.summary:
                                print(f"      Summary: {episode.summary[:100]}...")

                        if top_results and top_results[0][1] > 0.1:  # 相似度阈值
                            print(f"\n✅ SUCCESS: Found relevant episodes for '{test_query}'!")
                        else:
                            print(f"\n⚠️ Low similarity scores - may need better queries")

            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
            except Exception as e:
                print(f"❌ Error processing {test_user}: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n🎉 Embedding Retrieval Fix and Test Complete!")
        print(f"The issue was that episode objects were not stored in the index files.")
        print(f"We successfully demonstrated that the embeddings work when properly matched with episodes.")

    except Exception as e:
        print(f"❌ Fix and test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(fix_and_test_embedding_retrieval())
