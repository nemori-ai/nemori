"""
调试搜索索引对齐问题
检查向量索引和原始数据是否对齐
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from src.models import SemanticMemory
from src.utils import EmbeddingClient
from dotenv import load_dotenv

load_dotenv()


def load_semantic_memories_from_jsonl(user_id: str, storage_path: str) -> list:
    """从JSONL文件加载语义记忆"""
    memories = []
    jsonl_path = Path(storage_path) / "semantic" / f"{user_id}_semantic.jsonl"
    
    if jsonl_path.exists():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    memory_data = json.loads(line)
                    memory = SemanticMemory.from_dict(memory_data)
                    memories.append({
                        'idx': idx,
                        'memory_id': memory.memory_id,
                        'content': memory.content
                    })
    
    return memories


def load_embeddings(user_id: str, storage_path: str) -> np.ndarray:
    """加载embedding文件"""
    embedding_path = Path(storage_path) / "semantic" / "vector_db" / f"{user_id}_embeddings.npy"
    
    if embedding_path.exists():
        return np.load(embedding_path)
    else:
        return None


def check_target_memory(memories: list) -> tuple:
    """查找目标记忆的索引"""
    target_sub = "Caroline's painting for the art show was inspired by her visit to an LGBTQ center"
    
    for mem in memories:
        if target_sub.lower() in mem['content'].lower():
            return mem['idx'], mem['memory_id'], mem['content']
    
    return None, None, None


def compute_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """计算余弦相似度"""
    # 查询向量归一化
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # embeddings已经归一化，直接点乘即可
    similarities = np.dot(embeddings, query_norm)
    
    return similarities


def main():
    user_id = "Caroline_0"
    storage_path = os.path.join('evaluation', 'memories')
    query = "What inspired Caroline's painting for the art show?"
    
    print(f"=== 调试搜索索引对齐问题 ===")
    print(f"用户: {user_id}")
    print(f"查询: {query}")
    print(f"存储路径: {storage_path}\n")
    
    # 1. 加载JSONL文件中的记忆
    memories = load_semantic_memories_from_jsonl(user_id, storage_path)
    print(f"1. 从JSONL加载了 {len(memories)} 条语义记忆")
    
    # 2. 查找目标记忆
    target_idx, target_id, target_content = check_target_memory(memories)
    if target_idx is not None:
        print(f"\n2. 找到目标记忆:")
        print(f"   索引: {target_idx}")
        print(f"   ID: {target_id}")
        print(f"   内容: {target_content[:100]}...")
    else:
        print("\n2. 未找到目标记忆")
        return
    
    # 3. 加载embeddings
    embeddings = load_embeddings(user_id, storage_path)
    if embeddings is None:
        print("\n3. 未找到embedding文件")
        return
    
    print(f"\n3. 加载了embeddings: shape={embeddings.shape}")
    
    # 4. 检查数量是否匹配
    if len(memories) != embeddings.shape[0]:
        print(f"\n❌ 数量不匹配!")
        print(f"   JSONL记忆数: {len(memories)}")
        print(f"   Embedding数: {embeddings.shape[0]}")
        return
    else:
        print(f"\n✅ 数量匹配: {len(memories)}")
    
    # 5. 生成查询embedding
    print(f"\n4. 生成查询embedding...")
    client = EmbeddingClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    query_embedding = np.array(client.embed_text(query))
    
    # 6. 计算相似度
    similarities = compute_similarity(query_embedding, embeddings)
    
    # 7. 获取top-k结果
    top_k = 20
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n5. Top-{top_k} 搜索结果:")
    for rank, idx in enumerate(top_indices, 1):
        mem = memories[idx]
        score = similarities[idx]
        is_target = "🎯" if idx == target_idx else "  "
        print(f"{is_target} #{rank}\tscore={score:.4f}\tidx={idx}\t{mem['content'][:80]}...")
    
    # 8. 检查目标记忆的排名
    if target_idx in top_indices:
        target_rank = np.where(top_indices == target_idx)[0][0] + 1
        print(f"\n✅ 目标记忆在第 {target_rank} 位")
    else:
        print(f"\n❌ 目标记忆不在Top-{top_k}中!")
        print(f"   目标记忆相似度: {similarities[target_idx]:.4f}")
        print(f"   目标记忆排名: {np.where(np.argsort(similarities)[::-1] == target_idx)[0][0] + 1}")
    
    # 9. 打印你系统返回的第一条记忆（根据你提供的结果）
    print("\n6. 你系统返回的第一条记忆（应该对应索引）:")
    your_first_content = "Caroline attended an LGBTQ+ pride parade on June 26, 2023."
    for mem in memories:
        if your_first_content in mem['content']:
            print(f"   找到: idx={mem['idx']}, content={mem['content']}")
            print(f"   该记忆的相似度: {similarities[mem['idx']]:.4f}")
            break


if __name__ == "__main__":
    main()
