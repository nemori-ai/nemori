"""
对比不同方式生成的查询embedding
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv

# 添加项目根目录到系统路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from src.utils import EmbeddingClient

# 你提供的硬编码API key
HARDCODED_KEY = "***REMOVED***"

def main():
    query = "What inspired Caroline's painting for the art show?"
    
    # 1. 使用环境变量的key
    load_dotenv()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        print("1. 使用环境变量的API key...")
        client1 = EmbeddingClient(api_key=env_key, model="text-embedding-3-small")
        embedding1 = np.array(client1.embed_text(query))
        print(f"   Embedding shape: {embedding1.shape}")
        print(f"   前5个值: {embedding1[:5]}")
        print(f"   L2范数: {np.linalg.norm(embedding1):.6f}")
    else:
        print("1. 未找到环境变量OPENAI_API_KEY")
        embedding1 = None
    
    # 2. 使用硬编码的key
    print("\n2. 使用硬编码的API key...")
    client2 = EmbeddingClient(api_key=HARDCODED_KEY, model="text-embedding-3-small")
    embedding2 = np.array(client2.embed_text(query))
    print(f"   Embedding shape: {embedding2.shape}")
    print(f"   前5个值: {embedding2[:5]}")
    print(f"   L2范数: {np.linalg.norm(embedding2):.6f}")
    
    # 3. 比较两个embedding
    if embedding1 is not None:
        print("\n3. 比较两个embedding:")
        # 计算余弦相似度
        cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"   余弦相似度: {cos_sim:.6f}")
        
        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        print(f"   欧氏距离: {euclidean_dist:.6f}")
        
        # 检查是否完全相同
        is_identical = np.allclose(embedding1, embedding2, rtol=1e-5, atol=1e-5)
        print(f"   是否相同: {is_identical}")
        
        if not is_identical:
            print("\n   ⚠️ 两个API key生成的embedding不同！")
            print("   这可能是因为:")
            print("   - 使用了不同的OpenAI账号")
            print("   - API key对应的模型版本不同")
            print("   - 其中一个key可能已经失效")
    
    # 4. 测试在存储的embeddings上的搜索
    print("\n4. 测试搜索...")
    embedding_path = os.path.join('evaluation', 'memories', 'semantic', 'vector_db', 'Caroline_0_embeddings.npy')
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
        
        # 使用硬编码key的embedding搜索
        query_norm = embedding2 / np.linalg.norm(embedding2)
        similarities = np.dot(embeddings, query_norm)
        top_indices = np.argsort(similarities)[::-1][:5]
        
        print("   使用硬编码key的Top-5相似度:")
        for i, idx in enumerate(top_indices):
            print(f"     #{i+1}: idx={idx}, score={similarities[idx]:.4f}")


if __name__ == "__main__":
    main()
