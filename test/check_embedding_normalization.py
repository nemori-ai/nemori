"""
检查embedding是否已经归一化
"""

import os
import numpy as np

def check_normalization(user_id: str):
    embedding_path = os.path.join('evaluation', 'memories', 'semantic', 'vector_db', f'{user_id}_embeddings.npy')
    
    if not os.path.exists(embedding_path):
        print(f"未找到embedding文件: {embedding_path}")
        return
    
    embeddings = np.load(embedding_path)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 计算每个向量的L2范数
    norms = np.linalg.norm(embeddings, axis=1)
    
    print(f"\n范数统计:")
    print(f"  最小值: {norms.min():.6f}")
    print(f"  最大值: {norms.max():.6f}")
    print(f"  平均值: {norms.mean():.6f}")
    print(f"  标准差: {norms.std():.6f}")
    
    # 检查是否归一化（范数应该接近1）
    is_normalized = np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)
    print(f"\n是否已归一化: {is_normalized}")
    
    # 显示前10个向量的范数
    print("\n前10个向量的范数:")
    for i in range(min(10, len(norms))):
        print(f"  向量{i}: {norms[i]:.6f}")


if __name__ == "__main__":
    check_normalization("Caroline_0")
