"""
修复向量索引对齐问题
"""

import os
import json
import numpy as np
from pathlib import Path

def fix_index_alignment(base_path: str):
    """修复向量索引对齐问题"""
    base_path = Path(base_path)
    
    # 需要修复的用户列表（从检查结果中得到）
    misaligned_users = [
        "Calvin_9", "Caroline_0", "John_2", "Jon_1", "Tim_4"
    ]
    
    print("=== 修复向量索引对齐问题 ===")
    
    for user_id in misaligned_users:
        print(f"\n修复用户: {user_id}")
        
        # 路径定义
        jsonl_path = base_path / "episodes" / f"{user_id}_episodes.jsonl"
        emb_path = base_path / "episodes" / "vector_db" / f"{user_id}_embeddings.npy"
        faiss_path = base_path / "episodes" / "vector_db" / f"{user_id}.faiss"
        
        if not jsonl_path.exists():
            print(f"  跳过: JSONL文件不存在")
            continue
            
        # 读取JSONL文件计算实际行数
        jsonl_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    jsonl_count += 1
        
        print(f"  JSONL记录数: {jsonl_count}")
        
        # 检查embedding文件
        if emb_path.exists():
            embeddings = np.load(emb_path)
            emb_count = embeddings.shape[0]
            print(f"  Embedding数: {emb_count}")
            
            if emb_count > jsonl_count:
                # 截断embedding到正确大小
                print(f"  截断embedding: {emb_count} -> {jsonl_count}")
                truncated_embeddings = embeddings[:jsonl_count]
                np.save(emb_path, truncated_embeddings)
                print(f"  ✅ Embedding已修复")
            else:
                print(f"  Embedding数量正确")
        
        # 删除FAISS索引文件，让系统重建
        if faiss_path.exists():
            faiss_path.unlink()
            print(f"  ✅ FAISS索引已删除，将自动重建")
        
        print(f"  用户 {user_id} 修复完成")
    
    print("\n=== 修复完成 ===")
    print("请重新运行检查脚本验证修复结果")


if __name__ == "__main__":
    base_path = "/Users/nanjiayan/Desktop/nemori_oos/Nemori-code/evaluation/memories"
    fix_index_alignment(base_path)
