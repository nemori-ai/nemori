"""
验证搜索结果的一致性
"""

import os
import sys
import json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from evaluation.locomo.search import MemorySystemSearch


def run_search_and_check():
    """运行搜索并检查结果"""
    user_id = "Caroline_0"
    query = "What inspired Caroline's painting for the art show?"
    storage_path = os.path.join('evaluation', 'memories')
    
    # 使用与evaluation相同的配置
    searcher = MemorySystemSearch(
        output_path="test_verify_results.json",
        storage_path=storage_path,
        model="gpt-4o-mini",
        language="en",
        top_k_episodes=10,
        top_k_semantic=20,  # 与main函数中一致
        include_original_messages_top_k=2,
        max_workers=100,
        save_batch_size=200,
        enable_memory_cleanup=False,
        search_method="vector"
    )
    
    print("=== 验证搜索一致性 ===")
    print(f"配置: top_k_semantic=20, search_method=vector")
    
    # 执行搜索
    memories, search_time = searcher.search_memory(user_id, query)
    
    # 分析语义记忆
    semantic_mems = [m for m in memories if m.get('memory_type') == 'semantic']
    print(f"\n返回了 {len(semantic_mems)} 条语义记忆")
    
    # 显示前20条语义记忆的ID和内容
    print("\n前20条语义记忆:")
    target_found = False
    for i, mem in enumerate(semantic_mems[:20], 1):
        mem_id = mem.get('episode_id', '')
        content = mem.get('memory', '')[:80]
        score = mem.get('score', 0)
        
        is_target = mem_id == "2651a8e2-f7ee-4646-881d-265ed862c6d4"
        if is_target:
            target_found = True
            print(f"🎯 #{i} score={score} id={mem_id} {content}...")
        else:
            print(f"   #{i} score={score} id={mem_id} {content}...")
    
    if not target_found:
        print("\n❌ 目标记忆不在前20条中!")
        # 检查是否在后续结果中
        for i, mem in enumerate(semantic_mems[20:], 21):
            if mem.get('episode_id') == "2651a8e2-f7ee-4646-881d-265ed862c6d4":
                print(f"   目标记忆在第 {i} 位")
                break
    else:
        print("\n✅ 目标记忆被成功检索!")
    
    # 保存结果用于对比
    with open("test_verify_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "user_id": user_id,
            "semantic_memories": semantic_mems[:20]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 test_verify_results.json")


if __name__ == "__main__":
    run_search_and_check()
