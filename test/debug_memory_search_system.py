"""
调试内存搜索系统，找出为什么目标记忆没有被检索到
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from evaluation.locomo.search import MemorySystemSearch
from src.models import SemanticMemory


def check_memory_in_jsonl(user_id: str, memory_id: str, storage_path: str):
    """检查目标记忆是否在JSONL文件中"""
    jsonl_path = Path(storage_path) / "semantic" / f"{user_id}_semantic.jsonl"
    
    print(f"1. 检查JSONL文件: {jsonl_path}")
    if not jsonl_path.exists():
        print(f"   ❌ 文件不存在")
        return None
    
    target_memory = None
    line_idx = -1
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                if data.get('memory_id') == memory_id:
                    target_memory = data
                    line_idx = idx
                    break
    
    if target_memory:
        print(f"   ✅ 找到目标记忆在第 {line_idx} 行")
        print(f"      内容: {target_memory.get('content', '')[:100]}...")
    else:
        print(f"   ❌ 未找到目标记忆")
    
    return target_memory, line_idx


def check_memory_in_loaded_data(searcher: MemorySystemSearch, user_id: str, memory_id: str):
    """检查内存系统是否正确加载了目标记忆"""
    print(f"\n2. 检查内存系统加载的数据")
    
    # 尝试获取已加载的语义记忆
    try:
        # 先触发数据加载
        searcher.memory_system.load_user_data_and_indices_for_method(user_id, searcher.search_method)
        
        # 获取语义记忆
        semantic_memories = searcher.memory_system.storage["semantic"].list_user_items(user_id)
        print(f"   加载了 {len(semantic_memories)} 条语义记忆")
        
        # 查找目标记忆
        for idx, mem in enumerate(semantic_memories):
            if mem.memory_id == memory_id:
                print(f"   ✅ 找到目标记忆在索引 {idx}")
                return True, idx
        
        print(f"   ❌ 未找到目标记忆在加载的数据中")
        return False, -1
        
    except Exception as e:
        print(f"   ❌ 加载数据时出错: {e}")
        return False, -1


def test_direct_search(searcher: MemorySystemSearch, user_id: str, query: str):
    """测试直接搜索"""
    print(f"\n3. 测试系统搜索")
    
    try:
        memories, search_time = searcher.search_memory(user_id, query)
        print(f"   搜索耗时: {search_time:.3f}s")
        print(f"   返回记忆数: {len(memories)}")
        
        # 只看语义记忆
        semantic_mems = [m for m in memories if m.get('memory_type') == 'semantic']
        print(f"   语义记忆数: {len(semantic_mems)}")
        
        # 检查前10个语义记忆
        print(f"\n   前10个语义记忆:")
        for i, mem in enumerate(semantic_mems[:10], 1):
            print(f"   #{i} score={mem.get('score')} id={mem.get('episode_id')} {mem.get('memory', '')[:60]}...")
        
        # 检查是否包含目标记忆
        target_id = "2651a8e2-f7ee-4646-881d-265ed862c6d4"
        for mem in semantic_mems:
            if mem.get('episode_id') == target_id:
                print(f"\n   ✅ 找到目标记忆!")
                return True
        
        print(f"\n   ❌ 未找到目标记忆")
        return False
        
    except Exception as e:
        print(f"   ❌ 搜索出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vector_search_directly(user_id: str, query: str, storage_path: str):
    """直接测试向量搜索"""
    print(f"\n4. 直接测试向量搜索引擎")
    
    try:
        from src.utils import EmbeddingClient
        from src.search import VectorSearch
        
        # 初始化
        client = EmbeddingClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        
        vector_search = VectorSearch(
            client,
            storage_path,
            1536
        )
        
        # 先加载用户数据
        print(f"   加载用户数据...")
        vector_search.load_user_semantic_memories(user_id)
        
        # 搜索语义记忆
        results = vector_search.search_semantic_memories(user_id, query, top_k=30)
        
        print(f"   返回 {len(results)} 条结果")
        
        # 检查前5个
        for i, res in enumerate(results[:5], 1):
            print(f"   #{i} score={res.get('score'):.4f} id={res.get('memory_id')} {res.get('content', '')[:60]}...")
        
        # 检查目标
        target_id = "2651a8e2-f7ee-4646-881d-265ed862c6d4"
        for i, res in enumerate(results):
            if res.get('memory_id') == target_id:
                print(f"\n   ✅ 找到目标记忆在第 {i+1} 位!")
                return True
        
        print(f"\n   ❌ 未找到目标记忆")
        return False
        
    except Exception as e:
        print(f"   ❌ 向量搜索出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 配置
    user_id = "Caroline_0"
    query = "What inspired Caroline's painting for the art show?"
    target_memory_id = "2651a8e2-f7ee-4646-881d-265ed862c6d4"
    
    # 使用与evaluation相同的路径
    storage_path = os.path.join('evaluation', 'memories')
    
    print("=== 调试内存搜索系统 ===")
    print(f"用户: {user_id}")
    print(f"查询: {query}")
    print(f"目标记忆ID: {target_memory_id}")
    print(f"存储路径: {storage_path}\n")
    
    # 1. 检查JSONL文件
    target_memory, line_idx = check_memory_in_jsonl(user_id, target_memory_id, storage_path)
    
    # 2. 初始化搜索系统（与evaluation相同的配置）
    searcher = MemorySystemSearch(
        output_path="test_debug_results.json",
        storage_path=storage_path,
        model="gpt-4o-mini",
        language="en",
        top_k_episodes=10,
        top_k_semantic=30,  # 增加到30以确保能看到
        include_original_messages_top_k=2,
        max_workers=8,
        save_batch_size=50,
        enable_memory_cleanup=False,
        search_method="vector"
    )
    
    # 3. 检查内存系统是否正确加载
    found_in_loaded, loaded_idx = check_memory_in_loaded_data(searcher, user_id, target_memory_id)
    
    # 4. 测试搜索
    found_in_search = test_direct_search(searcher, user_id, query)
    
    # 5. 直接测试向量搜索
    found_in_vector = check_vector_search_directly(user_id, query, storage_path)
    
    # 总结
    print("\n=== 总结 ===")
    print(f"JSONL文件中存在: {'✅' if target_memory else '❌'}")
    print(f"内存系统加载: {'✅' if found_in_loaded else '❌'}")
    print(f"系统搜索返回: {'✅' if found_in_search else '❌'}")
    print(f"直接向量搜索: {'✅' if found_in_vector else '❌'}")
    
    if not found_in_search and found_in_vector:
        print("\n问题可能在于:")
        print("- search_method 参数不一致")
        print("- top_k_semantic 设置太小")
        print("- 分数阈值过滤")
        print("- 结果格式化问题")


if __name__ == "__main__":
    main()
