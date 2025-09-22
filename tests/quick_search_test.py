#!/usr/bin/env python3
"""
快速搜索测试脚本
Quick Search Test Script

简化版本，用于快速测试特定用户的搜索结果
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.memory_system import MemorySystem
from src.config import MemoryConfig

def quick_search_test(user_id: str, query: str, memories_path: str = "evaluation/memories"):
    """
    快速搜索测试
    
    Args:
        user_id: 用户ID (例如: Caroline_0, Audrey_5)
        query: 搜索查询
        memories_path: memories文件夹路径
    """
    print(f"🔍 快速搜索测试")
    print(f"用户: {user_id}")
    print(f"查询: {query}")
    print(f"数据路径: {memories_path}")
    print("=" * 60)
    
    try:
        # 创建配置
        config = MemoryConfig(
            storage_path=memories_path,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            search_top_k_episodes=5,
            search_top_k_semantic=5
        )
        
        # 创建记忆系统
        memory_system = MemorySystem(config)
        
        # 加载用户数据和索引
        print(f"📂 加载用户数据和索引...")
        memory_system.load_user_data_and_indices_for_method(user_id, "vector")
        
        # 执行搜索
        print(f"🔍 执行搜索...")
        results = memory_system.search_all(
            user_id=user_id,
            query=query,
            top_k_episodes=5,
            top_k_semantic=5,
            search_method="vector"
        )
        
        # 显示结果
        print(f"\n📋 搜索结果:")
        print("=" * 60)
        
        # 显示情景记忆结果
        episodic_results = results.get('episodic', [])
        print(f"\n🎬 情景记忆 ({len(episodic_results)} 条):")
        for i, result in enumerate(episodic_results, 1):
            score = result.get('score', 0)
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            timestamp = result.get('timestamp', 'No timestamp')
            
            print(f"\n  {i}. 相似度: {score:.4f}")
            print(f"     时间: {timestamp}")
            print(f"     标题: {title}")
            print(f"     内容: {content[:150]}{'...' if len(content) > 150 else ''}")
        
        # 显示语义记忆结果
        semantic_results = results.get('semantic', [])
        print(f"\n🧠 语义记忆 ({len(semantic_results)} 条):")
        for i, result in enumerate(semantic_results, 1):
            score = result.get('score', 0)
            content = result.get('content', 'No content')
            knowledge_type = result.get('knowledge_type', 'knowledge')
            created_at = result.get('created_at', 'No timestamp')
            
            print(f"\n  {i}. 相似度: {score:.4f}")
            print(f"     类型: {knowledge_type}")
            print(f"     时间: {created_at}")
            print(f"     内容: {content}")
        
        print(f"\n✅ 搜索完成!")
        
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()

def list_users(memories_path: str = "evaluation/memories") -> List[str]:
    """列出所有可用用户"""
    episodes_dir = Path(memories_path) / "episodes"
    users = []
    
    if episodes_dir.exists():
        for file in episodes_dir.glob("*_episodes.jsonl"):
            user_id = file.stem.replace("_episodes", "")
            users.append(user_id)
    
    return sorted(users)

def main():
    """主函数 - 可以直接修改这里的参数进行测试"""
    
    # 🎯 在这里修改测试参数
    TEST_USER = "Caroline_0"  # 修改为你想测试的用户
    TEST_QUERY = "career goals"  # 修改为你想搜索的内容
    MEMORIES_PATH = "evaluation/memories"  # memories文件夹路径
    
    print("🚀 快速搜索测试启动...")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 显示可用用户
    users = list_users(MEMORIES_PATH)
    print(f"\n📋 可用用户: {users}")
    
    if TEST_USER not in users:
        print(f"❌ 用户 {TEST_USER} 不存在")
        print(f"可用用户: {users}")
        return
    
    # 执行搜索测试
    quick_search_test(TEST_USER, TEST_QUERY, MEMORIES_PATH)

if __name__ == "__main__":
    main()
