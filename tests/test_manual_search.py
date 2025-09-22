#!/usr/bin/env python3
"""
手动搜索测试脚本
Manual Search Test Script

这个脚本可以直接从memories文件夹中搜索指定用户的情景记忆和语义记忆
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Episode, SemanticMemory
from src.utils.embedding_client import EmbeddingClient
from src.search.vector_search import VectorSearch
from src.config import MemoryConfig

class ManualSearchTester:
    """手动搜索测试器"""
    
    def __init__(self, memories_path: str = "evaluation/memories"):
        """
        初始化搜索测试器
        
        Args:
            memories_path: memories文件夹路径
        """
        self.memories_path = Path(memories_path)
        self.config = MemoryConfig(
            storage_path=str(self.memories_path),
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536
        )
        
        # 初始化embedding客户端
        self.embedding_client = EmbeddingClient(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        # 初始化向量搜索
        self.vector_search = VectorSearch(
            embedding_client=self.embedding_client,
            storage_path=str(self.memories_path),
            dimension=self.config.embedding_dimension
        )
        
        print(f"✅ 初始化完成，memories路径: {self.memories_path}")
    
    def list_available_users(self) -> List[str]:
        """列出所有可用的用户"""
        episodes_dir = self.memories_path / "episodes"
        users = []
        
        if episodes_dir.exists():
            for file in episodes_dir.glob("*_episodes.jsonl"):
                # 从文件名提取用户ID (例如: Audrey_5_episodes.jsonl -> Audrey_5)
                user_id = file.stem.replace("_episodes", "")
                users.append(user_id)
        
        return sorted(users)
    
    def load_episodes(self, user_id: str) -> List[Episode]:
        """加载用户的情景记忆"""
        episodes_file = self.memories_path / "episodes" / f"{user_id}_episodes.jsonl"
        episodes = []
        
        if not episodes_file.exists():
            print(f"❌ 找不到用户 {user_id} 的情景记忆文件: {episodes_file}")
            return episodes
        
        try:
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        episode_data = json.loads(line)
                        episode = Episode.from_dict(episode_data)
                        episodes.append(episode)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  解析第{line_num}行时出错: {e}")
                        continue
            
            print(f"✅ 加载了 {len(episodes)} 个情景记忆 (用户: {user_id})")
            return episodes
            
        except Exception as e:
            print(f"❌ 加载情景记忆时出错: {e}")
            return []
    
    def load_semantic_memories(self, user_id: str) -> List[SemanticMemory]:
        """加载用户的语义记忆"""
        semantic_file = self.memories_path / "semantic" / f"{user_id}_semantic.jsonl"
        memories = []
        
        if not semantic_file.exists():
            print(f"❌ 找不到用户 {user_id} 的语义记忆文件: {semantic_file}")
            return memories
        
        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        memory_data = json.loads(line)
                        memory = SemanticMemory.from_dict(memory_data)
                        memories.append(memory)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  解析第{line_num}行时出错: {e}")
                        continue
            
            print(f"✅ 加载了 {len(memories)} 个语义记忆 (用户: {user_id})")
            return memories
            
        except Exception as e:
            print(f"❌ 加载语义记忆时出错: {e}")
            return []
    
    def search_episodes(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索情景记忆"""
        print(f"\n🔍 搜索情景记忆...")
        print(f"用户: {user_id}")
        print(f"查询: {query}")
        print(f"返回数量: {top_k}")
        
        # 加载episodes并建立索引
        episodes = self.load_episodes(user_id)
        if not episodes:
            return []
        
        try:
            # 建立索引
            self.vector_search.index_episodes(user_id, episodes)
            
            # 使用向量搜索
            results = self.vector_search.search_episodes(
                user_id=user_id,
                query=query,
                top_k=top_k
            )
            
            print(f"✅ 找到 {len(results)} 个相关情景记忆")
            return results
            
        except Exception as e:
            print(f"❌ 搜索情景记忆时出错: {e}")
            return []
    
    def search_semantic_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索语义记忆"""
        print(f"\n🔍 搜索语义记忆...")
        print(f"用户: {user_id}")
        print(f"查询: {query}")
        print(f"返回数量: {top_k}")
        
        # 加载semantic memories并建立索引
        memories = self.load_semantic_memories(user_id)
        if not memories:
            return []
        
        try:
            # 建立索引
            self.vector_search.index_semantic_memories(user_id, memories)
            
            # 使用向量搜索
            results = self.vector_search.search_semantic_memories(
                user_id=user_id,
                query=query,
                top_k=top_k
            )
            
            print(f"✅ 找到 {len(results)} 个相关语义记忆")
            return results
            
        except Exception as e:
            print(f"❌ 搜索语义记忆时出错: {e}")
            return []
    
    def display_episode_results(self, results: List[Dict[str, Any]]):
        """显示情景记忆搜索结果"""
        if not results:
            print("❌ 没有找到相关的情景记忆")
            return
        
        print(f"\n📋 情景记忆搜索结果 (共 {len(results)} 条):")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            timestamp = result.get('timestamp', 'No timestamp')
            message_count = result.get('message_count', 0)
            
            print(f"\n🔸 结果 {i} (相似度: {score:.4f})")
            print(f"📅 时间: {timestamp}")
            print(f"📝 标题: {title}")
            print(f"📄 内容: {content[:200]}{'...' if len(content) > 200 else ''}")
            print(f"💬 消息数: {message_count}")
            
            # 显示原始消息摘要
            original_messages = result.get('original_messages', [])
            if original_messages:
                print(f"💭 原始对话:")
                for j, msg in enumerate(original_messages[:3]):  # 只显示前3条
                    role = msg.get('role', 'Unknown')
                    msg_content = msg.get('content', '')[:100]
                    print(f"   {role}: {msg_content}{'...' if len(msg.get('content', '')) > 100 else ''}")
                if len(original_messages) > 3:
                    print(f"   ... 还有 {len(original_messages) - 3} 条消息")
            
            print("-" * 60)
    
    def display_semantic_results(self, results: List[Dict[str, Any]]):
        """显示语义记忆搜索结果"""
        if not results:
            print("❌ 没有找到相关的语义记忆")
            return
        
        print(f"\n📋 语义记忆搜索结果 (共 {len(results)} 条):")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            content = result.get('content', 'No content')
            knowledge_type = result.get('knowledge_type', 'knowledge')
            created_at = result.get('created_at', 'No timestamp')
            confidence = result.get('confidence', 0)
            related_episodes = result.get('related_episodes', [])
            
            print(f"\n🔸 结果 {i} (相似度: {score:.4f})")
            print(f"📅 创建时间: {created_at}")
            print(f"🏷️  知识类型: {knowledge_type}")
            print(f"📄 内容: {content}")
            print(f"🎯 置信度: {confidence}")
            
            if related_episodes:
                print(f"🔗 来源episodes: {len(related_episodes)} 个")
            
            print("-" * 60)
    
    def interactive_search(self):
        """交互式搜索"""
        print("\n🎯 交互式记忆搜索")
        print("=" * 50)
        
        # 显示可用用户
        users = self.list_available_users()
        if not users:
            print("❌ 没有找到任何用户数据")
            return
        
        print(f"\n📋 可用用户 ({len(users)} 个):")
        for i, user in enumerate(users, 1):
            print(f"  {i}. {user}")
        
        while True:
            try:
                # 选择用户
                print(f"\n请选择用户 (1-{len(users)}, 或输入 'q' 退出):")
                user_input = input("👤 用户选择: ").strip()
                
                if user_input.lower() == 'q':
                    print("👋 退出搜索")
                    break
                
                user_idx = int(user_input) - 1
                if user_idx < 0 or user_idx >= len(users):
                    print("❌ 无效的用户选择")
                    continue
                
                user_id = users[user_idx]
                print(f"✅ 选择用户: {user_id}")
                
                # 输入查询
                query = input("\n🔍 请输入搜索查询: ").strip()
                if not query:
                    print("❌ 查询不能为空")
                    continue
                
                # 输入返回数量
                top_k_input = input("📊 返回结果数量 (默认5): ").strip()
                top_k = int(top_k_input) if top_k_input.isdigit() else 5
                
                # 选择搜索类型
                print("\n📋 搜索类型:")
                print("  1. 情景记忆")
                print("  2. 语义记忆") 
                print("  3. 两者都搜索")
                
                search_type = input("🎯 选择搜索类型 (1-3): ").strip()
                
                if search_type in ['1', '3']:
                    # 搜索情景记忆
                    episode_results = self.search_episodes(user_id, query, top_k)
                    self.display_episode_results(episode_results)
                
                if search_type in ['2', '3']:
                    # 搜索语义记忆
                    semantic_results = self.search_semantic_memories(user_id, query, top_k)
                    self.display_semantic_results(semantic_results)
                
                print("\n" + "="*80)
                
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出搜索")
                break
            except Exception as e:
                print(f"❌ 搜索过程中出错: {e}")
                continue

def main():
    """主函数"""
    print("🚀 启动手动搜索测试器...")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        return
    
    try:
        # 创建搜索测试器
        tester = ManualSearchTester()
        
        # 启动交互式搜索
        tester.interactive_search()
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
