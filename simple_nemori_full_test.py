#!/usr/bin/env python3
"""
Nemori 简化功能测试脚本

这是一个更简单易用的测试脚本，测试nemori的核心功能：
1. 创建简单但丰富的对话数据
2. 测试episodic memory的创建和存储
3. 验证semantic memory的知识发现
4. 演示unified retrieval的搜索能力

使用方法:
python simple_nemori_full_test.py
"""

import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# 添加路径以导入nemori模块
current_dir = Path(__file__).parent
nemori_root = current_dir / "nemori"
sys.path.insert(0, str(nemori_root))
sys.path.insert(0, str(current_dir / "evaluation/memos/evaluation/scripts/locomo"))

from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
from nemori.retrieval import RetrievalQuery


def create_simple_test_conversations():
    """创建简单的测试对话数据"""
    conversations = [
        {
            "user_id": "conv_001",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1": [
                    {
                        "speaker": "Alice",
                        "text": "我最近在学习Python机器学习，你有什么建议的库吗？",
                        "timestamp": "2024-01-20T10:00:00Z"
                    },
                    {
                        "speaker": "Bob",
                        "text": "推荐scikit-learn作为入门，pandas用于数据处理，numpy用于数值计算。",
                        "timestamp": "2024-01-20T10:02:00Z"
                    },
                    {
                        "speaker": "Alice",
                        "text": "深度学习方面呢？我听说TensorFlow和PyTorch很流行。",
                        "timestamp": "2024-01-20T10:04:00Z"
                    },
                    {
                        "speaker": "Bob",
                        "text": "PyTorch对初学者更友好，动态图机制更容易调试。TensorFlow在生产环境应用更广泛。",
                        "timestamp": "2024-01-20T10:06:00Z"
                    }
                ],
                "session_1_date_time": "10:00 AM on 20 January, 2024"
            }
        },
        {
            "user_id": "conv_002", 
            "conversation": {
                "speaker_a": "Charlie",
                "speaker_b": "Diana",
                "session_1": [
                    {
                        "speaker": "Charlie",
                        "text": "我们公司要开发一个推荐系统，你觉得用什么算法比较好？",
                        "timestamp": "2024-01-21T14:00:00Z"
                    },
                    {
                        "speaker": "Diana",
                        "text": "协同过滤是经典选择，但现在深度学习的embedding方法效果更好。",
                        "timestamp": "2024-01-21T14:02:00Z"
                    },
                    {
                        "speaker": "Charlie",
                        "text": "冷启动问题怎么解决？新用户没有历史数据。",
                        "timestamp": "2024-01-21T14:04:00Z"
                    },
                    {
                        "speaker": "Diana",
                        "text": "可以结合内容过滤，基于物品特征推荐。或者用流行度作为fallback策略。",
                        "timestamp": "2024-01-21T14:06:00Z"
                    }
                ],
                "session_1_date_time": "2:00 PM on 21 January, 2024"
            }
        },
        {
            "user_id": "conv_003",
            "conversation": {
                "speaker_a": "Eve", 
                "speaker_b": "Frank",
                "session_1": [
                    {
                        "speaker": "Eve",
                        "text": "周末想去户外活动，你有什么推荐吗？",
                        "timestamp": "2024-01-22T16:00:00Z"
                    },
                    {
                        "speaker": "Frank",
                        "text": "可以去爬山，我知道一个地方风景很好，而且不太难爬。",
                        "timestamp": "2024-01-22T16:02:00Z"
                    },
                    {
                        "speaker": "Eve",
                        "text": "听起来不错！我喜欢拍照，那里适合摄影吗？",
                        "timestamp": "2024-01-22T16:04:00Z"
                    },
                    {
                        "speaker": "Frank",
                        "text": "绝对适合！山顶可以看日出，还有瀑布和森林，我经常带相机去。",
                        "timestamp": "2024-01-22T16:06:00Z"
                    }
                ],
                "session_1_date_time": "4:00 PM on 22 January, 2024"
            }
        }
    ]
    
    return pd.DataFrame(conversations)


async def simple_test():
    """运行简化的测试流程"""
    print("🚀 Nemori 简化功能测试")
    print("=" * 50)
    
    # 创建测试数据
    print("📋 创建测试数据...")
    test_data = create_simple_test_conversations()
    print(f"✅ 创建了 {len(test_data)} 个对话")
    
    # 初始化实验
    print("\n🔧 初始化 Nemori...")
    experiment = NemoriExperiment(
        version="simple_test",
        episode_mode="speaker",
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        max_concurrency=1
    )
    
    try:
        # 设置LLM (使用简单配置)
        print("\n🤖 设置 LLM...")
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"
        model = "gpt-4o-mini"
        
        llm_ok = await experiment.setup_llm_provider(model, api_key, base_url)
        if not llm_ok:
            print("❌ LLM 设置失败")
            return
            
        # 加载数据
        print("\n📊 加载数据...")
        experiment.load_locomo_data(test_data)
        
        # 设置存储和检索
        print("\n🗄️ 设置存储...")
        await experiment.setup_storage_and_retrieval(
            emb_api_key="EMPTY",
            emb_base_url="http://localhost:6007/v1",
            embed_model="qwen3-emb"
        )
        
        # 构建episodes和语义记忆
        print("\n🏗️ 构建记忆...")
        await experiment.build_episodes_semantic()
        
        # 显示结果
        print(f"\n📊 结果统计:")
        print(f"  Episodes: {len(experiment.episodes)}")
        semantic_count = getattr(experiment, 'actual_semantic_count', 0)
        print(f"  语义概念: {semantic_count}")
        
        # 简单的检索测试
        print(f"\n🔍 检索测试:")
        if experiment.episodes:
            owner_id = experiment.episodes[0].owner_id
            
            test_queries = ["机器学习", "推荐系统", "摄影"]
            
            for query_text in test_queries:
                print(f"\n  查询: '{query_text}'")
                try:
                    query = RetrievalQuery(
                        text=query_text,
                        owner_id=owner_id,
                        limit=2,
                        strategy=experiment.retrievalstrategy
                    )
                    
                    results = await experiment.retrieval_service.search(query)
                    
                    if results and results.episodes:
                        print(f"    ✅ 找到 {len(results.episodes)} 个相关结果")
                        for result in results.episodes[:1]:  # 只显示第一个
                            content = result.content[:100] + "..." if len(result.content) > 100 else result.content
                            print(f"    📄 {content}")
                    else:
                        print(f"    ❌ 未找到结果")
                        
                except Exception as e:
                    print(f"    ❌ 查询出错: {e}")
        
        print(f"\n🎉 测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test())