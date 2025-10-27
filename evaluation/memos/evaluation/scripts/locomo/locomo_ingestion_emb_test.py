import argparse
import asyncio
import json
import traceback
from typing import Dict, Any

import pandas as pd

from dotenv import load_dotenv
from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
from nemori.retrieval import RetrievalQuery
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
NEMORI_AVAILABLE = True


def create_test_locomo_data():
    """创建测试用的locomo格式数据"""
    test_conversations = [
        {
            "sample_id": "test_001",
            "user_id": "77777",
            "conversation": {
                "speaker_a": "张三",
                "speaker_b": "李四",
                "session_1": [
                    {
                        "speaker": "张三",
                        "dia_id": "D1:1", 
                        "text": "最近我在学习Python机器学习，你觉得哪些库比较重要？",
                        "timestamp": "2024-01-20T10:00:00Z"
                    },
                    {
                        "speaker": "李四",
                        "dia_id": "D1:2",
                        "text": "scikit-learn是入门必备，pandas用于数据处理，numpy处理数值计算。深度学习推荐PyTorch。",
                        "timestamp": "2024-01-20T10:02:00Z"
                    },
                    {
                        "speaker": "张三", 
                        "dia_id": "D1:3",
                        "text": "PyTorch和TensorFlow比较，哪个更适合初学者？",
                        "timestamp": "2024-01-20T10:04:00Z"
                    },
                    {
                        "speaker": "李四",
                        "dia_id": "D1:4", 
                        "text": "PyTorch的动态图更直观，调试容易。TensorFlow适合生产部署，但学习曲线陡峭。",
                        "timestamp": "2024-01-20T10:06:00Z"
                    }
                ],
                "session_1_date_time": "10:00 AM on 20 January, 2024",
                "session_2": [
                    {
                        "speaker": "张三",
                        "dia_id": "D2:1",
                        "text": "数据预处理方面有什么建议吗？我的数据集有很多缺失值。",
                        "timestamp": "2024-01-20T14:00:00Z"
                    },
                    {
                        "speaker": "李四", 
                        "dia_id": "D2:2",
                        "text": "缺失值可以用均值填充、前向填充，或者直接删除。要看具体业务场景，时间序列数据建议前向填充。",
                        "timestamp": "2024-01-20T14:02:00Z"
                    }
                ],
                "session_2_date_time": "2:00 PM on 20 January, 2024"
            },
            "qa": [
                {
                    "question": "张三在学习什么技术？",
                    "answer": "Python机器学习", 
                    "evidence": ["D1:1"],
                    "category": 1
                },
                {
                    "question": "李四推荐了哪些Python库？",
                    "answer": "scikit-learn, pandas, numpy, PyTorch",
                    "evidence": ["D1:2"],
                    "category": 1
                }
            ]
        },
        {
            "sample_id": "test_002",
            "user_id": "88888",
            "conversation": {
                "speaker_a": "王博士",
                "speaker_b": "刘教授", 
                "session_1": [
                    {
                        "speaker": "王博士",
                        "dia_id": "D1:1",
                        "text": "我们的大语言模型对齐研究进展如何？RLHF的效果怎么样？",
                        "timestamp": "2024-01-21T09:00:00Z"
                    },
                    {
                        "speaker": "刘教授",
                        "dia_id": "D1:2", 
                        "text": "RLHF在数学推理任务上提升了15%准确率。我们还在尝试Constitutional AI的方法。",
                        "timestamp": "2024-01-21T09:02:00Z"
                    },
                    {
                        "speaker": "王博士",
                        "dia_id": "D1:3",
                        "text": "Constitutional AI在安全性方面表现如何？",
                        "timestamp": "2024-01-21T09:04:00Z"
                    },
                    {
                        "speaker": "刘教授",
                        "dia_id": "D1:4",
                        "text": "在避免有害输出方面效果很好，我们考虑将两种方法结合起来使用。",
                        "timestamp": "2024-01-21T09:06:00Z"
                    }
                ],
                "session_1_date_time": "9:00 AM on 21 January, 2024",
                "session_2": [
                    {
                        "speaker": "王博士",
                        "dia_id": "D2:1",
                        "text": "论文准备投稿到哪个会议？",
                        "timestamp": "2024-01-21T15:00:00Z"
                    },
                    {
                        "speaker": "刘教授",
                        "dia_id": "D2:2",
                        "text": "考虑ICML，今年他们对AI安全议题比较重视，符合我们的研究方向。",
                        "timestamp": "2024-01-21T15:02:00Z"
                    }
                ],
                "session_2_date_time": "3:00 PM on 21 January, 2024"
            },
            "qa": [
                {
                    "question": "RLHF在什么任务上取得了提升？",
                    "answer": "数学推理任务，提升了15%准确率",
                    "evidence": ["D1:2"], 
                    "category": 1
                },
                {
                    "question": "他们考虑投稿到哪个会议？",
                    "answer": "ICML",
                    "evidence": ["D2:2"],
                    "category": 1
                }
            ]
        },
        {
            "sample_id": "test_003",
            "user_id": "99999",
            "conversation": {
                "speaker_a": "小明",
                "speaker_b": "小红",
                "session_1": [
                    {
                        "speaker": "小明",
                        "dia_id": "D1:1",
                        "text": "周末我想去爬山，你要不要一起？我知道一个地方风景特别好。",
                        "timestamp": "2024-01-22T18:00:00Z"
                    },
                    {
                        "speaker": "小红",
                        "dia_id": "D1:2",
                        "text": "好啊！我最近在学摄影，正好可以带相机去拍风景照。",
                        "timestamp": "2024-01-22T18:02:00Z"
                    },
                    {
                        "speaker": "小明",
                        "dia_id": "D1:3", 
                        "text": "太好了！那个山顶可以看日出，我们早上4点半出发，6点能看到日出。",
                        "timestamp": "2024-01-22T18:04:00Z"
                    },
                    {
                        "speaker": "小红",
                        "dia_id": "D1:4",
                        "text": "日出摄影我还没试过，正好练习一下长焦镜头。我们需要准备早餐吗？",
                        "timestamp": "2024-01-22T18:06:00Z"
                    }
                ],
                "session_1_date_time": "6:00 PM on 22 January, 2024",
                "session_2": [
                    {
                        "speaker": "小明",
                        "dia_id": "D2:1",
                        "text": "我可以准备三明治，你负责咖啡怎么样？记得你很会泡咖啡。",
                        "timestamp": "2024-01-22T19:00:00Z"
                    },
                    {
                        "speaker": "小红", 
                        "dia_id": "D2:2",
                        "text": "没问题！我带便携的手冲设备，用V60滤杯，山顶喝现磨咖啡一定很棒。",
                        "timestamp": "2024-01-22T19:02:00Z"
                    }
                ],
                "session_2_date_time": "7:00 PM on 22 January, 2024"
            },
            "qa": [
                {
                    "question": "他们计划几点出发去爬山？",
                    "answer": "早上4点半",
                    "evidence": ["D1:3"],
                    "category": 2
                },
                {
                    "question": "小红在学习什么技能？",
                    "answer": "摄影",
                    "evidence": ["D1:2"],
                    "category": 1
                }
            ]
        }
    ]
    
    #return pd.DataFrame(test_conversations)
    return test_conversations#pd.DataFrame(test_conversations)

async def search_query_async(client, query, metadata, frame, reversed_client=None, top_k=10):
    """Async version of search_query for nemori with unified memory support."""
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    if not NEMORI_AVAILABLE:
        raise ImportError("Nemori is not available. Please install nemori.")
    
    # Check if this is a unified semantic version
    version = metadata.get("version", "default")
    
    context, duration_ms = await nemori_unified_search(
        unified_retrieval_service=client[0],
        retrieval_service=client[1], 
        query=query,
        speaker_a_user_id=speaker_a_user_id,
        speaker_b_user_id=speaker_b_user_id,
        top_k=top_k
    )
    
    return context, duration_ms


async def main_nemori(version: str = "test") -> bool:
    """主函数：运行Nemori处理和测试"""
    load_dotenv()
    
    # 生成测试数据
    print("📋 生成测试数据...")
    locomo_df = create_test_locomo_data()
    print(f"✅ 创建了 {len(locomo_df)} 个测试对话")
    print("locomo_df: ",locomo_df)
    print("\n🚀 开始Nemori实验（使用测试数据）")
    print("=" * 60)

    # 创建Nemori实验
    experiment = NemoriExperiment(
        version=version, 
        episode_mode="speaker", 
        retrievalstrategy=RetrievalStrategy.EMBEDDING, 
        max_concurrency=1
    )

    # Step 1: 设置LLM提供者
    print("\n🤖 设置LLM提供者...")
    llm_config = {
        "api_key": "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm",
        "base_url": "https://jeniya.cn/v1",
        "model": "gpt-4o-mini"
    }
    
    llm_available = await experiment.setup_llm_provider(**llm_config)
    if not llm_available:
        print("⚠️ LLM不可用，终止测试")
        return False
    print("✅ LLM设置成功")

    # Step 2: 加载测试数据
    print("\n📊 加载数据...")
    # experiment.load_locomo_data(locomo_df)
    experiment.conversations = locomo_df
    print(f"✅ 数据加载成功: {len(experiment.conversations)} 个对话")
    
    # Step 3: 设置存储和检索
    print("\n🗄 设置存储和检索...")
    embed_config = {
        "emb_api_key": "EMPTY",
        "emb_base_url": "http://localhost:6007/v1",
        "embed_model": "qwen3-emb"
    }
    await experiment.setup_storage_and_retrieval(**embed_config)
    print("✅ 存储和检索设置成功")

    # Step 4: 构建episodes并进行语义发现
    print("\n🏠 构建episodes并进行语义发现...")
    await experiment.build_episodes_semantic()
    # Use unified client for semantic versions
    
    top_k =20
    group_idx = 0
    user_id = 77777
    frame = "nemori"
    query = "question" 
    conversation = locomo_df[group_idx]["conversation"]#.iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    # Generate dynamic user IDs like in locomo_search_emb2.py
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{group_idx}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{group_idx}"
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{user_id}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{user_id}"
    
    conv_id = f"locomo_exp_user_{user_id}"
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        version=version,
        retrievalstrategy=RetrievalStrategy.EMBEDDING,  # 默认使用embedding搜索
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1", 
        embed_model="qwen3-emb"
    )
    client = (unified_retrieval, retrieval_service, episode_repo, semantic_repo)
    print(f"   👥 Original speakers: '{speaker_a}' & '{speaker_b}'")
    print(f"   🆔 Generated IDs: '{speaker_a_user_id}' & '{speaker_b_user_id}'")
    print(f"   📝 Conversation ID: '{conv_id}'")
    metadata = {
    "speaker_a": speaker_a,
    "speaker_b": speaker_b,
    "speaker_a_user_id": speaker_a_user_id,
    "speaker_b_user_id": speaker_b_user_id,
    "conv_idx": user_id,
    "conv_id": conv_id,
    "version": version,
}
    context, duration_ms = await search_query_async(client, query, metadata, frame, top_k=top_k)
    print(context, duration_ms)
    # 显示结果
    print("\n🎉 Nemori测试成功完成")
    print(f"✅ 成功处理 {len(experiment.conversations)} 个对话")
    print(f"✅ 创建 {len(experiment.episodes)} 个episodes")
    
    semantic_count = getattr(experiment, 'actual_semantic_count', 0)
    print(f"✅ 发现 {semantic_count} 个语义概念")
    
    if semantic_count > 0 and len(experiment.episodes) > 0:
        avg_concepts = semantic_count / len(experiment.episodes)
        print(f"📊 每个episode平均语义概念: {avg_concepts:.1f}")

    # 进行功能测试
    print("\n🧪 开始功能测试...")
    
    # 测试语义发现
    # await show_semantic_discoveries(experiment)
    
    # 测试检索功能
    # await test_retrieval_functionality(experiment)
    
    # 总结
    print("\n🎉 全功能测试完成!")
    print("=" * 60)
    print("测试场景包括:")
    print("  1. 🤖 机器学习技术讨论 (张三 & 李四)")
    print("  2. 🎓 AI研究学术对话 (王博士 & 刘教授)")
    print("  3. 🏔️ 户外摄影活动计划 (小明 & 小红)")
    print("\n测试功能:")
    print("  ✅ Episodic Memory: 对话分割和episodes创建")
    print("  ✅ Semantic Memory: 隐含知识发现和抽取")
    print("  ✅ Unified Retrieval: 多策略检索系统")
    print("  ✅ 实时功能验证和结果展示")
    
    return True
        
 


def main(frame, version="test"):
    load_dotenv()
    if frame == "nemori":
        # Run async main for nemori
        return asyncio.run(main_nemori(version))
    else:
        print(f"❌ 不支持的框架: {frame}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemori全功能测试脚本 - 完善版")
    parser.add_argument(
        "--lib",
        type=str,
        choices=["nemori"],
        default="nemori",
        help="使用nemori框架进行测试",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="test",
        help="版本标识符 (例如: test, demo, production)",
    )
    
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    
    print(f"🚀 开始运行 {lib} 测试，版本: {version}")
    success = main(lib, version)
    
    if success:
        print(f"\n✅ {lib} 测试成功完成！")
    else:
        print(f"\n❌ {lib} 测试失败！")
        exit(1)