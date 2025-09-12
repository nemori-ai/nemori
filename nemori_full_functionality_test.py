#!/usr/bin/env python3
"""
Nemori全功能测试脚本

这个脚本创建一个包含多种场景的对话数据样例，用于测试nemori的全部功能：
- Episodic Memory: 对话片段的分割和记忆
- Semantic Memory: 隐含知识的发现和抽取
- Unified Retrieval: 统一的检索系统

测试场景包括：
1. 技术讨论 - 软件开发和技术选型
2. 学术研究 - 论文写作和研究方向
3. 生活日常 - 日常活动和个人兴趣
4. 工作协作 - 项目管理和团队协作
5. 专业知识 - 机器学习和人工智能
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Import nemori components
import sys
from pathlib import Path

# 添加路径以导入nemori模块
current_dir = Path(__file__).parent
nemori_root = current_dir / "nemori" 
sys.path.insert(0, str(nemori_root))
sys.path.insert(0, str(current_dir / "evaluation/memos/evaluation/scripts/locomo"))
from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
from nemori.retrieval import RetrievalQuery


def create_comprehensive_test_data():
    """创建包含多种场景的综合测试数据"""
    
    # 定义不同的对话场景
    conversations = [
        {
            "user_id": "tech_discussion_001",
            "conversation": {
                "speaker_a": "张三",
                "speaker_b": "李四",
                "session_1": [
                    {
                        "speaker": "张三",
                        "text": "最近我们项目需要选择前端框架，你觉得React和Vue哪个更适合？",
                        "timestamp": "2024-01-15T09:00:00Z"
                    },
                    {
                        "speaker": "李四", 
                        "text": "这要看具体需求。React的生态系统更成熟，但Vue的学习曲线更平缓。我们的团队大部分人对JavaScript基础比较扎实。",
                        "timestamp": "2024-01-15T09:02:00Z"
                    },
                    {
                        "speaker": "张三",
                        "text": "确实，我考虑到我们需要开发一个数据可视化的Dashboard，性能要求比较高。",
                        "timestamp": "2024-01-15T09:04:00Z"
                    },
                    {
                        "speaker": "李四",
                        "text": "那建议用React配合D3.js，React的虚拟DOM在处理大量数据更新时表现很好。我之前用这套方案做过类似项目。",
                        "timestamp": "2024-01-15T09:06:00Z"
                    }
                ],
                "session_1_date_time": "9:00 AM on 15 January, 2024",
                "session_2": [
                    {
                        "speaker": "张三",
                        "text": "好的，那状态管理你推荐用什么？Redux还是Context API？",
                        "timestamp": "2024-01-15T09:30:00Z"
                    },
                    {
                        "speaker": "李四",
                        "text": "对于中等规模的项目，我建议用Zustand，比Redux简单很多，而且TypeScript支持很好。",
                        "timestamp": "2024-01-15T09:32:00Z"
                    },
                    {
                        "speaker": "张三", 
                        "text": "听起来不错，我去研究一下Zustand的文档。另外，我们还需要考虑国际化的需求。",
                        "timestamp": "2024-01-15T09:34:00Z"
                    }
                ],
                "session_2_date_time": "9:30 AM on 15 January, 2024"
            }
        },
        
        {
            "user_id": "academic_research_002", 
            "conversation": {
                "speaker_a": "王教授",
                "speaker_b": "刘博士",
                "session_1": [
                    {
                        "speaker": "王教授",
                        "text": "我们的论文关于大语言模型的对齐研究进展如何？",
                        "timestamp": "2024-01-16T14:00:00Z"
                    },
                    {
                        "speaker": "刘博士",
                        "text": "目前我们已经完成了RLHF的基础实验，发现在数学推理任务上提升了15%的准确率。",
                        "timestamp": "2024-01-16T14:02:00Z"
                    },
                    {
                        "speaker": "王教授",
                        "text": "很好！那Constitutional AI的对比实验呢？我记得你提到过这个方向。",
                        "timestamp": "2024-01-16T14:04:00Z"
                    },
                    {
                        "speaker": "刘博士",
                        "text": "Constitutional AI在安全性评估上表现更好，特别是在避免有害输出方面。我们可以结合两种方法。",
                        "timestamp": "2024-01-16T14:06:00Z"
                    }
                ],
                "session_1_date_time": "2:00 PM on 16 January, 2024",
                "session_2": [
                    {
                        "speaker": "王教授",
                        "text": "论文投稿的话，你觉得ICML和NeurIPS哪个更合适？",
                        "timestamp": "2024-01-16T15:00:00Z"
                    },
                    {
                        "speaker": "刘博士",
                        "text": "考虑到我们的工作偏向应用，ICML可能更合适。而且ICML今年对AI安全的议题比较重视。",
                        "timestamp": "2024-01-16T15:02:00Z"
                    }
                ],
                "session_2_date_time": "3:00 PM on 16 January, 2024"
            }
        },
        
        {
            "user_id": "daily_life_003",
            "conversation": {
                "speaker_a": "小明",
                "speaker_b": "小红", 
                "session_1": [
                    {
                        "speaker": "小明",
                        "text": "周末你有什么计划吗？我想去爬山。",
                        "timestamp": "2024-01-17T18:00:00Z"
                    },
                    {
                        "speaker": "小红",
                        "text": "爬山听起来不错！我最近在学摄影，可以带相机去拍风景。",
                        "timestamp": "2024-01-17T18:02:00Z"
                    },
                    {
                        "speaker": "小明",
                        "text": "太好了！我知道一个山上有很美的日出，早上6点日出，4点半就要出发。",
                        "timestamp": "2024-01-17T18:04:00Z"
                    },
                    {
                        "speaker": "小红",
                        "text": "好的，我最近在练习长焦镜头的使用，正好可以拍日出。你会做饭吗？我们可以带点简单的早餐。",
                        "timestamp": "2024-01-17T18:06:00Z"
                    }
                ],
                "session_1_date_time": "6:00 PM on 17 January, 2024",
                "session_2": [
                    {
                        "speaker": "小明",
                        "text": "我会做简单的三明治，你负责咖啡怎么样？我知道你是咖啡爱好者。",
                        "timestamp": "2024-01-17T19:00:00Z"
                    },
                    {
                        "speaker": "小红",
                        "text": "没问题！我最近在研究手冲咖啡，用V60滤杯，可以带便携设备上山。",
                        "timestamp": "2024-01-17T19:02:00Z"
                    }
                ],
                "session_2_date_time": "7:00 PM on 17 January, 2024"
            }
        },
        
        {
            "user_id": "work_collaboration_004",
            "conversation": {
                "speaker_a": "项目经理",
                "speaker_b": "开发工程师",
                "session_1": [
                    {
                        "speaker": "项目经理",
                        "text": "我们的AI推荐系统项目进度如何？客户希望下个月上线。",
                        "timestamp": "2024-01-18T10:00:00Z"
                    },
                    {
                        "speaker": "开发工程师",
                        "text": "核心算法已经完成，我们使用了collaborative filtering配合deep learning embedding。目前准确率达到了85%。",
                        "timestamp": "2024-01-18T10:02:00Z"
                    },
                    {
                        "speaker": "项目经理",
                        "text": "很好！那A/B测试的结果呢？用户体验有提升吗？",
                        "timestamp": "2024-01-18T10:04:00Z"
                    },
                    {
                        "speaker": "开发工程师",
                        "text": "A/B测试显示点击率提升了23%，用户停留时间增加了18%。不过冷启动问题还需要优化。",
                        "timestamp": "2024-01-18T10:06:00Z"
                    }
                ],
                "session_1_date_time": "10:00 AM on 18 January, 2024",
                "session_2": [
                    {
                        "speaker": "项目经理",
                        "text": "冷启动问题确实重要。新用户的推荐准确率怎么样？",
                        "timestamp": "2024-01-18T14:00:00Z"
                    },
                    {
                        "speaker": "开发工程师",
                        "text": "我们采用了content-based filtering作为补充，结合用户的基础信息和热门内容。新用户的转化率达到了12%。",
                        "timestamp": "2024-01-18T14:02:00Z"
                    }
                ],
                "session_2_date_time": "2:00 PM on 18 January, 2024"
            }
        },
        
        {
            "user_id": "ml_knowledge_005",
            "conversation": {
                "speaker_a": "数据科学家",
                "speaker_b": "ML工程师",
                "session_1": [
                    {
                        "speaker": "数据科学家",
                        "text": "我们的模型在处理imbalanced dataset时表现不好，你有什么建议？",
                        "timestamp": "2024-01-19T09:00:00Z"
                    },
                    {
                        "speaker": "ML工程师",
                        "text": "可以尝试SMOTE进行数据增强，或者调整class weights。我之前用focal loss在类似问题上效果很好。",
                        "timestamp": "2024-01-19T09:02:00Z"
                    },
                    {
                        "speaker": "数据科学家",
                        "text": "Focal loss听起来不错，是在object detection中用的那个吗？",
                        "timestamp": "2024-01-19T09:04:00Z"
                    },
                    {
                        "speaker": "ML工程师",
                        "text": "对的，它可以动态调整难易样本的权重。对于我们的二分类问题，可以让模型更关注困难的minority class。",
                        "timestamp": "2024-01-19T09:06:00Z"
                    }
                ],
                "session_1_date_time": "9:00 AM on 19 January, 2024",
                "session_2": [
                    {
                        "speaker": "数据科学家",
                        "text": "那evaluation metrics呢？accuracy明显不适用，你推荐用什么？",
                        "timestamp": "2024-01-19T10:00:00Z"
                    },
                    {
                        "speaker": "ML工程师",
                        "text": "建议用precision, recall和F1-score的组合，或者直接用AUC-ROC。我们还可以看confusion matrix来分析具体的错误类型。",
                        "timestamp": "2024-01-19T10:02:00Z"
                    }
                ],
                "session_2_date_time": "10:00 AM on 19 January, 2024"
            }
        }
    ]
    
    # 将数据转换为DataFrame格式，兼容现有的加载方式
    return pd.DataFrame(conversations)


async def test_episodic_memory(experiment):
    """测试episodic memory功能"""
    print("\n🧠 测试 Episodic Memory 功能")
    print("=" * 60)
    
    # 检查创建的episodes
    if experiment.episodes:
        print(f"✅ 成功创建 {len(experiment.episodes)} 个episodes")
        
        # 显示一些episode的详细信息
        for i, episode in enumerate(experiment.episodes[:3]):  # 只显示前3个
            print(f"\n📝 Episode {i+1}: {episode.episode_id}")
            print(f"   👤 Owner: {episode.owner_id}")
            print(f"   🕐 时间: {episode.temporal_info.timestamp}")
            print(f"   📄 内容长度: {len(episode.content)} characters")
            print(f"   🏷️ 标签: {episode.tags}")
            
            # 显示episode的部分内容
            content_preview = episode.content[:200] + "..." if len(episode.content) > 200 else episode.content
            print(f"   📖 内容预览: {content_preview}")
    else:
        print("❌ 没有创建任何episodes")


async def test_semantic_memory(experiment):
    """测试semantic memory功能"""
    print("\n🔍 测试 Semantic Memory 功能")  
    print("=" * 60)
    
    if not experiment.semantic_repo:
        print("❌ Semantic repository 未初始化")
        return
        
    # 获取所有owners的semantic knowledge
    owner_ids = {episode.owner_id for episode in experiment.episodes} if experiment.episodes else set()
    
    total_concepts = 0
    for owner_id in owner_ids:
        try:
            semantic_nodes = await experiment.semantic_repo.get_all_semantic_nodes_for_owner(owner_id)
            if semantic_nodes:
                print(f"\n👤 {owner_id} 的语义知识:")
                for i, node in enumerate(semantic_nodes[:3]):  # 只显示前3个
                    print(f"   🔑 {node.key}: {node.value}")
                    print(f"   🎯 信心度: {node.confidence:.2f}")
                    print(f"   📝 上下文: {node.context[:100]}...")
                    print(f"   🔗 关联episodes: {len(node.linked_episode_ids)}")
                    print()
                total_concepts += len(semantic_nodes)
                if len(semantic_nodes) > 3:
                    print(f"   ... 还有 {len(semantic_nodes) - 3} 个概念")
            else:
                print(f"\n👤 {owner_id}: 未发现语义知识")
        except Exception as e:
            print(f"\n❌ 获取 {owner_id} 的语义知识时出错: {e}")
    
    print(f"\n📊 总共发现 {total_concepts} 个语义概念")


async def test_unified_retrieval(experiment):
    """测试unified retrieval功能"""
    print("\n🔍 测试 Unified Retrieval 功能")
    print("=" * 60)
    
    if not experiment.retrieval_service:
        print("❌ Retrieval service 未初始化")
        return
        
    # 定义测试查询
    test_queries = [
        "React和Vue的选择",
        "大语言模型对齐研究", 
        "摄影和爬山",
        "推荐系统算法",
        "imbalanced dataset处理"
    ]
    
    # 获取一个测试用的owner_id
    owner_ids = list({episode.owner_id for episode in experiment.episodes}) if experiment.episodes else []
    if not owner_ids:
        print("❌ 没有可用的owner_id进行检索测试")
        return
        
    test_owner = owner_ids[0]
    print(f"🎯 使用owner_id: {test_owner}")
    
    for query_text in test_queries:
        print(f"\n🔍 查询: '{query_text}'")
        
        try:
            # 创建检索查询
            query = RetrievalQuery(
                text=query_text,
                owner_id=test_owner,
                limit=3,
                strategy=experiment.retrievalstrategy
            )
            
            # 执行检索
            results = await experiment.retrieval_service.search(query)
            
            if results and results.episodes:
                print(f"   ✅ 找到 {len(results.episodes)} 个相关episodes:")
                for i, result in enumerate(results.episodes[:2]):  # 只显示前2个
                    print(f"      📄 {i+1}. Episode ID: {result.episode_id}")
                    print(f"         👤 Owner: {result.owner_id}")
                    print(f"         📊 相关度分数: {getattr(result, 'score', 'N/A')}")
                    content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
                    print(f"         📖 内容: {content_preview}")
                    print()
            else:
                print("   ❌ 未找到相关结果")
                
        except Exception as e:
            print(f"   ❌ 检索出错: {e}")


async def test_search_functionality(experiment):
    """测试搜索功能的综合性能"""
    print("\n🎯 测试搜索功能综合性能")
    print("=" * 60)
    
    # 针对不同owner进行搜索测试
    owner_ids = list({episode.owner_id for episode in experiment.episodes}) if experiment.episodes else []
    
    for owner_id in owner_ids[:3]:  # 只测试前3个owner
        print(f"\n👤 测试 Owner: {owner_id}")
        
        # 获取该owner的episode信息
        try:
            result = await experiment.episode_repo.get_episodes_by_owner(owner_id)
            owner_episodes = result.episodes if hasattr(result, "episodes") else result
            print(f"   📊 该用户共有 {len(owner_episodes)} 个episodes")
            
            # 针对该用户的内容进行相关搜索
            if owner_episodes:
                # 从第一个episode中提取关键词进行搜索
                first_episode = owner_episodes[0]
                # 简单提取前几个词作为搜索关键词
                words = first_episode.content.split()[:3]
                search_query = " ".join(words)
                
                print(f"   🔍 使用关键词搜索: '{search_query}'")
                
                query = RetrievalQuery(
                    text=search_query,
                    owner_id=owner_id,
                    limit=5,
                    strategy=experiment.retrievalstrategy
                )
                
                search_results = await experiment.retrieval_service.search(query)
                
                if search_results and search_results.episodes:
                    print(f"   ✅ 找到 {len(search_results.episodes)} 个相关episodes")
                    print(f"   📈 搜索召回率: {len(search_results.episodes)}/{len(owner_episodes)} = {len(search_results.episodes)/len(owner_episodes)*100:.1f}%")
                else:
                    print("   ❌ 搜索未返回结果")
                    
        except Exception as e:
            print(f"   ❌ 搜索测试出错: {e}")


async def main():
    """主测试函数"""
    print("🚀 Nemori 全功能测试")
    print("=" * 80)
    print("测试场景包括:")
    print("  1. 技术讨论 - 前端框架选择和技术栈")
    print("  2. 学术研究 - AI模型对齐和论文投稿") 
    print("  3. 日常生活 - 爬山摄影和兴趣爱好")
    print("  4. 工作协作 - AI推荐系统项目管理")
    print("  5. 专业知识 - 机器学习算法和评估指标")
    print("=" * 80)
    
    # 加载环境变量
    load_dotenv()
    
    # 创建测试数据
    print("\n📋 创建测试数据...")
    test_data = create_comprehensive_test_data()
    print(f"✅ 创建了 {len(test_data)} 个对话场景")
    
    # 初始化nemori实验
    print("\n🔧 初始化 Nemori 实验...")
    experiment = NemoriExperiment(
        version="full_test",
        episode_mode="speaker", 
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        max_concurrency=1  # 串行处理，便于观察
    )
    
    try:
        # Step 1: 设置LLM provider
        print("\n🤖 设置 LLM Provider...")
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"  
        model = "gpt-4o-mini"
        
        llm_available = await experiment.setup_llm_provider(
            model=model, 
            api_key=api_key, 
            base_url=base_url
        )
        
        if not llm_available:
            print("⚠️ LLM连接失败，将影响语义发现功能")
            return
            
        # Step 2: 加载测试数据
        print("\n📊 加载测试数据...")
        experiment.load_locomo_data(test_data)
        
        # Step 3: 设置存储和检索
        print("\n🗄️ 设置存储和检索服务...")
        emb_api_key = "EMPTY"
        emb_base_url = "http://localhost:6007/v1"
        emb_model = "qwen3-emb"
        
        await experiment.setup_storage_and_retrieval(
            emb_api_key=emb_api_key,
            emb_base_url=emb_base_url, 
            embed_model=emb_model
        )
        
        # Step 4: 构建episodes和语义记忆
        print("\n🏗️ 构建Episodes和语义记忆...")
        await experiment.build_episodes_semantic()
        
        # 基础统计信息
        print("\n📊 处理结果统计:")
        print(f"✅ 处理了 {len(experiment.conversations)} 个对话")
        print(f"✅ 创建了 {len(experiment.episodes)} 个episodes")
        semantic_count = getattr(experiment, 'actual_semantic_count', 0)
        print(f"✅ 发现了 {semantic_count} 个语义概念")
        
        if semantic_count > 0 and len(experiment.episodes) > 0:
            print(f"📈 平均每个episode发现语义概念: {semantic_count/len(experiment.episodes):.1f}")
        
        # 运行功能测试
        await test_episodic_memory(experiment)
        await test_semantic_memory(experiment) 
        await test_unified_retrieval(experiment)
        await test_search_functionality(experiment)
        
        print("\n🎉 全功能测试完成!")
        print("=" * 80)
        print("测试总结:")
        print(f"📝 Episodic Memory: {len(experiment.episodes)} episodes")
        print(f"🧠 Semantic Memory: {semantic_count} concepts")
        print(f"🔍 Retrieval Strategy: {experiment.retrievalstrategy.value}")
        print(f"💾 存储位置: {experiment.db_dir}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        print("\n🧹 清理资源...")
        if hasattr(experiment, 'cleanup'):
            await experiment.cleanup()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())