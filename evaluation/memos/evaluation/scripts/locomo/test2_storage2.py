import asyncio
from dotenv import load_dotenv

# 从我们刚刚创建的文件中导入 NemoriExperiment 类
from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy

# --- 定义我们的中文对话样例数据 ---
# 这是一个包含单个对话的列表
# 每个对话是一个字典，包含 conversation_id 和 messages 列表
my_conversations = [
    {
        "conversation": {
            "conversation_id": "beijing_trip_plan",
            "sessions": [  # <--- 添加 'sessions' 列表，这是关键
                {
                    # 这是一个 session 对象，它包含了消息列表
                    "messages": [  # <--- 所有的消息都放在这个 session 内部
                        {
                            "speaker": "小明",
                            "text": "你好啊，小红！最近在忙什么？",
                            "timestamp": "2024-05-20T10:00:00Z",
                        },
                        {
                            "speaker": "小红",
                            "text": "我最近在准备去北京旅游，正在看攻略呢。故宫和长城是必去的！",
                            "timestamp": "2024-05-20T10:00:30Z",
                        },
                        {
                            "speaker": "小明",
                            "text": "听起来不错！北京烤鸭也一定要尝尝。你打算什么时候去？",
                            "timestamp": "2024-05-20T10:01:00Z",
                        },
                        {
                            "speaker": "小红",
                            "text": "计划下个月，大概6月15号左右出发。希望能订到合适的机票和酒店。",
                            "timestamp": "2024-05-20T10:01:30Z",
                        },
                    ]
                }
            ],
        }
    }
]


async def main():
    """主执行函数"""
    load_dotenv()
    print("🚀 Starting Nemori Experiment...")
    print("==================================================")

    # 1. 初始化实验
    experiment = NemoriExperiment(version="my-first-run", episode_mode="speaker")

    # 2. 定义查询
    query_text = "小红的北京旅游计划是什么？"
    # 我们想从小红的视角来查询
    query_owner_name = "小红"
    llm_available = await experiment.setup_llm_provider()
    if not llm_available:
        print("⚠️ Continuing with fallback mode (no LLM)")

    # Step 2: Load data
    experiment.load_conversation_data(my_conversations)

    # Step 3: Setup storage and retrieval
    await experiment.setup_storage_and_retrieval()

    # Step 4: Build episodes
    await experiment.build_episodes()

    print("\n🎉 Nemori Ingestion Complete")
    print(f"✅ Successfully processed {len(experiment.conversations)} conversations")
    print(f"✅ Created {len(experiment.episodes)} episodes")

    print("\n🎉 Experiment finished.")


if __name__ == "__main__":

    asyncio.run(main())
