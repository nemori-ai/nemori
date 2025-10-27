import asyncio
from dotenv import load_dotenv

# 从我们刚刚创建的文件中导入 NemoriExperiment 类
from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy, RetrievalConfig

# --- 定义我们的中文对话样例数据 ---
# 这是一个包含单个对话的列表
# 每个对话是一个字典，包含 conversation_id 和 messages 列表
my_conversations = [
    {
        "conversation_id": "beijing_trip_plan",
        "messages": [
            {"speaker": "小明", "text": "你好啊，小红！最近在忙什么？", "timestamp": "2024-05-20T10:00:00Z"},
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
        ],
    }
]
my_conversations = [
    {
        "conversation_id": "beijing_trip_plan",
        "messages": [
            {"speaker": "小明", "text": "你好啊，小红！最近在忙什么？", "timestamp": "2024-05-20T10:00:00Z"},
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
        ],
    }
]


async def main():
    """主执行函数"""
    load_dotenv()
    print("🚀 Starting Nemori Experiment...")
    print("==================================================")

    # 1. 初始化实验
    retrieval_config = RetrievalConfig.from_env(strategy=RetrievalStrategy.EMBEDDING)
    experiment = NemoriExperiment(version="my-first-run", episode_mode="speaker", retrieval_config=retrieval_config)

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
    # Create tasks for all conversations
    tasks = [experiment.process_conversation(i, conv_data) for i, conv_data in enumerate(my_conversations)]
    print(f"   📋 Created {len(tasks)} concurrent tasks")

    # Wait for all tasks to complete
    print("   ⏳ Starting concurrent execution...")
    results = await asyncio.gather(*tasks)
    print("   🏁 All conversations completed")

    # Collect all episodes
    for _, episodes in results:
        experiment.episodes.extend(episodes)

    print("\n📊 Episode Building Complete")
    print(f"✅ Successfully created {len(experiment.episodes)} episodes")
    await experiment.build_bm25_indices()
    # Step 4: Build episodes
    # await experiment.build_episodes()

    print("\n🎉 Nemori Ingestion Complete")
    print(f"✅ Successfully processed {len(experiment.conversations)} conversations")
    print(f"✅ Created {len(experiment.episodes)} episodes")
    for i in experiment.episodes:
        print(i.owner_id)
    print("\n🎉 Experiment finished.")
    # --- Part 2: Querying ---
    conversation_id = my_conversations[0]["conversation_id"]
    # Query 1: From 小红's perspective
    query_owner_id_xiao_hong = f"{query_owner_name}_{conversation_id}"
    print("\n" + "=" * 50)
    print(f"🔍 Running Query from '{query_owner_id_xiao_hong}' perspective")
    print(f"   ❓ Query: '{query_text}'")
    print("=" * 50)

    # 1. Create a retrieval query object
    retrieval_query = RetrievalQuery(
        text=query_text,
        owner_id=query_owner_id_xiao_hong,
        limit=25,  # Retrieve top 5 relevant episodes
        strategy=RetrievalStrategy.EMBEDDING,
    )

    # 2. Perform the search
    result_a = await experiment.retrieval_service.search(retrieval_query)

    print(result_a)
    # --- FIX 2: Get the length of the .results list ---
    print(f"   ✅ Found {len(result_a.episodes)} relevant episode(s):")

    # 3. Display the results
    for i, episode in enumerate(result_a.episodes[:2]):
        print(f"     {i+1}. Title: '{episode.title}'")
        print(f"        Content: '{episode.content[:100]}...'")
        print(f"        Summary: '{episode.summary}'")


if __name__ == "__main__":

    asyncio.run(main())
