import asyncio
from dotenv import load_dotenv
from sklearn import base

# 从我们刚刚创建的文件中导入 NemoriExperiment 类
from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy,RetrievalConfig

# --- 定义我们的中文对话样例数据 ---
# 这是一个包含单个对话的列表
# 每个对话是一个字典，包含 conversation_id 和 messages 列表
my_conversations = [
   {
    "user_id": "123456",
    "conversation": {
    "speaker_a": "小明",
    "speaker_b": "小红",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
            {
                "speaker": "小明",
                "text": "你好啊，小红！最近在忙什么？",
                "timestamp": "2024-05-20T10:00:00Z"
            },
            {
                "speaker": "小红",
                "text": "我最近在准备去北京旅游，正在看攻略呢。故宫和长城是必去的！",
                "timestamp": "2024-05-20T10:00:30Z"
            },
            {
                "speaker": "小明",
                "text": "听起来不错！北京烤鸭也一定要尝尝。你打算什么时候去？",
                "timestamp": "2024-05-20T10:01:00Z"
            },
            {
                "speaker": "小红",
                "text": "计划下个月，大概6月15号左右出发。希望能订到合适的机票和酒店。",
                "timestamp": "2024-05-20T10:01:30Z"
            },
    ]
   }}
]
async def main():
    """主执行函数"""
    load_dotenv()
    print("🚀 Starting Nemori Retrieval Test...")
    print("==================================================")

    # 1. 初始化实验
    experiment = NemoriExperiment(version="my-first-run", episode_mode="speaker", retrievalstrategy = RetrievalStrategy.EMBEDDING)

    # 2. 定义查询
    query_text = "小红的北京旅游计划是什么？"
    query_owner_name = "小红"
    
    api_key = "EMPTY"
    base_url = "http://localhost:6003/v1"
    model = "bce-emb"
    # 3. 设置存储和检索
    await experiment.setup_storage_and_retrieval(emb_api_key=api_key,emb_base_url=base_url,embed_model=model, from_exist=True)

    print("
🔍 Performing Retrieval...")
    
    conversation_id = "123456"
    query_owner_id_xiao_hong = f"{query_owner_name}_{conversation_id}"
    print("
" + "="*50)
    print(f"🔍 Running Query from '{query_owner_id_xiao_hong}' perspective")
    print(f"   ❓ Query: '{query_text}'")
    print("="*50)

    # 4. 创建检索查询对象
    retrieval_query = RetrievalQuery(
        text=query_text,
        owner_id=query_owner_id_xiao_hong,
        limit=5,  # 检索前5个最相关的片段
        strategy=RetrievalStrategy.EMBEDDING,
    )

    # 5. 执行搜索
    result = await experiment.retrieval_service.search(retrieval_query)
    
    print(f"   ✅ Found {len(result.episodes)} relevant episode(s):")
    
    # 6. 显示结果
    for i, episode in enumerate(result.episodes):
        print(f"     {i+1}. Title: '{episode.title}'")
        print(f"        Content: '{episode.content[:100]}...'")
        print(f"        Summary: '{episode.summary}'")

    print("
🎉 Retrieval test finished.")

if __name__ == "__main__":
    asyncio.run(main())
