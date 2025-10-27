import asyncio
from dotenv import load_dotenv

# 从我们刚刚创建的文件中导入 NemoriExperiment 类
from nemori_eval.experiment import NemoriExperiment, RetrievalQuery, RetrievalStrategy


async def main():
    """主执行函数"""
    load_dotenv()
    print("🚀 Starting Nemori Retrieval Test...")
    print("==================================================")

    # 1. 初始化实验
    # 使用与 test2_storage.py 中相同的版本，以确保加载正确的数据
    experiment = NemoriExperiment(
        version="default", episode_mode="speaker", retrievalstrategy=RetrievalStrategy.EMBEDDING
    )

    # 可选：为检索过程中的任何潜在的基于 LLM 的处理设置 LLM
    llm_api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
    llm_base_url = "https://jeniya.cn/v1"
    llm_model = "gpt-4o-mini"
    llm_available = await experiment.setup_llm_provider(model=llm_model, api_key=llm_api_key, base_url=llm_base_url)
    if not llm_available:
        print("⚠️ Continuing with fallback mode (no LLM)")

    # 2. 设置存储和检索服务
    # 这将根据实验版本连接到现有存储
    api_key = "EMPTY"
    base_url = "http://localhost:6007/v1"
    model = "qwen3-emb"
    await experiment.setup_storage_and_retrieval(emb_api_key=api_key, emb_base_url=base_url, embed_model=model)

    print("✅ Storage and retrieval services are ready.")

    # --- Part 2: 查询 ---
    query_text = "Audrey reached out to Andrew after a long time, sharing her recent experience of taking her pets, referred to as her 'fur kids"
    query_owner_name = "andrew"
    conversation_id = "5"  # 这必须与数据注入时使用的 ID 匹配

    # 为查询构建 owner_id
    query_owner_id_xiao_hong = f"{query_owner_name}_{conversation_id}"

    print("" + "=" * 50)
    print(f"🔍 Running Query from '{query_owner_id_xiao_hong}' perspective")
    print(f"   ❓ Query: '{query_text}'")
    print("=" * 50)

    # 1. 创建 RetrievalQuery 对象
    retrieval_query = RetrievalQuery(
        text=query_text,
        owner_id=query_owner_id_xiao_hong,
        limit=5,  # 检索最相关的 5 个片段
        strategy=RetrievalStrategy.EMBEDDING,
    )

    # 2. 执行搜索
    result = await experiment.retrieval_service.search(retrieval_query)

    # 3. 显示结果
    if result and result.episodes:
        print(f"   ✅ Found {len(result.episodes)} relevant episode(s):")
        for i, episode in enumerate(result.episodes[:2]):  # 显示前 2 个结果
            Title = episode["title"]
            Content = episode["content"]
            Summary = episode["summary"]
            print(f"     {i+1}. Title: '{Title}'")
            print(f"        Content: '{Content}...'")
            print(f"        Summary: '{Summary}'")
    else:
        print("   ❌ No relevant episodes found for the query.")

    print("🎉 Retrieval test finished.")


if __name__ == "__main__":
    asyncio.run(main())
