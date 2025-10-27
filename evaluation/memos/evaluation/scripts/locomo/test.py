import asyncio
import os
from pathlib import Path
import shutil
from datetime import datetime

from dotenv import load_dotenv

# --- Nemori 核心组件导入 ---
from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)

# [新增] 导入 DuckDBRawDataRepository
from nemori.storage.duckdb_storage import (
    DuckDBEpisodicMemoryRepository,
    DuckDBRawDataRepository,
)
from nemori.storage.storage_types import StorageConfig

# --- 1. 定义中文对话样例数据 (与之前相同) ---
sample_conversation = [
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
]


# --- 2. 辅助函数：设置 Nemori 核心组件 ---
async def setup_nemori_components(db_dir: Path, llm_provider: OpenAIProvider):
    """一个独立的函数，用于初始化所有 Nemori 服务"""
    print("\n🗄️ 正在设置存储和检索服务...")

    if db_dir.exists():
        shutil.rmtree(db_dir)
        print(f"🧹 已清理旧的数据库目录: {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "memory_example.duckdb"

    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
    )

    # [新增] 初始化原始数据仓库
    raw_data_repo = DuckDBRawDataRepository(storage_config)
    await raw_data_repo.initialize()
    print("✅ 原始数据仓库 (RawDataRepository) 已初始化")

    # 初始化情节记忆仓库
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()
    print(f"✅ DuckDB 情节存储已初始化: {db_path}")

    # 设置检索服务 (与之前相同)
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(db_dir)},
    )
    retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)
    await retrieval_service.initialize()
    print("✅ BM25 检索服务已配置")

    # 设置情节构建器 (与之前相同)
    builder_registry = EpisodeBuilderRegistry()
    conversation_builder = ConversationEpisodeBuilder(llm_provider=llm_provider)
    builder_registry.register(conversation_builder)

    # [修改] 将 raw_data_repo 实例传递给 EpisodeManager
    episode_manager = EpisodeManager(
        raw_data_repo=raw_data_repo,
        episode_repo=episode_repo,
        builder_registry=builder_registry,
        retrieval_service=retrieval_service,
    )
    print("✅ 情节管理器已初始化 (包含原始数据存储功能)")

    # [修改] 返回所有需要的组件
    return episode_manager, retrieval_service, episode_repo, raw_data_repo


# --- 3. 辅助函数：将样例数据转换为 Nemori 格式 (与之前相同) ---
def convert_to_raw_event_data(conversation: list, conversation_id: str) -> RawEventData:
    messages = []
    for msg in conversation:
        speaker_name = msg["speaker"]
        speaker_id = f"{'xiaoming' if speaker_name == '小明' else 'xiaohong'}_{conversation_id}"

        # [!!!] 在这里就处理好时间戳字符串 [!!!]
        # 将 'Z' 替换为 '+00:00'，以兼容 Python 3.10
        iso_timestamp = msg["timestamp"].replace("Z", "+00:00")

        messages.append(
            {
                "speaker_id": speaker_id,
                "user_name": speaker_name,
                "content": msg["text"],
                # 使用处理后的时间戳
                "timestamp": iso_timestamp,
            }
        )

    # 现在这里的 fromisoformat 也会正确工作，因为 messages 里的 timestamp 已经修复了
    first_timestamp = datetime.fromisoformat(messages[0]["timestamp"])
    # x = messages[0]["timestamp"]
    # print(f"{type(x)},{x} datetime.fromisoformat-> {type(first_timestamp)},{first_timestamp}")
    last_timestamp = datetime.fromisoformat(messages[-1]["timestamp"])
    # x = messages[-1]["timestamp"]
    # print(f"{type(x)},{x} datetime.fromisoformat-> {type(last_timestamp)},{last_timestamp}")
    duration = (last_timestamp - first_timestamp).total_seconds()
    # print(f"duration, {duration}")
    return RawEventData(
        data_type=DataType.CONVERSATION,
        content=messages,
        source="example_script_with_raw_data",
        temporal_info=TemporalInfo(timestamp=first_timestamp, duration=duration, timezone="UTC"),
        metadata={"conversation_id": conversation_id},
    )


async def main():
    """主执行函数"""
    load_dotenv()
    print("🚀 开始 Nemori 储存与检索示例 (包含原始数据存储)")
    print("=" * 50)

    DB_DIR = Path("nemori_example_storage")

    # --- 步骤 1: 设置 LLM 提供者 (与之前相同) ---
    print("\n🤖 正在设置 LLM Provider...")
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 未找到 OPENAI_API_KEY 环境变量。")
        return
    llm_provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.1)
    if not await llm_provider.test_connection():
        print("❌ OpenAI 连接失败!")
        return
    print(f"✅ OpenAI 连接成功! 模型: {llm_provider.model}")

    # --- 步骤 2: 设置存储、检索和情节管理器 ---
    # [修改] 接收新增的 raw_data_repo
    episode_manager, retrieval_service, episode_repo, raw_data_repo = await setup_nemori_components(
        DB_DIR, llm_provider
    )

    # --- 步骤 3: 转换并处理数据以构建情节 ---
    print("\n🏗️ 正在构建情节记忆 (并自动存储原始数据)...")
    conversation_id = "conv_beijing_trip"
    raw_data_to_ingest = convert_to_raw_event_data(sample_conversation, conversation_id)

    owner_ids = {"xiaoming_conv_beijing_trip", "xiaohong_conv_beijing_trip"}
    created_episodes = []
    for owner_id in owner_ids:
        print(f"   为所有者 '{owner_id}' 处理数据...")
        # process_raw_data 可能返回一个 Episode 对象或 None
        episode = await episode_manager.process_raw_data_to_episode(raw_data_to_ingest, owner_id=owner_id)

        # 检查返回的是否是一个有效的 Episode 对象
        # if episode:
        # 使用 .append() 将单个 Episode 对象添加到列表中
        created_episodes.append(episode)
        # 相应的，打印语句也要修改，因为我们一次只处理一个情节
        print(f"   ✅ 成功为 '{owner_id}' 创建并存储了 1 个情节。")

    print(f"\n📊 总共创建了 {len(created_episodes)} 个情节。")

    # --- [新增] 步骤 4: 验证原始数据已存储 ---
    print("\n🔎 正在验证原始数据是否已成功存储...")
    if created_episodes:
        # 每个情节都包含其来源的原始数据的 ID
        first_episode = created_episodes[0]
        raw_data_id = first_episode.episode_id

        print(f"   从第一个情节中获取到 raw_data_id: {raw_data_id}")

        # 使用 ID 从原始数据仓库中取回数据
        retrieved_raw_data = await raw_data_repo.get_raw_data(raw_data_id)

        if retrieved_raw_data:
            print("   ✅ 验证成功！已从数据库中取回原始数据。")
            print(f"      - 数据来源 (Source): {retrieved_raw_data.source}")
            print(f"      - 消息数量 (Message count): {len(retrieved_raw_data.content)}")
        else:
            print("   ❌ 验证失败！未能取回原始数据。")
    else:
        print("   ⚠️ 未创建任何情节，无法进行验证。")

    # --- 步骤 5: 构建 BM25 检索引索 (与之前相同) ---
    print("\n🔧 正在构建 BM25 检索引索...")
    for owner_id in owner_ids:
        dummy_query = RetrievalQuery(text=".", owner_id=owner_id, limit=1, strategy=RetrievalStrategy.BM25)
        await retrieval_service.search(dummy_query)
        print(f"   ✅ 已为所有者 '{owner_id}' 触发索引构建。")
    print("✅ 检索引索构建完成。")

    # --- 步骤 6: 执行检索查询 (与之前相同) ---
    print("\n🔍 正在执行检索查询...")
    query_text = "小红去北京旅游的计划是什么？"
    query_owner_id = "xiaohong_conv_beijing_trip"
    retrieval_query = RetrievalQuery(text=query_text, owner_id=query_owner_id, limit=5, strategy=RetrievalStrategy.BM25)
    search_results = await retrieval_service.search(retrieval_query)

    print(f"\n--- 检索结果 for query: '{query_text}' (所有者: {query_owner_id}) ---")
    if search_results.episodes:
        for i, episode in enumerate(search_results.episodes):
            print(f"\n[结果 {i+1}] (分数: {episode.score:.4f})")
            print(f"  - 摘要: {episode.summary}")
    else:
        print("   未找到相关记忆。")
    print("----------------------------------------------------------")

    # --- 步骤 7: 清理资源 ---
    print("\n🧹 正在清理资源...")
    await retrieval_service.close()
    await episode_repo.close()
    await raw_data_repo.close()  # [新增] 关闭原始数据仓库连接
    print("✅ 清理完成。")


if __name__ == "__main__":
    asyncio.run(main())
