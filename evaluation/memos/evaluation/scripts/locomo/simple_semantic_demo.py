"""
Enhanced Semantic Memory Demo for Nemori Framework

This demo showcases the semantic memory functionality integrated with episodic memory,
using a proper end-to-end workflow similar to the evaluation system.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo, SemanticNode
from nemori.core.episode import Episode, EpisodeType, EpisodeLevel
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBSemanticMemoryRepository, DuckDBEpisodicMemoryRepository
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder
from nemori.llm.providers.openai_provider import OpenAIProvider

experiment = NemoriExperiment(version=version, episode_mode="speaker", retrievalstrategy=RetrievalStrategy.EMBEDDING)


async def create_demo_conversation():
    """Create demo conversation data in Nemori format."""
    # Create conversation messages
    messages = [
        {
            "speaker_id": "john_conv1",
            "user_name": "John",
            "content": "我最近在研究AI Agent的行为规划",
            "timestamp": "2024-01-15T10:01:00",
            "metadata": {},
        },
        {
            "speaker_id": "user_conv1",
            "user_name": "User",
            "content": "这个方向很有前景",
            "timestamp": "2024-01-15T10:02:00",
            "metadata": {},
        },
        {
            "speaker_id": "john_conv1",
            "user_name": "John",
            "content": "是的，特别是决策机制这块很有挑战性",
            "timestamp": "2024-01-15T10:03:00",
            "metadata": {},
        },
    ]

    # Create RawEventData
    raw_data = RawEventData(
        data_type=DataType.CONVERSATION,
        content=messages,
        source="demo_conversation",
        temporal_info=TemporalInfo(
            timestamp=datetime.fromisoformat("2024-01-15T10:01:00"), duration=120.0, timezone="UTC"  # 2 minutes
        ),
        metadata={"conversation_id": "conv1", "participant_count": 2},
    )

    return raw_data


async def test_end_to_end_semantic_memory():
    """Test end-to-end semantic memory workflow with episodic integration."""
    print("🚀 End-to-End 语义记忆演示 | Semantic Memory Demo")
    print("=" * 60)

    # Setup storage
    print("\n🔧 初始化存储 | Initialize Storage...")
    db_dir = Path("demo_storage")
    db_dir.mkdir(exist_ok=True)

    # Clean up existing database
    db_path = db_dir / "semantic_demo.duckdb"
    if db_path.exists():
        db_path.unlink()
        print("🧹 Cleaned existing database")

    # Initialize storage
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=True,
    )

    episodic_repo = DuckDBEpisodicMemoryRepository(storage_config)
    semantic_repo = DuckDBSemanticMemoryRepository(storage_config)

    await episodic_repo.initialize()
    await semantic_repo.initialize()

    print("✅ 存储初始化完成 | Storage initialized")

    # Setup LLM provider
    print("\n🤖 初始化LLM提供者 | Initialize LLM Provider...")
    mock_llm = MockLLMProvider()
    print("✅ Mock LLM provider initialized")

    # Setup semantic components
    print("\n🧠 初始化语义组件 | Initialize Semantic Components...")
    discovery_engine = ContextAwareSemanticDiscoveryEngine(mock_llm)
    evolution_manager = SemanticEvolutionManager(semantic_repo, discovery_engine)
    unified_retrieval = UnifiedRetrievalService(episodic_repo, semantic_repo)

    # Enhanced episode builder
    enhanced_builder = EnhancedConversationEpisodeBuilder(llm_provider=mock_llm, semantic_manager=evolution_manager)

    print("✅ 语义组件初始化完成 | Semantic components initialized")

    try:
        # Create demo conversation
        print("\n📝 创建演示对话 | Create Demo Conversation...")
        raw_data = await create_demo_conversation()
        conversation_data = ConversationData(raw_data)

        print(f"✅ 对话创建完成: {len(conversation_data.messages)} 条消息")
        print(f"   参与者: {[msg.user_name for msg in conversation_data.messages]}")

        # Build episode with semantic processing
        print("\n🏗️ 构建情景记忆 | Build Episodic Memory...")
        episode = await enhanced_builder.build_episode(conversation_data, "john_conv1")

        print(f"✅ 情景记忆构建完成:")
        print(f"   标题: {episode.title}")
        print(f"   内容: {episode.content[:100]}...")
        print(f"   发现的语义知识数量: {episode.metadata.custom_fields.get('discovered_semantics', 0)}")

        # Test semantic retrieval
        print("\n🔍 测试语义检索 | Test Semantic Retrieval...")
        semantic_results = await unified_retrieval.search_semantic_memories(
            owner_id="john_conv1", query="研究", limit=5
        )

        print(f"✅ 语义检索结果: {len(semantic_results)} 个节点")
        for i, node in enumerate(semantic_results[:3]):
            print(f"   {i+1}. {node.key}: {node.value}")
            print(f"      置信度: {node.confidence}")

        # Test episodic retrieval
        print("\n📚 测试情景检索 | Test Episodic Retrieval...")
        episodic_results = await unified_retrieval.search_episodic_memories(
            owner_id="john_conv1", query="Agent", limit=3
        )

        print(f"✅ 情景检索结果: {len(episodic_results)} 个情景")
        for i, ep in enumerate(episodic_results):
            print(f"   {i+1}. {ep.title}")
            print(f"      内容预览: {ep.content[:50]}...")

        # Test knowledge evolution
        print("\n🔄 测试知识演变 | Test Knowledge Evolution...")
        await test_knowledge_evolution(enhanced_builder, unified_retrieval)

        print("\n🎉 所有测试通过！| All tests passed!")
        print("语义记忆系统与情景记忆完全集成 | Semantic memory fully integrated with episodic memory")

    finally:
        await episodic_repo.close()
        await semantic_repo.close()


async def test_knowledge_evolution(builder, unified_retrieval):
    """Test knowledge evolution with a follow-up conversation."""
    print("\n   📝 创建演变对话...")

    # Create follow-up conversation showing evolution
    messages = [
        {
            "speaker_id": "john_conv2",
            "user_name": "John",
            "content": "我现在不只研究AI Agent行为规划了",
            "timestamp": "2024-02-15T14:01:00",
            "metadata": {},
        },
        {
            "speaker_id": "user_conv2",
            "user_name": "User",
            "content": "有什么新的方向吗？",
            "timestamp": "2024-02-15T14:02:00",
            "metadata": {},
        },
        {
            "speaker_id": "john_conv2",
            "user_name": "John",
            "content": "现在还在研究多智能体系统的协调机制",
            "timestamp": "2024-02-15T14:03:00",
            "metadata": {},
        },
    ]

    raw_data = RawEventData(
        data_type=DataType.CONVERSATION,
        content=messages,
        source="demo_conversation_evolution",
        temporal_info=TemporalInfo(
            timestamp=datetime.fromisoformat("2024-02-15T14:01:00"), duration=120.0, timezone="UTC"
        ),
        metadata={"conversation_id": "conv2", "participant_count": 2},
    )

    conversation_data = ConversationData(raw_data)

    # Build episode to trigger evolution
    episode = await builder.build_episode(conversation_data, "john_conv2")

    print(f"   ✅ 演变情景创建: {episode.title}")
    print(f"   发现的语义知识数量: {episode.metadata.custom_fields.get('discovered_semantics', 0)}")

    # Check semantic knowledge
    semantic_results = await unified_retrieval.search_semantic_memories(
        owner_id="john_conv2", query="研究方向", limit=10
    )

    print(f"   📊 更新后的语义知识: {len(semantic_results)} 个节点")
    for node in semantic_results:
        if node.version > 1:
            print(f"   🔄 演变的知识: {node.key} -> {node.value} (版本 {node.version})")
            print(f"      历史值: {node.evolution_history}")


async def test_semantic_discovery():
    """Test semantic knowledge discovery."""
    print("🔬 Testing Semantic Discovery...")

    # Setup
    mock_llm = MockLLMProvider()
    discovery_engine = ContextAwareSemanticDiscoveryEngine(mock_llm)

    # Create a test episode
    episode = Episode(
        owner_id="test_user",
        title="John讨论研究方向",
        content="John表示最近在研究AI Agent的行为规划，特别关注决策机制。",
        summary="John的研究重点转向AI Agent行为规划",
        episode_type=EpisodeType.CONVERSATIONAL,
        level=EpisodeLevel.ATOMIC,
    )

    # Original conversation content
    original_content = """
    [2024-01-15 10:00] User: John，你最近在研究什么？
    [2024-01-15 10:01] John: 我最近在研究AI Agent的行为规划
    [2024-01-15 10:02] User: 这个方向很有前景
    [2024-01-15 10:03] John: 是的，特别是决策机制这块很有挑战性
    """

    # Discover semantic knowledge
    discovered_nodes = await discovery_engine.discover_semantic_knowledge(episode, original_content)

    print(f"✅ Discovered {len(discovered_nodes)} semantic nodes:")
    for node in discovered_nodes:
        print(f"   - {node.key}: {node.value}")
        print(f"     Context: {node.context}")
        print(f"     Confidence: {node.confidence}")

    return discovered_nodes


async def run_complete_demo():
    """Run the complete semantic memory demo."""
    print("🎯 选择演示模式 | Select Demo Mode:")
    print("1. 完整端到端演示 (推荐)")
    print("2. 基础语义发现测试")

    choice = input("请选择 (1-2) [默认: 1]: ").strip() or "1"

    if choice == "1":
        await test_end_to_end_semantic_memory()
    elif choice == "2":
        await test_semantic_discovery()
    else:
        print("无效选择，运行默认演示...")
        await test_end_to_end_semantic_memory()


if __name__ == "__main__":
    asyncio.run(run_complete_demo())
