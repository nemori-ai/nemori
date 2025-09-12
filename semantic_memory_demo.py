"""
Semantic Memory Demo for Nemori Framework

This demo shows how to use the enhanced Nemori framework with semantic memory capabilities.
It demonstrates the complete workflow from conversation processing to semantic knowledge discovery.
"""

import asyncio
from datetime import datetime

from nemori.core.data_types import ConversationData, ConversationMessage, DataType, RawEventData, TemporalInfo
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder


class MockLLMProvider:
    """Mock LLM provider for demo purposes."""

    def __init__(self):
        self.responses = {}
        self.call_count = 0

    def set_response_for_pattern(self, pattern: str, response: str):
        """Set response for a specific pattern in prompts."""
        self.responses[pattern] = response

    async def generate_async(self, prompt: str, **kwargs):
        """Generate response based on prompt patterns."""
        self.call_count += 1

        # Reconstruction prompts
        if "reconstruct" in prompt.lower():
            return """
            User: 你最近在研究什么？
            John: 我在研究一些AI技术。
            User: 能具体说说吗？
            John: 主要是关于机器学习的。
            """
        # Knowledge gap analysis prompts
        elif "knowledge gap" in prompt.lower():
            return """
            {
                "knowledge_gaps": [
                    {
                        "key": "John的研究方向",
                        "value": "AI Agent行为规划",
                        "context": "专注于决策机制",
                        "gap_type": "personal_fact",
                        "confidence": 0.9
                    }
                ]
            }
            """
        # Episode generation prompts
        elif "episodic memory generation" in prompt.lower():
            return """
            {
                "title": "John分享AI Agent研究进展",
                "content": "在2024年1月15日上午10点，John与用户讨论了他的研究方向。John表示他最近专注于AI Agent的行为规划研究，特别是决策机制方面的挑战。这标志着John从之前的研究领域转向了更具挑战性的Agent技术。"
            }
            """

        return "Mock LLM response"

    async def generate(self, prompt: str, **kwargs):
        """Sync version of generate."""
        return await self.generate_async(prompt, **kwargs)


async def run_semantic_memory_demo():
    """Run the complete semantic memory demo."""
    print("🚀 Nemori 语义记忆演示 | Semantic Memory Demo")
    print("=" * 60)

    # Initialize components
    print("\n🔧 初始化组件 | Initializing Components...")

    # Storage
    config = StorageConfig(connection_string="duckdb:///:memory:")
    semantic_storage = DuckDBSemanticMemoryRepository(config)
    await semantic_storage.initialize()

    # LLM Provider
    mock_llm = MockLLMProvider()

    # Semantic components
    discovery_engine = ContextAwareSemanticDiscoveryEngine(mock_llm)
    evolution_manager = SemanticEvolutionManager(semantic_storage, discovery_engine)
    unified_retrieval = UnifiedRetrievalService(None, semantic_storage)

    # Enhanced builder
    builder = EnhancedConversationEpisodeBuilder(llm_provider=mock_llm, semantic_manager=evolution_manager)

    print("✅ 组件初始化完成 | Components initialized")

    try:
        # Demo 1: First conversation about research
        print("\n📝 演示1：首次研究讨论 | Demo 1: First Research Discussion")
        print("-" * 40)

        messages_1 = [
            ConversationMessage(
                speaker_id="User", content="John，你最近在研究什么？", timestamp=datetime(2024, 1, 15, 10, 0)
            ),
            ConversationMessage(
                speaker_id="John", content="我最近在研究AI Agent的行为规划", timestamp=datetime(2024, 1, 15, 10, 1)
            ),
            ConversationMessage(speaker_id="User", content="听起来很有趣！", timestamp=datetime(2024, 1, 15, 10, 2)),
            ConversationMessage(
                speaker_id="John", content="是的，特别是决策机制这块很有挑战性", timestamp=datetime(2024, 1, 15, 10, 3)
            ),
        ]

        raw_data_1 = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[msg.__dict__ for msg in messages_1],
            temporal_info=TemporalInfo(datetime(2024, 1, 15, 10, 0)),
        )

        conversation_data_1 = ConversationData(raw_data_1)
        episode_1 = await builder.build_episode(conversation_data_1, "john")

        print(f"📖 情景标题 | Episode Title: {episode_1.title}")
        print(
            f"🧠 发现的语义知识 | Discovered Semantic Knowledge: {episode_1.metadata.custom_fields.get('discovered_semantics', 0)} pieces"
        )

        # Check what was discovered
        discovered_nodes = await semantic_storage.get_all_semantic_nodes_for_owner("john")
        for node in discovered_nodes:
            print(f"   💡 {node.key}: {node.value}")

        # Demo 2: Search semantic knowledge
        print("\n🔍 演示2：语义知识搜索 | Demo 2: Semantic Knowledge Search")
        print("-" * 40)

        search_results = await unified_retrieval.search_semantic_memories("john", "研究", 5)
        print(f"搜索'研究'找到 {len(search_results)} 条结果:")
        for result in search_results:
            print(f"   🔎 {result.key}: {result.value} (置信度: {result.confidence:.2f})")

        # Demo 3: Knowledge evolution (simulate research change)
        print("\n🔄 演示3：知识演变 | Demo 3: Knowledge Evolution")
        print("-" * 40)

        # Mock different LLM response for evolution
        mock_llm.responses[
            "gap"
        ] = """
        {
            "knowledge_gaps": [
                {
                    "key": "John的研究方向", 
                    "value": "多模态AI Agent系统",
                    "context": "从行为规划扩展到多模态",
                    "gap_type": "personal_fact",
                    "confidence": 0.95
                }
            ]
        }
        """

        messages_2 = [
            ConversationMessage(
                speaker_id="User", content="John，你的Agent研究有新进展吗？", timestamp=datetime(2024, 3, 20, 14, 0)
            ),
            ConversationMessage(
                speaker_id="John",
                content="是的，我现在专注于多模态AI Agent系统了",
                timestamp=datetime(2024, 3, 20, 14, 1),
            ),
            ConversationMessage(
                speaker_id="John", content="不仅要处理文本，还要整合视觉和语音", timestamp=datetime(2024, 3, 20, 14, 2)
            ),
        ]

        raw_data_2 = RawEventData(
            data_type=DataType.CONVERSATION,
            content=[msg.__dict__ for msg in messages_2],
            temporal_info=TemporalInfo(datetime(2024, 3, 20, 14, 0)),
        )

        conversation_data_2 = ConversationData(raw_data_2)
        episode_2 = await builder.build_episode(conversation_data_2, "john")

        print(f"📖 新情景标题 | New Episode Title: {episode_2.title}")

        # Check evolution
        evolved_nodes = await semantic_storage.get_all_semantic_nodes_for_owner("john")
        for node in evolved_nodes:
            if node.version > 1:
                print(f"   🧬 演变的知识 | Evolved Knowledge: {node.key}")
                print(f"      当前值 | Current: {node.value} (v{node.version})")
                print(f"      历史值 | History: {', '.join(node.evolution_history)}")

        # Demo 4: Statistics and analysis
        print("\n📊 演示4：统计分析 | Demo 4: Statistics & Analysis")
        print("-" * 40)

        stats = await semantic_storage.get_semantic_statistics("john")
        print(f"📈 统计信息 | Statistics:")
        print(f"   总节点数 | Total Nodes: {stats['total_nodes']}")
        print(f"   平均置信度 | Avg Confidence: {stats['average_confidence']:.2f}")
        print(f"   总访问次数 | Total Access: {stats['total_access_count']}")
        evolution_stats = await evolution_manager.get_knowledge_evolution_stats("john")
        print(f"🔄 演变统计 | Evolution Stats:")
        print(f"   演变节点数 | Evolved Nodes: {evolution_stats['evolved_nodes']}")
        print(f"   演变率 | Evolution Rate: {evolution_stats['evolution_rate']:.1%}")
        print(f"   总演变次数 | Total Evolutions: {evolution_stats['total_evolutions']}")

        # Demo 5: Contextual retrieval
        print("\n🎯 演示5：上下文检索 | Demo 5: Contextual Retrieval")
        print("-" * 40)

        context = await unified_retrieval.get_contextual_knowledge(episode_2)
        print(f"🔗 上下文知识摘要 | Context Summary: {context['context_summary']}")
        print(f"   关联语义节点 | Related Semantics: {len(context['related_semantics'])}")
        print(f"   相关历史情景 | Related Episodes: {len(context['related_episodes'])}")
        print("\n🎉 语义记忆演示完成！| Semantic Memory Demo Complete!")
        print("\n💡 关键特性展示 | Key Features Demonstrated:")
        print("   ✅ 自动语义知识发现 | Automatic semantic knowledge discovery")
        print("   ✅ 知识演变跟踪 | Knowledge evolution tracking")
        print("   ✅ 双向关联链接 | Bidirectional association linking")
        print("   ✅ 统一检索服务 | Unified retrieval service")
        print("   ✅ 上下文感知生成 | Context-aware generation")
    finally:
        await semantic_storage.close()


async def simple_usage_example():
    """Simple usage example for quick testing."""
    print("\n🎯 简单使用示例 | Simple Usage Example")
    print("=" * 50)

    # Minimal setup for quick testing
    from nemori.core.data_types import SemanticNode

    config = StorageConfig(connection_string="duckdb:///:memory:")
    storage = DuckDBSemanticMemoryRepository(config)
    await storage.initialize()

    try:
        # Create and store knowledge
        knowledge = SemanticNode(
            owner_id="demo_user",
            key="最喜欢的编程语言",
            value="Python",
            context="在讨论技术栈时提到最喜欢Python的简洁语法",
            confidence=0.9,
        )

        await storage.store_semantic_node(knowledge)
        print(f"✅ 存储知识: {knowledge.key} = {knowledge.value}")

        # Search and retrieve
        results = await storage.similarity_search_semantic_nodes("demo_user", "编程", 5)
        print(f"🔍 搜索'编程'找到 {len(results)} 条结果")

        for result in results:
            print(f"   💡 {result.key}: {result.value}")

    finally:
        await storage.close()


if __name__ == "__main__":
    print("🧠 Nemori Semantic Memory Framework")
    print("Choose demo to run:")
    print("1. Complete semantic memory demo")
    # print("2. Simple usage example")

    choice = "1"  # input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(run_semantic_memory_demo())
    elif choice == "2":
        asyncio.run(simple_usage_example())
    else:
        print("Running complete demo by default...")
        asyncio.run(run_semantic_memory_demo())
