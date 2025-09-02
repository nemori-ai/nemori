# Semantic Memory Extension for Nemori

## Overview | 概述

This extension adds semantic memory capabilities to the Nemori episodic memory framework. Semantic memory complements episodic memory by capturing and maintaining evolving private domain knowledge that would otherwise be lost during episodic compression.

该扩展为Nemori情景记忆框架添加了语义记忆功能。语义记忆通过捕获和维护在情景压缩过程中可能丢失的演变中的私域知识来补充情景记忆。

### Core Innovation | 核心创新

**Using Episodic Memory as a "Knowledge Mask"** | **使用情景记忆作为"知识掩码"**

The system automatically discovers semantic information that Large Language Models (LLMs) don't possess by:

系统通过以下方式自动发现大语言模型不具备的语义信息：

1. **Masking Phase** | **掩码阶段**: Episodic memory naturally masks semantic details through compression
2. **Reconstruction Phase** | **重建阶段**: LLM attempts to reconstruct original content using world knowledge  
3. **Differential Analysis** | **差分分析**: Compare reconstructed vs. original content to identify knowledge gaps
4. **Semantic Extraction** | **语义提取**: Knowledge gaps represent valuable private domain information

## Architecture | 架构

### Key Components | 关键组件

#### 1. Data Structures | 数据结构

**SemanticNode** - Represents individual pieces of semantic knowledge
```python
@dataclass
class SemanticNode:
    node_id: str
    owner_id: str
    key: str           # Knowledge identifier (e.g., "John的研究方向")
    value: str         # Knowledge content (e.g., "AI Agent行为规划") 
    context: str       # Original discovery context
    confidence: float  # Confidence level [0-1]
    version: int       # Evolution tracking
    evolution_history: list[str]  # Previous values
    linked_episode_ids: list[str] # Bidirectional episode links
    # ... additional fields
```

**SemanticRelationship** - Represents relationships between semantic nodes
```python
@dataclass
class SemanticRelationship:
    relationship_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    strength: float    # Relationship strength [0-1]
    # ... additional fields
```

#### 2. Discovery Engine | 发现引擎

**ContextAwareSemanticDiscoveryEngine** - Discovers semantic knowledge through differential analysis

```python
engine = ContextAwareSemanticDiscoveryEngine(llm_provider)
discovered_nodes = await engine.discover_semantic_knowledge(episode, original_content)
```

#### 3. Evolution Manager | 演变管理器

**SemanticEvolutionManager** - Manages knowledge evolution and bidirectional linking

```python
manager = SemanticEvolutionManager(storage, discovery_engine)
processed_nodes = await manager.process_episode_for_semantics(episode, original_content)
```

#### 4. Unified Retrieval | 统一检索

**UnifiedRetrievalService** - Provides dual retrieval for both episodic and semantic memories

```python
retrieval = UnifiedRetrievalService(episodic_storage, semantic_storage)

# Independent searches
episodes = await retrieval.search_episodic_memories(owner_id, query)
semantics = await retrieval.search_semantic_memories(owner_id, query)

# Bidirectional associations  
episode_semantics = await retrieval.get_episode_semantics(episode_id)
semantic_episodes = await retrieval.get_semantic_episodes(semantic_node_id)
```

## Usage Examples | 使用示例

### Basic Setup | 基本设置

```python
from nemori.semantic import (
    ContextAwareSemanticDiscoveryEngine,
    SemanticEvolutionManager, 
    UnifiedRetrievalService
)
from nemori.storage.duckdb_storage import DuckDBSemanticMemoryRepository
from nemori.builders import EnhancedConversationEpisodeBuilder

# Setup storage
config = StorageConfig(connection_string="duckdb:///semantic_memory.db")
semantic_storage = DuckDBSemanticMemoryRepository(config) 
await semantic_storage.initialize()

# Setup semantic components
discovery_engine = ContextAwareSemanticDiscoveryEngine(llm_provider)
evolution_manager = SemanticEvolutionManager(semantic_storage, discovery_engine)
unified_retrieval = UnifiedRetrievalService(episodic_storage, semantic_storage)

# Enhanced builder with semantic integration
builder = EnhancedConversationEpisodeBuilder(
    llm_provider=llm_provider,
    semantic_manager=evolution_manager
)
```

### Processing Conversations with Semantic Discovery | 处理对话并发现语义知识

```python
# Create conversation data
messages = [
    ConversationMessage(speaker_id="User", content="John，你最近在研究什么？"),
    ConversationMessage(speaker_id="John", content="我最近在研究AI Agent的行为规划"),
    ConversationMessage(speaker_id="John", content="特别关注决策机制这块")
]

conversation_data = ConversationData(raw_data)

# Build episode with automatic semantic discovery
episode = await builder.build_episode(conversation_data, "john")

# Check discovered semantic knowledge
semantic_count = episode.metadata.custom_fields.get("discovered_semantics", 0)
print(f"Discovered {semantic_count} pieces of semantic knowledge")
```

### Knowledge Evolution Example | 知识演变示例

```python
# First conversation: Initial knowledge
# "John的研究方向" → "大语言模型提示工程"

# Later conversation: Knowledge evolution  
# "John的研究方向" → "AI Agent行为规划" (evolved from previous)

# The system automatically:
# 1. Detects the knowledge change
# 2. Updates the semantic node with new value
# 3. Preserves evolution history
# 4. Links both episodes bidirectionally
```

### Dual Retrieval Capabilities | 双重检索能力

```python
# Search semantic memories independently
semantic_results = await unified_retrieval.search_semantic_memories(
    owner_id="john", 
    query="研究方向", 
    limit=10
)

# Search episodic memories independently  
episode_results = await unified_retrieval.search_episodic_memories(
    owner_id="john",
    query="研究方向", 
    limit=10
)

# Get bidirectional associations
episode_semantics = await unified_retrieval.get_episode_semantics(episode_id)
semantic_episodes = await unified_retrieval.get_semantic_episodes(semantic_node_id)
```

### Adaptive Memory Service | 自适应记忆服务

```python
from nemori.semantic.unified_retrieval import AdaptiveMemoryService

adaptive = AdaptiveMemoryService(unified_retrieval)

# Get memory optimized for different needs
factual_memory = await adaptive.get_memory_for_query(
    owner_id="john", 
    query="John在研究什么？",
    quality_preference="factual"  # Prioritizes semantic knowledge
)

contextual_memory = await adaptive.get_memory_for_query(
    owner_id="john",
    query="告诉我关于John的研究讨论", 
    quality_preference="contextual"  # Prioritizes episodic memories
)
```

## Business Use Cases | 业务应用场景

### 1. Personal AI Assistant | 个人AI助手

```python
# Scenario: User's preferences and habits evolution
# 场景：用户偏好和习惯演变

# Initial: "用户饮品偏好" → "喜欢咖啡"
# Later: "用户饮品偏好" → "最近改喝茶了" (evolved)

# Benefits:
# - Automatic preference tracking
# - Evolution history for context
# - Precise factual retrieval
```

### 2. Knowledge Management | 知识管理

```python
# Scenario: Team member expertise tracking
# 场景：团队成员专业技能跟踪

# Discovery: "张三的技术栈" → "React + Node.js"
# Evolution: "张三的技术栈" → "React + Python + AI" (expanded)

# Benefits:
# - Automatic skill discovery from conversations
# - Evolution tracking for career development
# - Relationship mapping between team members
```

### 3. Customer Relationship Management | 客户关系管理

```python
# Scenario: Customer preference evolution
# 场景：客户偏好演变

# Initial: "客户A的产品偏好" → "基础版产品"
# Later: "客户A的产品偏好" → "企业版高级功能" (upgraded)

# Benefits:
# - Automatic preference extraction from conversations
# - Purchase behavior evolution tracking
# - Context-aware recommendation generation
```

## Performance Considerations | 性能考虑

### Scalability | 可扩展性

- **Database Indexes**: Optimized queries for frequent access patterns
- **Connection Pooling**: Efficient concurrent access management
- **Batch Processing**: Multiple episode processing for semantic discovery
- **Async Processing**: Non-blocking semantic knowledge discovery

### Memory Management | 内存管理

- **LRU Caching**: Semantic node and relationship caching
- **Configurable Cache Size**: Adjustable based on system resources
- **Memory Monitoring**: Production usage tracking

### Discovery Efficiency | 发现效率

- **Context Sampling**: Large conversation dataset handling
- **Differential Analysis Optimization**: Efficient gap detection
- **Evolution Detection**: Smart change identification

## Configuration | 配置

### Storage Configuration | 存储配置

```python
from nemori.storage.storage_types import StorageConfig

config = StorageConfig(
    backend_type="duckdb",
    connection_string="duckdb:///semantic_memory.db",
    enable_semantic_search=True,
    cache_size=10000,
    batch_size=1000
)
```

### Discovery Configuration | 发现配置

```python
discovery_engine = ContextAwareSemanticDiscoveryEngine(
    llm_provider=llm_provider,
    confidence_threshold=0.7,      # Minimum confidence for knowledge acceptance
    evolution_threshold=0.1,       # Minimum change for evolution detection
    context_window=5              # Number of related memories for context
)
```

## Testing | 测试

### Running Tests | 运行测试

```bash
# Run semantic memory tests
uv run pytest tests/test_semantic_memory.py -v

# Run the demo
python semantic_memory_demo.py
```

### Test Coverage | 测试覆盖率

The test suite covers:
- Basic semantic node storage and retrieval
- Knowledge discovery from conversations
- Knowledge evolution tracking
- Bidirectional episode-semantic linking
- Unified retrieval service functionality
- Integration with enhanced conversation builder

## Integration with Existing Nemori | 与现有Nemori的集成

### Backward Compatibility | 向后兼容性

- **Existing Code**: All existing Nemori functionality remains unchanged
- **Optional Integration**: Semantic features are opt-in
- **Gradual Migration**: Can be added incrementally to existing systems

### Enhanced Components | 增强组件

- **EnhancedConversationEpisodeBuilder**: Drop-in replacement with semantic capabilities
- **UnifiedRetrievalService**: Combines episodic and semantic retrieval
- **Adaptive Memory Service**: Smart memory selection based on query characteristics

## Future Enhancements | 未来增强

### Planned Features | 计划功能

1. **Advanced NLP**: Semantic similarity models for better relationship detection
2. **Knowledge Graphs**: More sophisticated graph structures
3. **Multi-modal Support**: Images, audio, and video content
4. **Federated Learning**: Privacy-preserving knowledge sharing
5. **Active Learning**: LLM-guided discovery of missing knowledge gaps

### Roadmap | 路线图

- **Phase 1** ✅: Core semantic memory implementation
- **Phase 2**: Advanced relationship discovery
- **Phase 3**: Multi-modal semantic knowledge
- **Phase 4**: Distributed semantic memory networks

## Contributing | 贡献

We welcome contributions to the semantic memory extension! Please see the main Nemori repository for contribution guidelines.

欢迎为语义记忆扩展做出贡献！请查看主要的Nemori仓库了解贡献指南。

## License | 许可证

This semantic memory extension follows the same license as the main Nemori framework.

该语义记忆扩展遵循与主要Nemori框架相同的许可证。