# Nemori 语义记忆设计文档 | Semantic Memory Design Document

## 项目概述 | Project Overview

### English Overview

This document outlines the design and implementation strategy for adding **Semantic Memory** capabilities to the Nemori episodic memory system. Semantic memory complements episodic memory by capturing and maintaining evolving private domain knowledge that would otherwise be lost during episodic compression.

**Core Innovation**: Using episodic memory as a "knowledge mask" to automatically discover semantic information that Large Language Models (LLMs) don't possess, creating an iteratively updatable private knowledge base with relationship mapping.

### 中文概述

本文档概述了为 Nemori 情景记忆系统增加**语义记忆**能力的设计与实现策略。语义记忆通过捕获和维护在情景压缩过程中可能丢失的演变中的私域知识，来补充情景记忆的不足。

**核心创新**：使用情景记忆作为"知识掩码"，自动发现大语言模型不具备的语义信息，创建可迭代更新的带关系映射的私域知识库。

---

## 设计理念 | Design Philosophy

### The Knowledge Loss Problem | 知识丢失问题

**English**: Episodic memory, by design, compresses raw conversations into narrative summaries. This compression inevitably leads to the loss of detailed semantic information. The lost information falls into two categories:

1. **Common Knowledge**: Information already known to the LLM (e.g., "Python vs JavaScript advantages")
2. **Private Domain Knowledge**: User-specific information unknown to the LLM (e.g., "Tanka project details", "John's research focus")

**中文**: 情景记忆设计上会将原始对话压缩为叙述性摘要。这种压缩不可避免地导致详细语义信息的丢失。丢失的信息分为两类：

1. **通用知识**：大语言模型已知的信息（如"Python vs JavaScript 的优势"）
2. **私域知识**：大语言模型未知的用户特定信息（如"Tanka 项目详情"、"John 的研究重点"）

### The Semantic Discovery Mechanism | 语义发现机制

**English**: Our approach uses episodic memory as a "mask" to identify private domain knowledge through a differential reconstruction process:

1. **Masking Phase**: Episodic memory naturally masks semantic details through compression
2. **Reconstruction Phase**: LLM attempts to reconstruct original content using its world knowledge
3. **Differential Analysis**: Compare reconstructed vs. original content to identify knowledge gaps
4. **Semantic Extraction**: Knowledge gaps represent valuable private domain information

**中文**: 我们的方法使用情景记忆作为"掩码"，通过差分重建过程识别私域知识：

1. **掩码阶段**：情景记忆通过压缩自然地掩盖语义细节
2. **重建阶段**：大语言模型尝试使用其世界知识重建原始内容
3. **差分分析**：比较重建内容与原始内容，识别知识差距
4. **语义提取**：知识差距代表有价值的私域信息

---

## 核心架构设计 | Core Architecture Design

### 设计原则 | Design Principles

**English**: Based on the core requirements, the semantic memory system follows these principles:

1. **Gap-Driven Discovery**: All semantic memory originates from episodic memory analysis
2. **Dual Retrieval Capability**: Independent similarity-based search for both semantic and episodic memories
3. **Bidirectional ID-based Association**: Precise linking between episodes and semantic nodes, including evolved versions
4. **Context-Aware Generation**: Utilize related semantic memories and historical episodes during generation

**中文**: 基于核心需求，语义记忆系统遵循以下原则：

1. **间隙驱动发现**：所有语义记忆都源于情景记忆分析
2. **双重检索能力**：语义记忆和情景记忆的独立相似度搜索
3. **双向ID关联**：情景与语义节点的精确链接，包括演变版本
4. **上下文感知生成**：生成时利用相关语义记忆和历史情景

### 1. 语义记忆数据结构 | Semantic Memory Data Structures

#### SemanticNode | 语义节点

```python
@dataclass
class SemanticNode:
    """
    Represents a single piece of semantic knowledge.
    表示单个语义知识片段。
    """
    # Core identification | 核心标识
    node_id: str = field(default_factory=lambda: str(uuid4()))
    owner_id: str = ""
    
    # Knowledge content | 知识内容
    key: str = ""          # Knowledge key/identifier | 知识键/标识符
    value: str = ""        # Knowledge content | 知识内容
    context: str = ""      # Original context where discovered | 发现时的原始上下文
    
    # Confidence and evolution | 置信度与演变
    confidence: float = 1.0           # Confidence in this knowledge | 对该知识的置信度
    version: int = 1                  # Version number for evolution | 演变的版本号
    evolution_history: list[str] = field(default_factory=list)  # Previous values | 历史值
    
    # Temporal information | 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    
    # Discovery metadata | 发现元数据
    discovery_episode_id: str | None = None  # Episode that led to discovery | 导致发现的情景
    discovery_method: str = "diff_analysis"   # How this was discovered | 发现方法
    
    # Bidirectional associations | 双向关联
    linked_episode_ids: list[str] = field(default_factory=list)  # Episodes that reference this knowledge | 引用此知识的情景
    evolution_episode_ids: list[str] = field(default_factory=list)  # Episodes that caused evolution | 导致演变的情景
    
    # Search optimization | 搜索优化
    search_keywords: list[str] = field(default_factory=list)  # Keywords for similarity search | 相似度搜索关键词
    embedding_vector: list[float] | None = None  # Optional vector for semantic search | 可选的语义搜索向量
    
    # Usage statistics | 使用统计
    access_count: int = 0
    relevance_score: float = 0.0
    importance_score: float = 0.0
```

#### SemanticRelationship | 语义关系

```python
@dataclass 
class SemanticRelationship:
    """
    Represents relationships between semantic nodes.
    表示语义节点之间的关系。
    """
    # Core identification | 核心标识
    relationship_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Relationship definition | 关系定义
    source_node_id: str = ""
    target_node_id: str = ""
    relationship_type: RelationshipType = RelationshipType.RELATED
    
    # Relationship properties | 关系属性
    strength: float = 0.5      # Relationship strength [0-1] | 关系强度 [0-1]
    description: str = ""      # Optional description | 可选描述
    
    # Temporal information | 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    
    # Discovery context | 发现上下文
    discovery_episode_id: str | None = None
    
class RelationshipType(Enum):
    """Types of semantic relationships | 语义关系类型"""
    RELATED = "related"           # General relationship | 一般关系
    EVOLVED_FROM = "evolved_from" # One concept evolved from another | 一个概念从另一个演变而来
    PART_OF = "part_of"          # Part-whole relationship | 部分-整体关系
    SIMILAR_TO = "similar_to"    # Similarity relationship | 相似关系
    OPPOSITE_TO = "opposite_to"  # Opposition relationship | 对立关系
    TEMPORAL = "temporal"        # Time-based relationship | 基于时间的关系
```

### 2. 双重检索系统 | Dual Retrieval System

#### UnifiedRetrievalService | 统一检索服务

```python
class UnifiedRetrievalService:
    """
    Unified service providing similarity-based retrieval for both episodic and semantic memories.
    为情景记忆和语义记忆提供相似度检索的统一服务。
    """
    
    def __init__(
        self, 
        episodic_storage: EpisodeStorage,
        semantic_storage: SemanticStorage
    ):
        self.episodic_storage = episodic_storage
        self.semantic_storage = semantic_storage
    
    async def search_episodic_memories(
        self, 
        owner_id: str, 
        query: str, 
        limit: int = 10
    ) -> list[Episode]:
        """
        Independent similarity search for episodic memories.
        情景记忆的独立相似度搜索。
        """
        return await self.episodic_storage.similarity_search(owner_id, query, limit)
    
    async def search_semantic_memories(
        self, 
        owner_id: str, 
        query: str, 
        limit: int = 10
    ) -> list[SemanticNode]:
        """
        Independent similarity search for semantic memories.
        语义记忆的独立相似度搜索。
        """
        return await self.semantic_storage.similarity_search(owner_id, query, limit)
    
    async def get_episode_semantics(self, episode_id: str) -> list[SemanticNode]:
        """
        Get all semantic nodes discovered from a specific episode.
        获取从特定情景发现的所有语义节点。
        """
        return await self.semantic_storage.find_by_discovery_episode(episode_id)
    
    async def get_semantic_episodes(self, semantic_node_id: str) -> dict[str, list[Episode]]:
        """
        Get all episodes associated with a semantic node, including evolution history.
        获取与语义节点关联的所有情景，包括演变历史。
        
        Returns:
            {
                "linked_episodes": [episodes that reference this knowledge],
                "evolution_episodes": [episodes that caused knowledge evolution]
            }
        """
        semantic_node = await self.semantic_storage.get_by_id(semantic_node_id)
        if not semantic_node:
            return {"linked_episodes": [], "evolution_episodes": []}
        
        linked_episodes = await self.episodic_storage.get_by_ids(semantic_node.linked_episode_ids)
        evolution_episodes = await self.episodic_storage.get_by_ids(semantic_node.evolution_episode_ids)
        
        return {
            "linked_episodes": linked_episodes,
            "evolution_episodes": evolution_episodes
        }
```

### 3. 语义发现引擎 | Semantic Discovery Engine

#### ContextAwareSemanticDiscoveryEngine | 上下文感知语义发现引擎

```python
class ContextAwareSemanticDiscoveryEngine:
    """
    Context-aware engine for discovering semantic knowledge through differential analysis.
    通过差分分析发现语义知识的上下文感知引擎。
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider,
        retrieval_service: UnifiedRetrievalService
    ):
        self.llm_provider = llm_provider
        self.retrieval_service = retrieval_service
        
    async def discover_semantic_knowledge(
        self, 
        episode: Episode, 
        original_content: str
    ) -> list[SemanticNode]:
        """
        Discover semantic knowledge with context from related memories.
        利用相关记忆的上下文发现语义知识。
        
        Process | 流程:
        1. Gather related semantic memories and historical episodes | 收集相关语义记忆和历史情景
        2. Use episode as mask to reconstruct original | 使用情景作为掩码重建原始内容
        3. Compare reconstructed vs original with context | 结合上下文比较重建与原始内容
        4. Extract knowledge gaps as semantic nodes | 提取知识差距作为语义节点
        """
        # Step 1: Gather context from related memories
        # 步骤1：从相关记忆中收集上下文
        context = await self._gather_discovery_context(episode)
        
        # Step 2: Context-aware reconstruction
        # 步骤2：上下文感知重建
        reconstructed_content = await self._reconstruct_with_context(episode, context)
        
        # Step 3: Perform differential analysis
        # 步骤3：执行差分分析
        knowledge_gaps = await self._analyze_knowledge_gaps(
            original=original_content,
            reconstructed=reconstructed_content,
            episode=episode,
            context=context
        )
        
        # Step 4: Create semantic nodes with bidirectional links
        # 步骤4：创建带双向链接的语义节点
        semantic_nodes = []
        for gap in knowledge_gaps:
            node = SemanticNode(
                owner_id=episode.owner_id,
                key=gap.get("key", ""),
                value=gap.get("value", ""),
                context=gap.get("context", ""),
                discovery_episode_id=episode.episode_id,
                linked_episode_ids=[episode.episode_id],  # Initial linking
                confidence=gap.get("confidence", 1.0),
                search_keywords=self._extract_keywords(gap)
            )
            semantic_nodes.append(node)
            
        return semantic_nodes
    
    async def _gather_discovery_context(self, episode: Episode) -> dict[str, Any]:
        """
        Gather related semantic memories and historical episodes for context.
        收集相关语义记忆和历史情景作为上下文。
        """
        # Search for related semantic memories
        # 搜索相关语义记忆
        related_semantics = await self.retrieval_service.search_semantic_memories(
            owner_id=episode.owner_id,
            query=f"{episode.title} {episode.summary}",
            limit=5
        )
        
        # Search for related historical episodes
        # 搜索相关历史情景
        related_episodes = await self.retrieval_service.search_episodic_memories(
            owner_id=episode.owner_id,
            query=episode.content,
            limit=3
        )
        
        return {
            "related_semantic_memories": related_semantics,
            "related_historical_episodes": related_episodes,
            "current_episode": episode
        }
```

#### Differential Analysis Prompts | 差分分析提示词

```python
RECONSTRUCTION_PROMPT = """
You are an expert at reconstructing original conversations from episodic summaries.
您是从情景摘要重建原始对话的专家。

Given this episodic memory:
给定以下情景记忆：
{episode_content}

Please reconstruct what the original conversation might have looked like, using your general world knowledge.
请使用您的通用世界知识重建原始对话可能的样子。

Important guidelines | 重要准则:
1. Use only common knowledge that a typical LLM would know | 只使用典型大语言模型会知道的常识
2. Make reasonable assumptions for missing details | 对缺失细节做合理假设
3. Focus on factual reconstruction, not creative interpretation | 专注于事实重建，而非创意解释
4. Maintain the same conversation structure and flow | 保持相同的对话结构和流程

Return the reconstructed conversation:
返回重建的对话：
"""

KNOWLEDGE_GAP_ANALYSIS_PROMPT = """
You are an expert at identifying private domain knowledge gaps.
您是识别私域知识差距的专家。

Original content | 原始内容:
{original_content}

Reconstructed content (using general LLM knowledge) | 重建内容（使用通用大语言模型知识）:
{reconstructed_content}

Please identify specific pieces of information that exist in the original but are missing or incorrectly assumed in the reconstruction. These represent private domain knowledge.
请识别原始内容中存在但在重建中缺失或错误假设的具体信息。这些代表私域知识。

Focus on | 关注:
1. Proper names, project names, specific terminology | 专有名词、项目名称、特定术语
2. Personal preferences, habits, and characteristics | 个人偏好、习惯和特征
3. Specific facts, dates, numbers that differ | 具体的事实、日期、数字差异
4. Context-specific meanings and interpretations | 上下文特定的含义和解释

Return your analysis in JSON format:
以 JSON 格式返回分析：
{
    "knowledge_gaps": [
        {
            "key": "specific identifier or topic",
            "value": "the correct private knowledge",
            "context": "surrounding context from original",
            "gap_type": "proper_noun|personal_fact|specific_detail|contextual_meaning",
            "confidence": 0.0-1.0
        }
    ]
}
"""
```

### 4. 语义记忆演变管理器 | Semantic Memory Evolution Manager

#### SemanticEvolutionManager | 语义演变管理器

```python
class SemanticEvolutionManager:
    """
    Manages semantic memory evolution with bidirectional episode associations.
    管理语义记忆演变及双向情景关联。
    """
    
    def __init__(
        self, 
        storage: SemanticStorage, 
        discovery_engine: ContextAwareSemanticDiscoveryEngine,
        retrieval_service: UnifiedRetrievalService
    ):
        self.storage = storage
        self.discovery_engine = discovery_engine
        self.retrieval_service = retrieval_service
        
    async def process_episode_for_semantics(
        self, 
        episode: Episode, 
        original_content: str
    ) -> list[SemanticNode]:
        """
        Process an episode to discover and update semantic knowledge with full context.
        处理情景以发现和更新语义知识，包含完整上下文。
        """
        # Discover new semantic knowledge with context
        # 利用上下文发现新的语义知识
        discovered_nodes = await self.discovery_engine.discover_semantic_knowledge(
            episode, original_content
        )
        
        # Process each discovered node for evolution and linking
        # 处理每个发现的节点以进行演变和关联
        processed_nodes = []
        for new_node in discovered_nodes:
            processed_node = await self._process_semantic_node(new_node, episode)
            processed_nodes.append(processed_node)
            
        return processed_nodes
        
    async def _process_semantic_node(
        self, 
        new_node: SemanticNode, 
        episode: Episode
    ) -> SemanticNode:
        """
        Process a semantic node for evolution, linking, and storage.
        处理语义节点以进行演变、关联和存储。
        """
        # Check if knowledge already exists
        # 检查知识是否已存在
        existing_node = await self.storage.find_semantic_node_by_key(
            owner_id=new_node.owner_id, 
            key=new_node.key
        )
        
        if existing_node:
            # Handle evolution of existing knowledge
            # 处理现有知识的演变
            return await self._evolve_semantic_knowledge(existing_node, new_node, episode)
        else:
            # Store new knowledge with bidirectional linking
            # 存储新知识并建立双向链接
            await self.storage.store_semantic_node(new_node)
            await self._establish_bidirectional_links(new_node, episode)
            return new_node
    
    async def _evolve_semantic_knowledge(
        self, 
        existing: SemanticNode, 
        new: SemanticNode, 
        episode: Episode
    ) -> SemanticNode:
        """
        Evolve existing semantic knowledge with comprehensive evolution tracking.
        用综合演变跟踪演变现有语义知识。
        """
        if existing.value != new.value:
            # Knowledge evolution detected
            # 检测到知识演变
            evolved_node = replace(existing,
                value=new.value,
                context=new.context,  # Update context
                version=existing.version + 1,
                evolution_history=existing.evolution_history + [existing.value],
                evolution_episode_ids=existing.evolution_episode_ids + [episode.episode_id],
                last_updated=datetime.now(),
                confidence=(existing.confidence + new.confidence) / 2
            )
            
            # Update storage
            # 更新存储
            await self.storage.update_semantic_node(evolved_node)
            
            # Link this episode as an evolution trigger
            # 将此情景链接为演变触发器
            await self._establish_bidirectional_links(evolved_node, episode)
            
            return evolved_node
        else:
            # Reinforce existing knowledge without evolution
            # 强化现有知识但不演变
            reinforced_node = replace(existing,
                linked_episode_ids=list(set(existing.linked_episode_ids + [episode.episode_id])),
                confidence=min(1.0, existing.confidence + 0.1),
                last_accessed=datetime.now(),
                access_count=existing.access_count + 1
            )
            
            await self.storage.update_semantic_node(reinforced_node)
            await self._establish_bidirectional_links(reinforced_node, episode)
            
            return reinforced_node
    
    async def _establish_bidirectional_links(
        self, 
        semantic_node: SemanticNode, 
        episode: Episode
    ) -> None:
        """
        Establish bidirectional links between semantic node and episode.
        在语义节点和情景之间建立双向链接。
        """
        # Update episode to reference this semantic node
        # 更新情景以引用此语义节点
        if "semantic_node_ids" not in episode.metadata.custom_fields:
            episode.metadata.custom_fields["semantic_node_ids"] = []
        
        if semantic_node.node_id not in episode.metadata.custom_fields["semantic_node_ids"]:
            episode.metadata.custom_fields["semantic_node_ids"].append(semantic_node.node_id)
            
        # Note: Episode storage update would happen in the calling context
        # 注意：情景存储更新会在调用上下文中发生
    
    async def get_semantic_evolution_history(
        self, 
        semantic_node_id: str
    ) -> dict[str, Any]:
        """
        Get comprehensive evolution history including all related episodes.
        获取包括所有相关情景的综合演变历史。
        
        Returns:
            {
                "node": current_semantic_node,
                "evolution_timeline": [
                    {
                        "version": 1,
                        "value": "historical_value",
                        "episode": episode_that_caused_change,
                        "timestamp": change_timestamp
                    }
                ],
                "linked_episodes": [episodes that reference this knowledge],
                "evolution_episodes": [episodes that caused evolution]
            }
        """
        semantic_node = await self.storage.get_by_id(semantic_node_id)
        if not semantic_node:
            return {}
        
        # Build evolution timeline
        # 构建演变时间线
        evolution_timeline = []
        
        # Add historical versions
        # 添加历史版本
        for i, historical_value in enumerate(semantic_node.evolution_history):
            episode_id = semantic_node.evolution_episode_ids[i] if i < len(semantic_node.evolution_episode_ids) else None
            episode = None
            if episode_id:
                episodes = await self.retrieval_service.episodic_storage.get_by_ids([episode_id])
                episode = episodes[0] if episodes else None
            
            evolution_timeline.append({
                "version": i + 1,
                "value": historical_value,
                "episode": episode,
                "timestamp": episode.temporal_info.timestamp if episode else semantic_node.created_at
            })
        
        # Add current version
        # 添加当前版本
        evolution_timeline.append({
            "version": semantic_node.version,
            "value": semantic_node.value,
            "episode": None,  # Current version
            "timestamp": semantic_node.last_updated
        })
        
        # Get associated episodes
        # 获取关联情景
        associated_episodes = await self.retrieval_service.get_semantic_episodes(semantic_node_id)
        
        return {
            "node": semantic_node,
            "evolution_timeline": evolution_timeline,
            "linked_episodes": associated_episodes["linked_episodes"],
            "evolution_episodes": associated_episodes["evolution_episodes"]
        }
```

### 4. 关系发现与管理 | Relationship Discovery and Management

#### RelationshipDiscoveryEngine | 关系发现引擎

```python
class RelationshipDiscoveryEngine:
    """
    Discovers and manages relationships between semantic nodes.
    发现和管理语义节点之间的关系。
    """
    
    async def discover_relationships(
        self, 
        nodes: list[SemanticNode], 
        context_episode: Episode
    ) -> list[SemanticRelationship]:
        """
        Discover relationships between semantic nodes based on context.
        基于上下文发现语义节点之间的关系。
        """
        relationships = []
        
        # Simple bidirectional association discovery
        # 简单的双向关联发现
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                relationship_strength = await self._calculate_relationship_strength(
                    node_a, node_b, context_episode
                )
                
                if relationship_strength > 0.3:  # Threshold for meaningful relationship
                    relationship = SemanticRelationship(
                        source_node_id=node_a.node_id,
                        target_node_id=node_b.node_id,
                        relationship_type=RelationshipType.RELATED,
                        strength=relationship_strength,
                        discovery_episode_id=context_episode.episode_id,
                        description=f"Co-discovered in episode: {context_episode.title}"
                    )
                    relationships.append(relationship)
                    
        return relationships
        
    async def _calculate_relationship_strength(
        self, 
        node_a: SemanticNode, 
        node_b: SemanticNode, 
        context: Episode
    ) -> float:
        """
        Calculate relationship strength between two semantic nodes.
        计算两个语义节点之间的关系强度。
        
        Simple implementation - can be enhanced with NLP similarity analysis.
        简单实现 - 可通过 NLP 相似性分析增强。
        """
        # Temporal proximity
        # 时间邻近性
        time_factor = 1.0 if node_a.created_at == node_b.created_at else 0.7
        
        # Context similarity (same episode)
        # 上下文相似性（相同情景）
        context_factor = 1.0 if node_a.discovery_episode_id == node_b.discovery_episode_id else 0.5
        
        # Textual similarity (basic implementation)
        # 文本相似性（基本实现）
        text_similarity = self._calculate_text_similarity(node_a.context, node_b.context)
        
        return (time_factor * 0.3 + context_factor * 0.4 + text_similarity * 0.3)
```

---

## 存储层设计 | Storage Layer Design

### SemanticStorage Interface | 语义存储接口

```python
class SemanticStorage(ABC):
    """Abstract interface for semantic memory storage."""
    
    @abstractmethod
    async def store_semantic_node(self, node: SemanticNode) -> None:
        """Store a semantic node."""
        pass
        
    @abstractmethod
    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        """Find semantic node by key."""
        pass
        
    @abstractmethod
    async def store_relationship(self, relationship: SemanticRelationship) -> None:
        """Store a semantic relationship."""
        pass
        
    @abstractmethod
    async def find_related_nodes(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """Find nodes related to given node."""
        pass
        
    @abstractmethod
    async def query_semantic_knowledge(self, owner_id: str, query: str) -> list[SemanticNode]:
        """Query semantic knowledge with text search."""
        pass
```

### Database Schema | 数据库模式

#### PostgreSQL Implementation | PostgreSQL 实现

```sql
-- Semantic nodes table | 语义节点表
CREATE TABLE semantic_nodes (
    node_id VARCHAR(36) PRIMARY KEY,
    owner_id VARCHAR(255) NOT NULL,
    key VARCHAR(500) NOT NULL,
    value TEXT NOT NULL,
    context TEXT,
    confidence FLOAT DEFAULT 1.0,
    version INTEGER DEFAULT 1,
    evolution_history JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    discovery_episode_id VARCHAR(36),
    discovery_method VARCHAR(50) DEFAULT 'diff_analysis',
    access_count INTEGER DEFAULT 0,
    relevance_score FLOAT DEFAULT 0.0,
    importance_score FLOAT DEFAULT 0.0,
    
    UNIQUE(owner_id, key)
);

-- Semantic relationships table | 语义关系表
CREATE TABLE semantic_relationships (
    relationship_id VARCHAR(36) PRIMARY KEY,
    source_node_id VARCHAR(36) NOT NULL,
    target_node_id VARCHAR(36) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    strength FLOAT DEFAULT 0.5,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_episode_id VARCHAR(36),
    
    FOREIGN KEY (source_node_id) REFERENCES semantic_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES semantic_nodes(node_id) ON DELETE CASCADE,
    UNIQUE(source_node_id, target_node_id, relationship_type)
);

-- Indexes for performance | 性能索引
CREATE INDEX idx_semantic_nodes_owner_key ON semantic_nodes(owner_id, key);
CREATE INDEX idx_semantic_nodes_owner_updated ON semantic_nodes(owner_id, last_updated);
CREATE INDEX idx_semantic_relationships_source ON semantic_relationships(source_node_id);
CREATE INDEX idx_semantic_relationships_target ON semantic_relationships(target_node_id);

-- Full-text search | 全文搜索
CREATE INDEX idx_semantic_nodes_search ON semantic_nodes USING gin(to_tsvector('english', key || ' ' || value || ' ' || context));
```

---

## 检索层设计 | Retrieval Layer Design

### SemanticRetrievalService | 语义检索服务

```python
class SemanticRetrievalService:
    """
    Service for retrieving semantic knowledge.
    语义知识检索服务。
    """
    
    def __init__(self, storage: SemanticStorage):
        self.storage = storage
        
    async def query_semantic_knowledge(
        self, 
        owner_id: str, 
        query: str, 
        limit: int = 10
    ) -> list[SemanticNode]:
        """
        Query semantic knowledge with text search.
        使用文本搜索查询语义知识。
        """
        # Basic text search implementation
        # 基本文本搜索实现
        results = await self.storage.query_semantic_knowledge(owner_id, query, limit)
        
        # Sort by relevance and recency
        # 按相关性和时效性排序
        sorted_results = sorted(results, 
            key=lambda x: (x.relevance_score, x.last_updated), 
            reverse=True
        )
        
        return sorted_results[:limit]
        
    async def get_related_knowledge(
        self, 
        node_id: str, 
        max_depth: int = 2
    ) -> dict[str, list[SemanticNode]]:
        """
        Get related knowledge nodes with specified depth.
        获取指定深度的相关知识节点。
        """
        visited = set()
        result = {"direct": [], "indirect": []}
        
        # Get direct relationships
        # 获取直接关系
        direct_related = await self.storage.find_related_nodes(node_id)
        result["direct"] = [node for node, _ in direct_related]
        visited.add(node_id)
        
        # Get indirect relationships if max_depth > 1
        # 如果 max_depth > 1 则获取间接关系
        if max_depth > 1:
            for node, _ in direct_related:
                if node.node_id not in visited:
                    visited.add(node.node_id)
                    indirect_related = await self.storage.find_related_nodes(node.node_id)
                    for indirect_node, _ in indirect_related:
                        if indirect_node.node_id not in visited:
                            result["indirect"].append(indirect_node)
                            
        return result
        
    async def get_knowledge_evolution(
        self, 
        owner_id: str, 
        key: str
    ) -> list[dict[str, Any]]:
        """
        Get evolution history of a specific knowledge key.
        获取特定知识键的演变历史。
        """
        node = await self.storage.find_semantic_node_by_key(owner_id, key)
        if not node:
            return []
            
        evolution = []
        
        # Add historical versions
        # 添加历史版本
        for i, historical_value in enumerate(node.evolution_history):
            evolution.append({
                "version": i + 1,
                "value": historical_value,
                "timestamp": node.created_at,  # Approximate - could be enhanced
                "confidence": 1.0  # Historical confidence unknown
            })
            
        # Add current version
        # 添加当前版本
        evolution.append({
            "version": node.version,
            "value": node.value,
            "timestamp": node.last_updated,
            "confidence": node.confidence
        })
        
        return evolution
```

---

## 集成策略 | Integration Strategy

### Integration with Existing Nemori Architecture | 与现有 Nemori 架构的集成

#### 1. Episode Builder Enhancement | 情景构建器增强

```python
class EnhancedConversationEpisodeBuilder(ConversationEpisodeBuilder):
    """
    Enhanced conversation builder with semantic memory integration.
    集成语义记忆的增强对话构建器。
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider | None = None,
        semantic_manager: SemanticMemoryManager | None = None,
        **kwargs
    ):
        super().__init__(llm_provider, **kwargs)
        self.semantic_manager = semantic_manager
        
    async def build_episode(self, data: RawEventData, for_owner: str) -> Episode:
        """
        Build episode and process semantic knowledge.
        构建情景并处理语义知识。
        """
        # Build episode using parent method
        # 使用父方法构建情景
        episode = await super().build_episode(data, for_owner)
        
        # Process semantic knowledge if manager available
        # 如果管理器可用则处理语义知识
        if self.semantic_manager and isinstance(data, ConversationData):
            original_content = data.get_conversation_text(include_timestamps=True)
            
            # Discover and store semantic knowledge
            # 发现并存储语义知识
            semantic_nodes = await self.semantic_manager.process_episode_for_semantics(
                episode, original_content
            )
            
            # Add semantic metadata to episode
            # 向情景添加语义元数据
            if semantic_nodes:
                episode.metadata.custom_fields["discovered_semantics"] = len(semantic_nodes)
                episode.metadata.custom_fields["semantic_node_ids"] = [
                    node.node_id for node in semantic_nodes
                ]
                
        return episode
```

#### 2. Memory Retrieval Enhancement | 记忆检索增强

```python
class EnhancedRetrievalService:
    """
    Enhanced retrieval combining episodic and semantic memory.
    结合情景和语义记忆的增强检索服务。
    """
    
    def __init__(
        self, 
        episodic_retrieval: RetrievalService,
        semantic_retrieval: SemanticRetrievalService
    ):
        self.episodic_retrieval = episodic_retrieval
        self.semantic_retrieval = semantic_retrieval
        
    async def enhanced_query(
        self, 
        owner_id: str, 
        query: str, 
        include_semantic: bool = True
    ) -> dict[str, Any]:
        """
        Enhanced query combining episodic and semantic results.
        结合情景和语义结果的增强查询。
        """
        results = {
            "episodes": await self.episodic_retrieval.search_episodes(owner_id, query),
            "semantic_knowledge": []
        }
        
        if include_semantic:
            results["semantic_knowledge"] = await self.semantic_retrieval.query_semantic_knowledge(
                owner_id, query
            )
            
        return results
```


---

## 业务使用示例 | Business Usage Examples

### 完整的语义记忆生命周期 | Complete Semantic Memory Lifecycle

**场景：用户 John 的研究方向演变**

#### 第一次对话 | First Conversation
```python
# 原始对话
original_conversation = """
[2024-01-15 10:00] John: 我最近在研究大语言模型的提示工程
[2024-01-15 10:01] Assistant: 很有趣的方向！你主要关注哪些方面？
[2024-01-15 10:02] John: 主要是如何让LLM更好地理解复杂指令
"""

# 情景记忆（压缩后）
episode_1 = Episode(
    title="John讨论大语言模型提示工程研究",
    content="John表示最近在研究大语言模型的提示工程，特别关注如何让LLM更好地理解复杂指令。",
    owner_id="john"
)

# 语义发现过程
# LLM重建：John提到研究某种AI技术，可能是机器学习相关
# 差分发现：
semantic_node_1 = SemanticNode(
    key="John的研究方向",
    value="大语言模型提示工程",
    context="专注于让LLM理解复杂指令",
    discovery_episode_id=episode_1.episode_id,
    linked_episode_ids=[episode_1.episode_id]
)
```

#### 三个月后的对话 | Conversation After 3 Months
```python
# 原始对话
original_conversation_2 = """
[2024-04-20 14:30] John: 我现在转向AI Agent的行为规划了
[2024-04-20 14:31] Assistant: 从LLM转向Agent了？
[2024-04-20 14:32] John: 对，发现Agent的决策机制更有挑战性
"""

# 情景记忆
episode_2 = Episode(
    title="John转向AI Agent行为规划研究",
    content="John表示已从之前的LLM研究转向AI Agent的行为规划，认为Agent的决策机制更有挑战性。",
    owner_id="john"
)

# 语义演变过程
# 发现同一个key的不同value
evolved_semantic_node = SemanticNode(
    key="John的研究方向",
    value="AI Agent行为规划",  # 新值
    context="专注于Agent决策机制",
    version=2,  # 版本升级
    evolution_history=["大语言模型提示工程"],  # 保存历史
    evolution_episode_ids=[episode_2.episode_id],
    linked_episode_ids=[episode_1.episode_id, episode_2.episode_id]
)
```

### 业务检索应用场景 | Business Retrieval Scenarios

#### 场景1：独立相似度搜索 | Independent Similarity Search

```python
# 用户查询：John现在在研究什么？
query = "John现在在研究什么"

# 1. 搜索语义记忆
semantic_results = await retrieval_service.search_semantic_memories(
    owner_id="john", 
    query=query
)
# 返回：当前版本的"AI Agent行为规划"

# 2. 搜索情景记忆  
episodic_results = await retrieval_service.search_episodic_memories(
    owner_id="john",
    query=query
)
# 返回：相关的对话情景

# 业务可以选择使用哪种记忆类型
if need_factual_knowledge:
    return semantic_results  # 直接的事实知识
elif need_conversation_context:
    return episodic_results  # 对话上下文
else:
    return combine(semantic_results, episodic_results)  # 组合使用
```

#### 场景2：双向ID关联查询 | Bidirectional ID Association Query

```python
# 查询特定语义节点的演变历史
evolution_history = await evolution_manager.get_semantic_evolution_history(
    semantic_node_id="john_research_direction_node_id"
)

# 返回完整演变历史
{
    "node": current_semantic_node,
    "evolution_timeline": [
        {
            "version": 1,
            "value": "大语言模型提示工程",
            "episode": episode_1,  # 包含完整情景信息
            "timestamp": "2024-01-15T10:00:00"
        },
        {
            "version": 2, 
            "value": "AI Agent行为规划",
            "episode": episode_2,
            "timestamp": "2024-04-20T14:30:00"
        }
    ],
    "linked_episodes": [episode_1, episode_2],  # 所有关联情景
    "evolution_episodes": [episode_2]  # 导致演变的情景
}
```

#### 场景3：上下文感知生成 | Context-Aware Generation

```python
# 新对话到来时，发现过程会利用历史信息
new_episode = Episode(
    title="John讨论Agent架构设计",
    content="John分享了他在Agent架构设计方面的新想法...",
    owner_id="john"
)

# 上下文收集过程
context = await discovery_engine._gather_discovery_context(new_episode)
# 自动找到：
{
    "related_semantic_memories": [
        john_research_direction_node,  # John的研究方向
        agent_related_nodes  # 其他Agent相关知识
    ],
    "related_historical_episodes": [
        episode_2,  # 之前关于Agent研究的对话
        other_agent_episodes  # 其他相关情景
    ]
}

# 利用这些上下文进行更准确的语义发现
```

### 按需记忆质量选择 | On-Demand Memory Quality Selection

```python
class AdaptiveMemoryService:
    """
    Adaptive service that provides different memory qualities based on business needs.
    根据业务需求提供不同质量记忆的自适应服务。
    """
    
    async def get_memory_for_query(
        self, 
        owner_id: str, 
        query: str, 
        quality_preference: str = "balanced"
    ) -> dict[str, Any]:
        """
        Get memory with specified quality preference.
        获取指定质量偏好的记忆。
        """
        if quality_preference == "factual":
            # 需要准确事实：优先语义记忆
            return {
                "primary": await self.retrieval_service.search_semantic_memories(owner_id, query),
                "secondary": []
            }
            
        elif quality_preference == "contextual":
            # 需要丰富上下文：优先情景记忆
            return {
                "primary": await self.retrieval_service.search_episodic_memories(owner_id, query),
                "secondary": []
            }
            
        elif quality_preference == "comprehensive":
            # 需要全面信息：组合两种记忆
            semantic_results = await self.retrieval_service.search_semantic_memories(owner_id, query)
            episodic_results = await self.retrieval_service.search_episodic_memories(owner_id, query)
            
            # 对于每个语义节点，获取其关联的情景
            enriched_results = []
            for semantic_node in semantic_results:
                associated_episodes = await self.retrieval_service.get_semantic_episodes(semantic_node.node_id)
                enriched_results.append({
                    "semantic_knowledge": semantic_node,
                    "supporting_episodes": associated_episodes["linked_episodes"],
                    "evolution_context": associated_episodes["evolution_episodes"]
                })
            
            return {
                "primary": enriched_results,
                "secondary": episodic_results
            }
            
        else:  # balanced
            # 平衡模式：基于查询类型自动选择
            return await self._smart_memory_selection(owner_id, query)
```

---

## 性能考虑 | Performance Considerations

### Scalability | 可扩展性

**English Considerations:**

1. **Storage Optimization**
   - Use database indexes for frequent queries
   - Implement connection pooling for concurrent access
   - Consider read replicas for query-heavy workloads

2. **Memory Management**
   - Limit semantic node cache size
   - Implement LRU eviction for relationship cache
   - Monitor memory usage in production

3. **Discovery Efficiency**
   - Batch process multiple episodes for semantic discovery
   - Implement async processing for non-blocking discovery
   - Use sampling for large conversation datasets

**中文考虑：**

1. **存储优化**
   - 为频繁查询使用数据库索引
   - 为并发访问实现连接池
   - 为查询密集型工作负载考虑只读副本

2. **内存管理**
   - 限制语义节点缓存大小
   - 为关系缓存实现 LRU 淘汰
   - 在生产环境中监控内存使用

3. **发现效率**
   - 批量处理多个情景进行语义发现
   - 为非阻塞发现实现异步处理
   - 为大型对话数据集使用采样

### Monitoring & Analytics | 监控与分析

**Key Metrics | 关键指标:**

- **Discovery Rate**: Semantic nodes discovered per episode | 每个情景发现的语义节点数
- **Evolution Frequency**: How often knowledge evolves | 知识演变频率
- **Relationship Density**: Average relationships per node | 每个节点的平均关系数
- **Query Performance**: Response time for semantic queries | 语义查询响应时间
- **Storage Growth**: Rate of semantic knowledge accumulation | 语义知识积累速度

---

## 安全与隐私 | Security & Privacy

### Privacy Protection | 隐私保护

**English Measures:**

1. **Data Isolation**: Semantic knowledge strictly scoped to owner
2. **Encryption**: Sensitive knowledge encrypted at rest
3. **Access Control**: Role-based access to semantic data
4. **Audit Logging**: Track access to private domain knowledge

**中文措施：**

1. **数据隔离**：语义知识严格限定于所有者
2. **加密**：敏感知识静态加密
3. **访问控制**：基于角色的语义数据访问
4. **审计日志**：跟踪私域知识访问

### Data Retention | 数据保留

**Policies | 政策:**

- **Automatic Cleanup**: Remove unused semantic nodes after configurable period
- **Version Limits**: Limit evolution history to prevent unbounded growth
- **User Control**: Allow users to manually delete sensitive knowledge

---

## 总结 | Conclusion

### Innovation Summary | 创新总结

**English**: This semantic memory design introduces a novel approach to private domain knowledge extraction using episodic memory as a "knowledge mask." The differential reconstruction technique automatically identifies information gaps between LLM world knowledge and user-specific knowledge, creating an evolving semantic knowledge base with relationship mapping.

**Key Benefits:**
- **Automatic Discovery**: No manual knowledge entry required
- **Evolutionary Design**: Knowledge evolves with new information
- **Relationship Mapping**: Simple but effective knowledge connections
- **Integration**: Seamless integration with existing Nemori architecture

**中文**: 该语义记忆设计引入了一种新颖的私域知识提取方法，使用情景记忆作为"知识掩码"。差分重建技术自动识别大语言模型世界知识与用户特定知识之间的信息差距，创建带关系映射的演进语义知识库。

**主要优势：**
- **自动发现**：无需手动知识输入
- **演进设计**：知识随新信息演变
- **关系映射**：简单但有效的知识连接
- **集成性**：与现有 Nemori 架构无缝集成

### Future Enhancements | 未来增强

**Potential Improvements | 潜在改进:**

1. **Advanced NLP**: Use semantic similarity models for better relationship detection
2. **Knowledge Graphs**: Evolve to more sophisticated graph structures
3. **Multi-modal Support**: Extend to images, audio, and video content
4. **Federated Learning**: Enable privacy-preserving knowledge sharing
5. **Active Learning**: LLM-guided discovery of missing knowledge gaps

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-XX  
**Authors**: Nemori Development Team  

*Nemori - 赋予 AI 智能体长期记忆以驱动其自我进化 🚀*