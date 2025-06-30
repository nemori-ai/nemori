# Nemori Project Overview | Nemori 项目概览

## English Overview

### What is Nemori?

Nemori is a nature-inspired episodic memory system designed to transform raw user data into structured narrative episodes for large language models. The name "Nemori" derives from the Japanese word for "forest memory," reflecting the project's biomimetic approach to artificial intelligence memory.

### Project Vision

Nemori empowers large language models with human-like episodic memory capabilities. Unlike traditional memory systems, Nemori stores experiences as natural, event-centric traces that enable precise recall when needed. The ultimate vision is to make every piece of data remembered and retrieved as intuitively as human recollection.

### Key Design Philosophy

Nemori's design is inspired by human episodic memory patterns. When humans recall past events, our minds often flash with related images, actions, or sounds - our brains help us remember by making us re-experience what happened. This memory mechanism is called episodic memory, and Nemori seeks to replicate this for AI systems.

### Core Innovation: Granularity Alignment

A key insight in Nemori's design is that episodic memory granularity alignment offers optimization benefits for large language models. Since LLM training datasets align with human world textual distribution, aligning recall granularity simultaneously aligns with the "most probable event description granularity in the natural world."

This provides several advantages:
- **Reduced Distributional Shift**: When stored episodes match typical event spans in training corpora, recall prompts resemble pre-training distribution
- **Enhanced Retrieval Precision**: Memory indices storing "human-scale" events operate on semantically less entangled units

### Architecture Overview

Nemori follows a layered architecture with four main components:

1. **Data Ingestion Layer**: Handles various raw data types (conversations, activities, locations, etc.)
2. **Memory Processing Layer**: Transforms raw data into typed, structured formats
3. **Episodic Memory Layer**: Creates unified episode representations from processed data
4. **Storage & Retrieval Layer**: Optimizes episodes for search and recall

### Performance Results

Nemori has demonstrated superior performance on established benchmarks:
- **LoCoMo (Long-Context Conversation Modeling)**: Leading performance metrics
- **LongMemEval-s**: Competitive results compared to state-of-the-art systems

## Chinese Overview | 中文概览

### Nemori 是什么？

Nemori 是一个受自然启发的情景记忆系统，专为将原始用户数据转换为大语言模型的结构化叙事情景而设计。"Nemori"这个名字来源于日语中的"森林记忆"，反映了该项目对人工智能记忆的仿生学方法。

### 项目愿景

Nemori 旨在为大语言模型赋予类人的情景记忆能力。与传统记忆系统不同，Nemori 通过自然、事件化的索引方式存储经历，在需要时能够精准回忆。最终愿景是让每一次数据交互都能像人类记忆一样被理解、被回忆、被延续。

### 核心设计理念

Nemori 的设计灵感来自人类情景记忆模式。当人类回忆过去事件时，我们的脑海中经常闪现相关的图像、动作或声音——我们的大脑通过让我们重新体验当时发生的事情来帮助我们记忆。这种记忆机制被称为情景记忆，Nemori 致力于为 AI 系统复制这种能力。

### 核心创新：颗粒度对齐

Nemori 设计中的一个关键洞察是，情景记忆颗粒度对齐为大语言模型提供了优化收益。由于大模型的训练数据集与人类世界的文本分布对齐，对齐回忆颗粒度的同时也在对齐"自然世界中最大概率的事件表述颗粒度"。

这提供了几个优势：
- **减少分布偏移**：当存储的情景片段与训练语料中的典型事件跨度匹配时，回忆提示更接近预训练分布
- **增强检索精度**：存储"人类尺度"事件的记忆索引操作的是语义纠缠较少的单元

### 架构概览

Nemori 采用分层架构，包含四个主要组件：

1. **数据摄入层**：处理各种原始数据类型（对话、活动、位置等）
2. **记忆处理层**：将原始数据转换为类型化、结构化格式
3. **情景记忆层**：从处理后的数据创建统一的情景表示
4. **存储与检索层**：优化情景以便搜索和回忆

### 性能结果

Nemori 在已建立的基准测试中展现了卓越性能：
- **LoCoMo（长上下文对话建模）**：领先的性能指标
- **LongMemEval-s**：与最先进系统相比的竞争性结果

---

## Core Components | 核心组件

### 1. Data Types | 数据类型

**English**: The foundation of Nemori is its flexible data type system that can handle various forms of user experiences.

**中文**: Nemori 的基础是其灵活的数据类型系统，能够处理各种形式的用户体验。

#### Supported Data Types | 支持的数据类型:
- `CONVERSATION`: Chat messages, discussions | 聊天消息、讨论
- `ACTIVITY`: User actions, web browsing | 用户行为、网页浏览  
- `LOCATION`: GPS data, places visited | GPS 数据、访问的地点
- `MEDIA`: Images, videos, audio | 图像、视频、音频
- `DOCUMENT`: Files, notes, articles | 文件、笔记、文章
- `SENSOR`: Health data, device metrics | 健康数据、设备指标
- `EXTERNAL`: Third-party integrations | 第三方集成
- `CUSTOM`: User-defined types | 用户定义类型

#### Core Data Structures | 核心数据结构:

```python
@dataclass
class RawEventData:
    data_id: str                    # Unique identifier | 唯一标识符
    data_type: DataType            # Data type enum | 数据类型枚举
    content: Any                   # Raw content | 原始内容
    source: str                    # Data source | 数据来源
    temporal_info: TemporalInfo    # Time information | 时间信息
    metadata: dict[str, Any]       # Flexible metadata | 灵活元数据
    processed: bool                # Processing status | 处理状态
    processing_version: str        # Processing version | 处理版本
```

### 2. Episode System | 情景系统

**English**: Episodes are the unified output format that all user experiences are transformed into.

**中文**: 情景是所有用户体验都被转换成的统一输出格式。

#### Episode Types | 情景类型:
- `CONVERSATIONAL`: From dialogue data | 来自对话数据
- `BEHAVIORAL`: From activity data | 来自活动数据
- `SPATIAL`: From location data | 来自位置数据
- `CREATIVE`: From media/document creation | 来自媒体/文档创作
- `PHYSIOLOGICAL`: From sensor/health data | 来自传感器/健康数据
- `SOCIAL`: From external social data | 来自外部社交数据
- `MIXED`: From multiple sources | 来自多个来源
- `SYNTHETIC`: Generated/inferred episodes | 生成/推断的情景

#### Episode Levels | 情景层次:
- `ATOMIC` (1): Single event/interaction | 单个事件/交互
- `COMPOUND` (2): Multiple related events | 多个相关事件
- `THEMATIC` (3): Pattern-based insights | 基于模式的洞察
- `ARCHIVAL` (4): Long-term understanding | 长期理解

### 3. Builder System | 构建器系统

**English**: Specialized builders transform different data types into episodes using configurable LLM providers.

**中文**: 专门的构建器使用可配置的大语言模型提供者将不同数据类型转换为情景。

#### Key Features | 主要特性:
- **Intelligent Boundary Detection**: LLM-powered conversation segmentation | 智能边界检测：由大语言模型驱动的对话分段
- **Narrative Generation**: Converts dialogues to third-person narratives | 叙事生成：将对话转换为第三人称叙述
- **Graceful Degradation**: Fallback mode when LLM unavailable | 优雅降级：大语言模型不可用时的回退模式
- **Extensible Architecture**: Easy to add new data type builders | 可扩展架构：易于添加新的数据类型构建器

### 4. LLM Integration | 大语言模型集成

**English**: Nemori supports multiple LLM providers through a unified protocol interface.

**中文**: Nemori 通过统一的协议接口支持多个大语言模型提供者。

#### Supported Providers | 支持的提供者:
- **OpenAI**: GPT models (including GPT-4o-mini) | GPT 模型（包括 GPT-4o-mini）
- **Anthropic**: Claude models | Claude 模型
- **Google**: Gemini models | Gemini 模型

#### Provider Features | 提供者特性:
- Environment-based configuration | 基于环境的配置
- Automatic connection testing | 自动连接测试
- Temperature and token control | 温度和 token 控制
- Error handling and fallback | 错误处理和回退

---

## Technical Implementation | 技术实现

### Development Setup | 开发设置

**English**: Nemori uses modern Python development practices with uv for package management.

**中文**: Nemori 使用现代 Python 开发实践，采用 uv 进行包管理。

#### Requirements | 要求:
- Python 3.12+ | Python 3.12+
- uv package manager | uv 包管理器
- LangChain for LLM integration | 用于大语言模型集成的 LangChain
- OpenAI/Anthropic/Google API keys (optional) | OpenAI/Anthropic/Google API 密钥（可选）

#### Key Commands | 关键命令:
```bash
# Install dependencies | 安装依赖
uv sync

# Install in development mode | 开发模式安装
uv pip install -e .

# Run tests | 运行测试
uv run pytest

# Code formatting | 代码格式化
uv run black .
uv run ruff check .
```

### Testing Framework | 测试框架

**English**: Comprehensive test suite with pytest markers for different test categories.

**中文**: 使用 pytest 标记为不同测试类别提供的综合测试套件。

#### Test Categories | 测试类别:
- `unit`: Fast unit tests | 快速单元测试
- `integration`: Integration tests | 集成测试
- `llm`: Tests requiring LLM providers | 需要大语言模型提供者的测试
- `slow`: Time-intensive tests | 耗时测试

#### Mock Testing | 模拟测试:
- **MockLLMProvider**: Simulates LLM responses | 模拟大语言模型响应
- **Fixture System**: Comprehensive test data | 综合测试数据
- **Validation Helpers**: Episode and data validation | 情景和数据验证

---

## Usage Examples | 使用示例

### Basic Conversation Processing | 基本对话处理

**English**: Transform conversation data into episodic memories.

**中文**: 将对话数据转换为情景记忆。

```python
from nemori.core.data_types import RawEventData, DataType, TemporalInfo
from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.llm.providers.openai_provider import OpenAIProvider
from datetime import datetime

# Create conversation data | 创建对话数据
messages = [
    {
        "user_id": "alice", 
        "content": "I'm planning a trip to Japan",
        "timestamp": "2024-01-15T10:30:00"
    },
    {
        "user_id": "assistant", 
        "content": "That sounds exciting! What interests you most?",
        "timestamp": "2024-01-15T10:30:15"
    }
]

raw_data = RawEventData(
    data_type=DataType.CONVERSATION,
    content=messages,
    source="chat_app",
    temporal_info=TemporalInfo(timestamp=datetime.now())
)

# Set up builder with LLM | 使用大语言模型设置构建器
llm_provider = OpenAIProvider.from_env()
builder = ConversationEpisodeBuilder(llm_provider=llm_provider)

# Generate episode | 生成情景
episode = builder.build_episode(raw_data, for_owner="alice")

print(f"Title: {episode.title}")
print(f"Summary: {episode.summary}")
print(f"Level: {episode.level}")
```

### Multi-Provider Setup | 多提供者设置

**English**: Configure multiple LLM providers for different use cases.

**中文**: 为不同用例配置多个大语言模型提供者。

```python
from nemori.llm.providers import OpenAIProvider, AnthropicProvider, GeminiProvider

# Configure providers | 配置提供者
providers = {
    'openai': OpenAIProvider(model="gpt-4o-mini", temperature=0.3),
    'anthropic': AnthropicProvider(model="claude-3-haiku-20240307"),
    'gemini': GeminiProvider(model="gemini-pro")
}

# Test connections | 测试连接
for name, provider in providers.items():
    if provider.test_connection():
        print(f"✓ {name} connected")
    else:
        print(f"✗ {name} failed")
```

### Builder Registry | 构建器注册表

**English**: Manage multiple episode builders for different data types.

**中文**: 管理不同数据类型的多个情景构建器。

```python
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.builders.conversation_builder import ConversationEpisodeBuilder

# Create registry | 创建注册表
registry = EpisodeBuilderRegistry()

# Register builders | 注册构建器
conversation_builder = ConversationEpisodeBuilder(llm_provider=llm_provider)
registry.register(conversation_builder)

# Process data | 处理数据
episode = registry.build_episode(raw_data, for_owner="user_123")
```

---

## Project Structure | 项目结构

```
nemori/
├── nemori/                     # Main package | 主包
│   ├── core/                   # Core data structures | 核心数据结构
│   │   ├── data_types.py       # Data type definitions | 数据类型定义
│   │   ├── episode.py          # Episode classes | 情景类
│   │   └── builders.py         # Builder abstractions | 构建器抽象
│   ├── builders/               # Episode builders | 情景构建器
│   │   └── conversation_builder.py  # Conversation processing | 对话处理
│   └── llm/                    # LLM integration | 大语言模型集成
│       ├── protocol.py         # LLM protocol | 大语言模型协议
│       └── providers/          # Provider implementations | 提供者实现
├── tests/                      # Test suite | 测试套件
├── playground/                 # Examples and experiments | 示例和实验
├── figures/                    # Benchmark results | 基准测试结果
└── docs/                       # Documentation | 文档
```

### Key Files | 关键文件:

**English**: Core implementation files with their purposes.

**中文**: 核心实现文件及其用途。

- `data_types.py`: Defines all data structures and types | 定义所有数据结构和类型
- `episode.py`: Episode class and metadata definitions | 情景类和元数据定义
- `builders.py`: Abstract builder classes and registry | 抽象构建器类和注册表
- `conversation_builder.py`: Specialized conversation processing | 专门的对话处理
- `protocol.py`: LLM provider interface definition | 大语言模型提供者接口定义

---

## Future Roadmap | 未来路线图

### Short-term Goals | 短期目标

**English**: Immediate development priorities for enhancing Nemori's capabilities.

**中文**: 增强 Nemori 能力的即时开发优先级。

1. **Episode Aggregation | 情景聚合**
   - Similarity-based episode clustering | 基于相似性的情景聚类
   - Higher-level episode synthesis | 更高级别的情景合成
   - Temporal relationship mapping | 时间关系映射

2. **Enhanced Builders | 增强构建器**
   - Activity data builder | 活动数据构建器
   - Location data builder | 位置数据构建器
   - Media content builder | 媒体内容构建器

3. **Advanced Retrieval | 高级检索**
   - Vector-based semantic search | 基于向量的语义搜索
   - Temporal query support | 时间查询支持
   - Relevance scoring improvements | 相关性评分改进

### Long-term Vision | 长期愿景

**English**: Strategic directions for Nemori's evolution as an AI memory system.

**中文**: Nemori 作为 AI 记忆系统演进的战略方向。

1. **Multi-modal Memory | 多模态记忆**
   - Image and video episode processing | 图像和视频情景处理
   - Audio conversation analysis | 音频对话分析
   - Cross-modal relationship understanding | 跨模态关系理解

2. **Distributed Architecture | 分布式架构**
   - Scalable episode storage systems | 可扩展的情景存储系统
   - Federated learning capabilities | 联邦学习能力
   - Privacy-preserving memory sharing | 隐私保护的记忆共享

3. **AI Memory Research | AI 记忆研究**
   - Human vs AI memory pattern analysis | 人类与 AI 记忆模式分析
   - Cognitive alignment optimization | 认知对齐优化
   - Self-evolving memory architectures | 自进化记忆架构

---

## Research Impact | 研究影响

### Academic Contributions | 学术贡献

**English**: Nemori's contributions to AI memory research and cognitive science.

**中文**: Nemori 对 AI 记忆研究和认知科学的贡献。

- **Granularity Alignment Theory**: Novel approach to LLM memory optimization | 颗粒度对齐理论：大语言模型记忆优化的新方法
- **Episodic AI Architecture**: Biomimetic memory system design | 情景 AI 架构：仿生记忆系统设计
- **Benchmark Performance**: Superior results on LoCoMo and LongMemEval | 基准性能：在 LoCoMo 和 LongMemEval 上的卓越结果

### Industry Applications | 行业应用

**English**: Potential applications of Nemori in various domains.

**中文**: Nemori 在各个领域的潜在应用。

- **Personal AI Assistants**: Enhanced memory for user interactions | 个人 AI 助手：增强用户交互记忆
- **Customer Service**: Context-aware conversation handling | 客户服务：上下文感知的对话处理
- **Education Technology**: Personalized learning memory systems | 教育技术：个性化学习记忆系统
- **Healthcare**: Patient interaction and treatment history | 医疗保健：患者交互和治疗历史

---

## Getting Started | 开始使用

### Quick Installation | 快速安装

```bash
# Clone repository | 克隆仓库
git clone https://github.com/your-org/nemori.git
cd nemori

# Install with uv | 使用 uv 安装
uv sync

# Set up environment variables | 设置环境变量
export OPENAI_API_KEY="your-key-here"

# Run tests | 运行测试
uv run pytest
```

### Basic Usage | 基本用法

**English**: Start with the playground notebook for hands-on experience.

**中文**: 从 playground 笔记本开始获得实践体验。

```bash
# Navigate to playground | 导航到 playground
cd playground

# Open Jupyter notebook | 打开 Jupyter 笔记本
jupyter notebook build_memory.ipynb
```

### Documentation | 文档

**English**: Comprehensive documentation is available in multiple formats.

**中文**: 提供多种格式的综合文档。

- **API Reference**: Code documentation and examples | API 参考：代码文档和示例
- **Domain Model**: Detailed architecture description | 领域模型：详细的架构描述
- **Tutorials**: Step-by-step guides | 教程：逐步指南
- **Research Papers**: Academic publications | 研究论文：学术出版物

---

## Contributing | 贡献

### Development Guidelines | 开发指南

**English**: Standards and practices for contributing to Nemori.

**中文**: 为 Nemori 做贡献的标准和实践。

- **Code Style**: Black formatting with 120-character lines | 代码风格：120 字符行的 Black 格式化
- **Testing**: Comprehensive test coverage required | 测试：需要全面的测试覆盖
- **Documentation**: Bilingual documentation preferred | 文档：优选双语文档
- **Type Hints**: Full type annotation required | 类型提示：需要完整的类型注释

### Community | 社区

**English**: Join the Nemori community and contribute to AI memory research.

**中文**: 加入 Nemori 社区，为 AI 记忆研究做贡献。

- **Issues**: Report bugs and request features | 问题：报告错误和请求功能
- **Discussions**: Share ideas and feedback | 讨论：分享想法和反馈
- **Pull Requests**: Contribute code improvements | 拉取请求：贡献代码改进
- **Research**: Collaborate on academic projects | 研究：在学术项目上合作

---

*Nemori - Endowing AI agents with long-term memory to drive their self-evolution 🚀*

*Nemori - 赋予 AI 智能体长期记忆以驱动其自我进化 🚀*