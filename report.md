# Nemori 记忆系统架构报告

## 1. 概览
Nemori 是一套分层记忆系统，负责接收连续对话、判定情景边界、持久化情景记忆、提炼长期语义知识，并对外提供快速检索接口。核心类 `MemorySystem` 串联了缓冲、生成、存储与索引组件；周边模块则提供工具、存储抽象以及向量/BM25 混合检索能力。系统采用文件式持久化（JSONL + FAISS/NumPy），并通过 OpenAI 的 LLM 与嵌入接口完成语言理解任务。

整体处理流程：
1. `MemorySystem.add_messages` 接收消息字典，封装为 `Message` 并写入用户缓冲；
2. 边界检测（LLM + 启发式）判断是否需要截断缓冲生成新情景；
3. 触发情景生成：调用 LLM 产出标题/正文/时间戳，失败时退化为 deterministic 摘要；随后持久化并更新向量/BM25 索引；
4. 语义记忆提炼在后台线程执行，利用预测-纠错引擎比对已有知识后抽取新增事实；
5. 检索层结合 FAISS、BM25 与排序融合，支持情景与语义两类记忆查询。

## 2. 配置与运行时（`src/config.py`）
`MemoryConfig` 集中管理存储路径、模型名称、语言设定、缓冲阈值、语义功能开关、检索策略以及性能参数，并从环境变量读取 `OPENAI_API_KEY`。配置会校验缓冲上下限，以及是否启用智能边界、预测纠错、FAISS、NOR-LIFT 等特性。

## 3. 数据模型（`src/models`）
- `Message` / `MessageBuffer`：保存单条对话与用户缓冲，维护时间戳与元数据；
- `Episode`：描述情景记忆（标题、叙事、原始消息、边界原因、时间戳、标签等）；
- `SemanticMemory`：存储长期知识陈述，记录类型、置信度、来源情景与版本信息。

## 4. 工具层（`src/utils`）
- `LLMClient`：封装 OpenAI Chat Completions，内置重试、JSON 解析与兜底逻辑；
- `EmbeddingClient`：负责批量嵌入、指数回退重试及相似度计算；
- `PerformanceOptimizer`：实现分片 LRU 缓存与并行调度，复用昂贵的边界/情景生成结果。

## 5. 存储层（`src/storage`）
`BaseStorage` 定义统一接口，具体实现包括：
- `EpisodeStorage`：按用户写入 `{user}_episodes.jsonl`，现已改为同步写盘，保证磁盘、缓存、内存索引一致；同时维护 per-user 锁、缓存、索引与 JSON→JSONL 迁移；
- `SemanticStorage`：将语义记忆追加到 `{user}_semantic.jsonl`，提供去重、统计与删除功能。

目录结构示例：
```
memories/
  episodes/
    <user>_episodes.jsonl
    vector_db/
      <user>.faiss
      <user>_embeddings.npy
  semantic/
    <user>_semantic.jsonl
    vector_db/
      <user>.faiss
      <user>_embeddings.npy
```

## 6. 生成层（`src/generation`）
- `prompts.py`：集中维护边界、情景、语义、预测纠错等提示词；
- `EpisodeGenerator`：格式化缓冲消息并调用 LLM 输出结构化 JSON，失败时生成 fallback 摘要；
- `BoundaryDetector`：先跑快速启发式，再调用 LLM 判定；若出错则退化到关键词重叠检测；
- `PredictionCorrectionEngine`：实现“预测 → 比较 → 抽取”的两阶段学习，冷启动时直接从情景中提取高价值知识；
- `SemanticGenerator`：根据配置选择预测纠错或逐情景抽取，并可调用向量检索辅助。

## 7. 核心运行时（`src/core`）
### 7.1 `MessageBufferManager`
以用户维度加锁管理 `MessageBuffer`，提供新增、清空、删除操作以及缓冲状态查询。

### 7.2 `MemorySystem`
系统核心，负责：
- **初始化**：惰性加载存储、检索、生成器和缓存，并创建语义生成线程池；
- **`add_messages` 流程**：
  1. 解析输入消息并写入缓冲；
  2. 触发边界检测，必要时分批生成情景；
  3. 调用 `EpisodeGenerator` 生成情景并持久化，再增量更新向量索引；
  4. 为新情景调度语义生成任务；
  5. 更新统计数据与锁管理。
- **语义生成任务**：后台线程加载用户情景，选取历史语义知识，调用 `SemanticGenerator` 产出新知识并写入存储及索引；
- **检索加载**：`load_user_data_and_indices*` 会根据文件、索引时间戳决定是否重建向量库；
- **维护接口**：如 `delete_user_data`、`flush_all_buffers`、统计查询等。

### 7.3 `LightweightSearchSystem`
简化版检索系统，直接从磁盘预加载情景/语义至内存，并仅使用 `VectorSearch`，适合只读场景。

## 8. 检索子系统（`src/search`）
- `VectorSearch`：封装 FAISS/NumPy，支持全量索引、增量更新、持久化（`.faiss` + `.npy`）与查询；
- `BM25Search`：为情景/语义构建 BM25Okapi 索引，优先使用 spaCy 词元化，回退到正则分词，并缓存分词结果提升增量效率；
- `UnifiedSearchEngine`：调度向量与 BM25 检索，支持单引擎模式与基于 RRF 的混合排序，必要时可启用 NOR-LIFT。

## 9. 评估与脚本
`evaluation/locomo` 等目录中包含全流程脚本：`add.py` 写入记忆并等待语义生成，`search.py`/`evals.py` 运行检索与评分，`generate_scores.py` 汇总指标。生成的记忆与向量存放在 `evaluation/memories`。

## 10. 设计要点
- **用户级锁**：缓冲、存储、索引均按用户加锁，易于并发扩展；
- **缓存与重试**：性能优化器缓存高成本调用，LLM/嵌入客户端使用指数回退；
- **鲁棒性**：情景与语义生成具备 fallback 路径，避免 LLM 故障导致数据缺失；
- **索引一致性**：同步写盘 + 索引状态检查，必要时自动重建 FAISS；
- **模块化**：分层设计便于替换存储或检索实现。

综上，Nemori 将原始对话转化为结构化情景记忆，并提炼为可检索的长期知识，为后续智能体提供可扩展的认知记忆底座。
