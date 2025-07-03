# Nemori 存储层设计与实现

## 概述

基于 Nemori 领域模型的要求，我们设计并实现了一个完整的存储层解决方案，支持原始数据的存储和情景记忆的管理。存储层遵循以下核心原则：

1. **原始数据完整性**：保存所有原始内容，确保数据不丢失
2. **关联关系管理**：支持情景记忆与原始数据的双向关联
3. **灵活的检索功能**：提供多种检索方式满足不同需求
4. **可扩展架构**：支持不同存储后端的实现

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────┐
│               应用层                         │
├─────────────────────────────────────────────┤
│               存储接口层                      │
│  - StorageRepository (基础仓库)              │
│  - RawDataRepository (原始数据仓库)          │
│  - EpisodicMemoryRepository (情景记忆仓库)   │
├─────────────────────────────────────────────┤
│               查询类型层                      │
│  - RawDataQuery / EpisodeQuery              │
│  - 搜索结果类型                              │
│  - 配置和统计类型                            │
├─────────────────────────────────────────────┤
│               具体实现层                      │
│  - MemoryRawDataRepository                  │
│  - MemoryEpisodicMemoryRepository           │
│  - DuckDBRawDataRepository                  │
│  - DuckDBEpisodicMemoryRepository           │
│  - (未来可扩展其他实现)                      │
└─────────────────────────────────────────────┘
```

### 核心组件

#### 1. 存储配置 (StorageConfig)
- 存储后端类型配置
- 性能参数（批处理大小、缓存设置）
- 索引配置（全文搜索、语义搜索）
- 数据保留和备份策略

#### 2. 查询类型系统
- **RawDataQuery**: 原始数据查询参数
  - 支持按数据类型、来源、时间范围过滤
  - 支持内容搜索和元数据过滤
  - 支持分页和排序
  
- **EpisodeQuery**: 情景记忆查询参数
  - 支持按所有者、类型、层次过滤
  - 支持文本搜索、关键词搜索、实体搜索
  - 支持语义相似性搜索
  - 支持重要性和访问频率过滤

#### 3. 搜索结果容器
- **RawDataSearchResult**: 原始数据搜索结果
- **EpisodeSearchResult**: 情景记忆搜索结果（包含相关性评分）

## 核心功能

### 原始数据管理

#### 存储功能
```python
# 单条存储
data_id = await raw_repo.store_raw_data(raw_data)

# 批量存储
data_ids = await raw_repo.store_raw_data_batch(data_list)
```

#### 查询功能
```python
# 按类型查询
query = RawDataQuery(data_types=[DataType.CONVERSATION])
results = await raw_repo.search_raw_data(query)

# 按内容搜索
query = RawDataQuery(content_contains="关键词")
results = await raw_repo.search_raw_data(query)

# 按时间范围查询
time_range = TimeRange(start=start_time, end=end_time)
query = RawDataQuery(time_range=time_range)
results = await raw_repo.search_raw_data(query)
```

#### 处理状态管理
```python
# 标记为已处理
await raw_repo.mark_as_processed(data_id, "v1.0")

# 获取未处理数据
unprocessed = await raw_repo.get_unprocessed_data(data_type, limit=100)
```

### 情景记忆管理

#### 存储和检索
```python
# 存储情景
episode_id = await episode_repo.store_episode(episode)

# 按ID检索
episode = await episode_repo.get_episode(episode_id)

# 按所有者检索
results = await episode_repo.get_episodes_by_owner("user_123")
```

#### 多种搜索方式
```python
# 文本搜索
results = await episode_repo.search_episodes_by_text("Python 编程")

# 关键词搜索
results = await episode_repo.search_episodes_by_keywords(["学习", "编程"])

# 语义搜索（基于嵌入向量）
results = await episode_repo.search_episodes_by_embedding(
    embedding_vector, threshold=0.8
)

# 复合查询
query = EpisodeQuery(
    owner_ids=["user_123"],
    episode_types=[EpisodeType.CONVERSATIONAL],
    min_importance=0.5,
    recent_hours=24
)
results = await episode_repo.search_episodes(query)
```

### 关联关系管理

#### 情景与原始数据关联
```python
# 建立关联
await episode_repo.link_episode_to_raw_data(episode_id, [raw_data_id])

# 查询情景的原始数据
raw_data_ids = await episode_repo.get_raw_data_for_episode(episode_id)

# 查询原始数据生成的情景
episodes = await episode_repo.get_episodes_for_raw_data(raw_data_id)
```

#### 情景间关联
```python
# 建立情景关联
await episode_repo.link_related_episodes(episode_id1, episode_id2)

# 查询相关情景
related = await episode_repo.get_related_episodes(episode_id)
```

### 访问追踪和重要性管理

```python
# 标记访问
await episode_repo.mark_episode_accessed(episode_id)

# 更新重要性
await episode_repo.update_episode_importance(episode_id, 0.9)
```

## 存储实现

### 内存存储实现

#### MemoryRawDataRepository
- 基于字典的内存存储
- 多索引支持（按类型、来源）
- 处理状态追踪
- 支持复杂查询和排序

#### MemoryEpisodicMemoryRepository
- 内存中的情景记忆管理
- 多维度索引（所有者、类型、层次）
- 关联关系管理
- 相似性计算和相关性评分

#### 特性
- **快速访问**：内存操作，毫秒级响应
- **完整功能**：支持所有抽象接口定义的功能
- **数据一致性**：自动维护索引和关联关系
- **备份恢复**：支持 JSON 格式的备份和恢复

### DuckDB 存储实现

#### DuckDBRawDataRepository
- 基于 SQLModel 的类型安全实现
- 嵌入式 DuckDB 数据库，支持 SQL 标准查询
- 自动输入验证和 SQL 注入防护
- 支持事务和数据持久化

#### DuckDBEpisodicMemoryRepository  
- 关系型数据库设计，维护数据完整性
- 索引优化，支持复杂查询
- 并发安全，支持多进程访问
- 完整的备份和恢复功能

## 性能特性

### 查询性能
- 基于索引的快速过滤
- 支持组合查询条件
- 分页支持避免大结果集
- 查询时间统计

### 内存管理
- 增量索引更新
- 避免数据冗余
- 支持批量操作提高效率

### 相关性计算
- 文本匹配评分
- 关键词权重计算
- 语义相似性计算（余弦相似度）
- 多因子综合评分

## 系统监控

### 统计信息
```python
# 获取存储统计
stats = await repo.get_stats()

# 统计内容包括：
# - 数据总量和分类统计
# - 存储大小
# - 时间范围
# - 性能指标
```

### 健康检查
```python
# 检查存储健康状态
is_healthy = await repo.health_check()
```

### 备份恢复
```python
# 创建备份
success = await repo.backup("/path/to/backup.json")

# 从备份恢复
success = await repo.restore("/path/to/backup.json")
```


## 扩展性

### 新存储后端
可以通过实现抽象接口来添加新的存储后端：
- SQLite 存储实现
- PostgreSQL 存储实现
- Elasticsearch 存储实现
- 分布式存储实现

### 新查询类型
可以扩展查询参数类型以支持更复杂的搜索需求：
- 地理位置查询
- 图像相似性查询
- 音频内容查询

### 新索引类型
可以添加新的索引策略：
- 向量数据库集成
- 全文搜索引擎集成
- 图数据库集成

## 使用示例

存储层的基本使用流程：

1. **存储配置和初始化** - 选择合适的存储实现并初始化
2. **原始数据管理** - 存储、查询和标记处理状态
3. **情景记忆管理** - 创建、检索和更新情景记忆
4. **关联关系建立** - 建立情景与原始数据的关联
5. **多维度搜索** - 支持文本、关键词、语义等搜索方式
6. **访问追踪** - 记录访问历史和更新重要性评分
7. **系统监控** - 获取统计信息和健康状态
8. **数据备份** - 创建和恢复数据备份

## 总结

Nemori 存储层的设计和实现完成了以下目标：

✅ **原始数据完整性**：支持多种数据类型的原始内容保存  
✅ **情景记忆管理**：完整的情景记忆生命周期管理  
✅ **关联关系**：支持情景与原始数据、情景间的关联  
✅ **灵活检索**：多种检索方式满足不同场景需求  
✅ **多种实现**：内存存储和 DuckDB 存储满足不同需求  
✅ **可扩展性**：抽象接口设计支持多种存储后端  
✅ **系统监控**：完整的统计、健康检查和备份功能  

该存储层为 Nemori 项目提供了坚实的数据管理基础，支持未来的功能扩展和性能优化。