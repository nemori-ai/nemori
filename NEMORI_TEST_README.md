# Nemori 全功能测试脚本

这个目录包含了用于测试 Nemori 记忆系统全部功能的测试脚本。

## 文件说明

### 1. nemori_full_functionality_test.py
**完整功能测试脚本**

- 包含5个不同场景的对话数据样例：
  - 技术讨论：前端框架选择和技术栈
  - 学术研究：AI模型对齐和论文投稿  
  - 日常生活：爬山摄影和兴趣爱好
  - 工作协作：AI推荐系统项目管理
  - 专业知识：机器学习算法和评估指标

- 测试功能包括：
  - ✅ Episodic Memory: 对话分割、episodes创建和存储
  - ✅ Semantic Memory: 隐含知识发现和抽取
  - ✅ Unified Retrieval: 统一检索系统测试
  - ✅ 搜索性能评估: 召回率和相关性测试

### 2. simple_nemori_full_test.py  
**简化功能测试脚本**

- 包含3个简单对话场景：
  - Python机器学习学习讨论
  - 推荐系统开发咨询
  - 户外活动和摄影计划

- 专注于核心功能验证，输出简洁易读

## 使用方法

### 前置条件

1. **LLM服务**: 确保有可用的OpenAI兼容API服务
2. **Embedding服务**: 确保本地embedding服务运行在 `http://localhost:6007/v1`
3. **Python环境**: 安装nemori项目的依赖

### 运行测试

#### 快速测试（推荐）
```bash
cd /data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori
python simple_nemori_full_test.py
```

#### 完整测试
```bash
cd /data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori  
python nemori_full_functionality_test.py
```

### 配置修改

如需修改API配置，编辑脚本中的以下部分：

```python
# LLM配置
api_key = "your-api-key"
base_url = "your-base-url"
model = "your-model"

# Embedding配置  
emb_api_key = "EMPTY"
emb_base_url = "http://localhost:6007/v1"
emb_model = "qwen3-emb"
```

## 测试数据格式

测试脚本使用与LoCoMo数据集兼容的格式：

```python
{
    "user_id": "conversation_id",
    "conversation": {
        "speaker_a": "Speaker Name A",
        "speaker_b": "Speaker Name B", 
        "session_1": [
            {
                "speaker": "Speaker Name",
                "text": "对话内容",
                "timestamp": "2024-01-20T10:00:00Z"
            }
        ],
        "session_1_date_time": "10:00 AM on 20 January, 2024"
    }
}
```

## 输出说明

### 成功运行的输出示例

```
🚀 Nemori 全功能测试
================================================================================
✅ 创建了 5 个对话场景
🤖 设置 LLM Provider...
✅ OpenAI connection successful!
📊 加载测试数据...
🗄️ 设置存储和检索服务...
🏗️ 构建Episodes和语义记忆...
✅ Successfully created 30 episodes
✅ Discovered 45 semantic concepts

🧠 测试 Episodic Memory 功能
✅ 成功创建 30 个episodes

🔍 测试 Semantic Memory 功能  
✅ Successfully discovered 45 semantic concepts

🔍 测试 Unified Retrieval 功能
✅ 找到相关episodes

🎉 全功能测试完成!
```

## 故障排除

### 常见问题

1. **LLM连接失败**
   - 检查API密钥和基础URL
   - 确认网络连接正常

2. **Embedding服务连接失败**
   - 确认本地embedding服务运行正常
   - 检查端口6007是否可访问

3. **模块导入错误**
   - 确认在正确的目录下运行脚本
   - 检查Python路径设置

4. **数据库初始化失败**
   - 检查磁盘空间
   - 确认有写入权限

### 调试技巧

- 使用 `max_concurrency=1` 进行串行处理，便于观察日志
- 在脚本中添加更多print语句观察执行流程
- 检查 `results/locomo/` 目录下的存储文件

## 扩展测试

### 添加新的测试场景

1. 在 `create_comprehensive_test_data()` 函数中添加新的对话数据
2. 按照现有格式创建新的conversation对象
3. 可以测试不同类型的对话内容，如：
   - 技术讨论
   - 学术研究
   - 商务对话
   - 日常聊天
   - 专业咨询

### 自定义测试查询

在 `test_unified_retrieval()` 函数中修改 `test_queries` 列表，添加针对你的数据的特定查询。

## 性能监控

测试脚本会输出以下性能指标：

- Episodes创建数量和速度
- 语义概念发现数量
- 检索结果相关性  
- 平均每个episode的语义概念数量
- 搜索召回率

这些指标可以帮助评估Nemori系统的性能和效果。