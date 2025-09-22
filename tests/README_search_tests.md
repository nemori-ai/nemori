# 记忆搜索测试脚本使用说明

## 概述

这里提供了两个测试脚本，用于手动测试从 `evaluation/memories/` 文件夹中搜索指定用户的情景记忆和语义记忆。

## 脚本说明

### 1. `test_manual_search.py` - 完整交互式搜索

**功能特点：**
- 🎯 交互式用户界面
- 📋 自动列出所有可用用户
- 🔍 支持情景记忆和语义记忆搜索
- 📊 详细的搜索结果显示
- 💬 显示原始对话内容

**使用方法：**
```bash
cd /path/to/Nemori-code
python tests/test_manual_search.py
```

**交互流程：**
1. 选择用户 (从可用用户列表中选择)
2. 输入搜索查询
3. 设置返回结果数量
4. 选择搜索类型 (情景记忆/语义记忆/两者)
5. 查看详细搜索结果

### 2. `quick_search_test.py` - 快速测试脚本

**功能特点：**
- ⚡ 快速测试特定用户和查询
- 🎯 直接修改代码中的参数
- 📋 简洁的结果显示
- 🚀 适合开发和调试

**使用方法：**
1. 编辑 `quick_search_test.py` 中的参数：
```python
TEST_USER = "Caroline_0"     # 修改为目标用户
TEST_QUERY = "career goals"  # 修改为搜索内容
```

2. 运行脚本：
```bash
cd /path/to/Nemori-code
python tests/quick_search_test.py
```

## 可用用户列表

根据 `evaluation/memories/episodes/` 文件夹中的数据：

- `Audrey_5`
- `Calvin_9` 
- `Caroline_0`
- `Deborah_7`
- `Evan_8`
- `James_6`
- `Joanna_3`
- `John_2`
- `Jon_1`
- `Tim_4`

## 环境要求

### 必需环境变量
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Python依赖
确保已安装项目依赖：
```bash
pip install -r requirements.txt
```

## 搜索结果格式

### 情景记忆结果
```
🔸 结果 1 (相似度: 0.8542)
📅 时间: 2023-03-27T13:10:00
📝 标题: Andrew Shares New Financial Analyst Job Update...
📄 内容: On March 27, 2023, at 1:10 PM, Audrey greeted...
💬 消息数: 2
💭 原始对话:
   Audrey: Hey Andrew! Good to see ya! What's been up...
   Andrew: Hey Audrey! So, I started a new job as a...
```

### 语义记忆结果
```
🔸 结果 1 (相似度: 0.7823)
📅 创建时间: 2023-05-08T13:56:00
🏷️  知识类型: knowledge
📄 内容: Caroline is interested in counseling
🎯 置信度: 0.9
🔗 来源episodes: 1 个
```

## 常见问题

### Q: 搜索结果为空怎么办？
A: 检查以下几点：
1. 用户ID是否正确
2. 查询词是否存在于记忆中
3. 向量索引文件是否存在 (`vector_db/` 文件夹)
4. OpenAI API Key是否正确设置

### Q: 如何调整搜索结果数量？
A: 
- **交互式脚本**: 在运行时输入想要的数量
- **快速脚本**: 修改 `search_all()` 方法中的 `top_k_episodes` 和 `top_k_semantic` 参数

### Q: 相似度分数如何理解？
A: 
- 分数范围通常在 0-1 之间
- 分数越高表示与查询越相似
- 0.7+ 通常表示高度相关
- 0.5-0.7 表示中等相关
- <0.5 表示相关性较低

## 调试技巧

1. **查看详细错误信息**: 脚本会打印完整的错误堆栈
2. **检查数据文件**: 确认 `.jsonl` 文件格式正确
3. **验证向量文件**: 检查 `vector_db/` 中的 `.npy` 和 `.faiss` 文件
4. **测试API连接**: 确认OpenAI API可以正常访问

## 示例查询

### 情景记忆查询示例
- "job interview"
- "meeting with friends" 
- "travel plans"
- "birthday celebration"

### 语义记忆查询示例
- "career goals"
- "favorite books"
- "technical skills"
- "personal preferences"

## 扩展功能

如需添加更多功能，可以修改脚本：
- 添加BM25搜索支持
- 实现混合搜索 (向量+BM25)
- 添加搜索结果导出功能
- 实现批量查询测试
