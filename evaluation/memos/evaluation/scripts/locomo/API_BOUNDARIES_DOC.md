# Nemori API 文档 - 对话边界检测与记忆更新接口

## 概述

Nemori API 提供了完整的对话处理工作流，包括边界检测和记忆更新功能。现在支持两种处理模式：

1. **完整处理模式**: 自动检测边界 → 构建episodes和语义发现
2. **预分割模式**: 使用预定义边界 → 直接构建episodes和语义发现

## API 端点

### POST /api/boundaries/detect

检测给定消息列表中的对话边界，返回划分的段落信息。

#### 请求格式

```http
POST /api/boundaries/detect
Content-Type: application/json

{
    "version": "string",           // 必需：版本标识符
    "messages": [                  // 必需：消息列表
        {
            "speaker": "string",   // 说话人名称
            "content": "string",   // 消息内容 (或使用 "text")
            "text": "string",      // 消息内容 (content 的替代字段)
            "timestamp": "string"  // 可选：ISO 格式时间戳
        }
    ]
}
```

### POST /api/memory/update-v2 ⭐ 新增

使用预分割的边界信息直接进行记忆更新，跳过边界检测步骤。

#### 请求格式

```http
POST /api/memory/update-v2
Content-Type: application/json

{
    "version": "string",           // 必需：版本标识符
    "messages": [                  // 必需：消息列表（格式同上）
        {
            "speaker": "string",
            "content": "string",
            "timestamp": "string"
        }
    ],
    "boundaries": [                // 必需：预定义的边界信息
        {
            "start_index": 0,      // 必需：段落开始消息索引
            "end_index": 3,        // 必需：段落结束消息索引
            "reason": "string"     // 可选：分割原因描述
        }
    ]
}
```

#### 响应格式

**成功响应 (200 OK):**

```json
{
    "status": "completed",
    "episodes_created": 4,
    "semantic_concepts": 12,
    "processed_speakers": 2,
    "boundary_segments_used": 2,
    "method": "pre_segmented_boundaries"
}
```

**错误响应 (400 Bad Request):**

```json
{
    "status": "error",
    "message": "Boundary at index 0: invalid index range 0-5 for 3 messages."
}
```

## 使用场景对比

### 场景 1: 完整自动处理

适用于：不确定如何分割对话，需要AI自动检测边界

```python
# 第一步：检测边界
boundary_response = requests.post('/api/boundaries/detect', json={
    "version": "auto_v1",
    "messages": messages
})

# 第二步：使用检测结果更新记忆
boundaries = boundary_response.json()["boundaries"]
simplified_boundaries = [
    {
        "start_index": b["start_index"],
        "end_index": b["end_index"], 
        "reason": b["reason"]
    } for b in boundaries
]

memory_response = requests.post('/api/memory/update-v2', json={
    "version": "auto_v1",
    "messages": messages,
    "boundaries": simplified_boundaries
})
```

### 场景 2: 预定义边界处理

适用于：已知对话分割方式，或使用外部算法检测边界

```python
# 直接使用预定义边界更新记忆
manual_boundaries = [
    {"start_index": 0, "end_index": 3, "reason": "项目讨论"},
    {"start_index": 4, "end_index": 7, "reason": "技术交流"}
]

response = requests.post('/api/memory/update-v2', json={
    "version": "manual_v1",
    "messages": messages,
    "boundaries": manual_boundaries
})
```

## 边界格式详解

### 边界对象结构

```json
{
    "start_index": 0,        // 必需：起始消息索引（包含）
    "end_index": 3,          // 必需：结束消息索引（包含）
    "reason": "string"       // 可选：分割原因，用于日志和调试
}
```

### 边界验证规则

1. **索引类型**: `start_index` 和 `end_index` 必须是整数
2. **索引范围**: 索引必须在 `[0, len(messages)-1]` 范围内
3. **逻辑关系**: `start_index <= end_index`
4. **覆盖性**: 建议边界覆盖所有消息，避免遗漏
5. **连续性**: 建议边界连续，避免消息空隙

### 边界示例

```python
# 6条消息的对话，分为3个段落
messages = [msg0, msg1, msg2, msg3, msg4, msg5]

boundaries = [
    {"start_index": 0, "end_index": 1, "reason": "开场寒暄"},     # msg0, msg1
    {"start_index": 2, "end_index": 4, "reason": "技术讨论"},     # msg2, msg3, msg4  
    {"start_index": 5, "end_index": 5, "reason": "结束语"}        # msg5
]
```

## V2 API 的优势

### 🚀 性能优势
- **跳过边界检测**: 减少1-2次LLM调用
- **并行处理**: 可以并行进行边界检测和其他处理
- **批量优化**: 支持批量预处理边界信息

### 🎯 控制优势
- **精确控制**: 可以精确指定对话分割点
- **算法选择**: 支持使用不同的边界检测算法
- **人工干预**: 支持人工审核和调整边界
- **一致性**: 确保相同输入得到一致的分割结果

### 🔄 工作流优势
- **解耦处理**: 边界检测和记忆构建可以分离
- **错误恢复**: 边界检测失败时可以使用备用方案
- **增量处理**: 支持增量添加新的对话段落

## 完整的 Python 示例

```python
import requests
import json
from datetime import datetime, timedelta

API_BASE = "http://localhost:5001"

def create_conversation():
    """创建测试对话"""
    base_time = datetime.now()
    return [
        {
            "speaker": "Alice",
            "content": "今天的项目进展如何？",
            "timestamp": base_time.isoformat() + "Z"
        },
        {
            "speaker": "Bob", 
            "content": "前端已经完成80%，还在调试CSS样式",
            "timestamp": (base_time + timedelta(minutes=2)).isoformat() + "Z"
        },
        {
            "speaker": "Alice",
            "content": "后端API的进度呢？",
            "timestamp": (base_time + timedelta(minutes=4)).isoformat() + "Z"
        },
        # 主题转换
        {
            "speaker": "Bob",
            "content": "对了，你看到昨天的系统性能报告了吗？",
            "timestamp": (base_time + timedelta(hours=2)).isoformat() + "Z"
        }
    ]

def workflow_auto_detection():
    """工作流1: 自动边界检测 + V2更新"""
    messages = create_conversation()
    version = "auto_workflow_v1"
    
    # 步骤1: 检测边界
    print("🔍 检测对话边界...")
    boundary_resp = requests.post(f"{API_BASE}/api/boundaries/detect", json={
        "version": version,
        "messages": messages
    })
    
    if boundary_resp.status_code != 200:
        print(f"❌ 边界检测失败: {boundary_resp.json()}")
        return
    
    boundaries = boundary_resp.json()["boundaries"]
    print(f"✅ 检测到 {len(boundaries)} 个段落")
    
    # 步骤2: 使用V2 API更新记忆
    print("🏗️ 构建episodes和语义记忆...")
    simplified_boundaries = [
        {
            "start_index": b["start_index"],
            "end_index": b["end_index"],
            "reason": b["reason"]
        } for b in boundaries
    ]
    
    memory_resp = requests.post(f"{API_BASE}/api/memory/update-v2", json={
        "version": version,
        "messages": messages,
        "boundaries": simplified_boundaries
    })
    
    if memory_resp.status_code == 200:
        result = memory_resp.json()
        print("✅ 记忆更新成功!")
        print(f"   Episodes: {result['episodes_created']}")
        print(f"   语义概念: {result['semantic_concepts']}")
    else:
        print(f"❌ 记忆更新失败: {memory_resp.json()}")

def workflow_manual_boundaries():
    """工作流2: 手动边界 + V2更新"""
    messages = create_conversation()
    version = "manual_workflow_v1"
    
    # 预定义边界（基于业务逻辑）
    manual_boundaries = [
        {"start_index": 0, "end_index": 2, "reason": "项目进度讨论"},
        {"start_index": 3, "end_index": 3, "reason": "性能报告话题"}
    ]
    
    print("🎯 使用预定义边界更新记忆...")
    memory_resp = requests.post(f"{API_BASE}/api/memory/update-v2", json={
        "version": version,
        "messages": messages,
        "boundaries": manual_boundaries
    })
    
    if memory_resp.status_code == 200:
        result = memory_resp.json()
        print("✅ 记忆更新成功!")
        print(f"   Episodes: {result['episodes_created']}")
        print(f"   语义概念: {result['semantic_concepts']}")
    else:
        print(f"❌ 记忆更新失败: {memory_resp.json()}")

if __name__ == "__main__":
    print("🚀 Nemori API V2 完整示例")
    print("=" * 50)
    
    # 测试自动检测工作流
    workflow_auto_detection()
    
    print("\n" + "=" * 50)
    
    # 测试手动边界工作流
    workflow_manual_boundaries()
```

## 测试工具

我们提供了专门的测试脚本：

1. **test_boundaries_api.py**: 边界检测API测试
2. **test_memory_v2_api.py**: V2记忆更新API测试

```bash
# 测试边界检测API
python test_boundaries_api.py

# 测试V2记忆更新API
python test_memory_v2_api.py
```

## 注意事项

### 性能考虑
- V2 API跳过边界检测，但仍需要进行LLM episode生成和语义发现
- 大量消息的处理仍需要相当的时间
- 建议合理控制单次请求的消息数量（建议≤100条）

### 数据一致性
- 确保边界索引的准确性，错误的索引会导致处理失败
- 边界应该完整覆盖所有消息，避免信息丢失
- 时间戳格式应保持一致

### 错误处理
- 实现客户端重试机制
- 处理网络超时（建议设置120秒以上超时）
- 验证服务器返回的错误信息

## 服务启动

```bash
python test_api.py
```

服务默认运行在 `http://localhost:5001`，包含以下端点：
- `POST /api/boundaries/detect` - 边界检测
- `POST /api/memory/update-v2` - V2记忆更新
- `POST /api/memory/update` - 标准记忆更新  
- `POST /api/memory/query` - 记忆查询
- `GET /health` - 健康检查