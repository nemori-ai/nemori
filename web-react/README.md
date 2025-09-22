# LoCoMo 结果分析器

一个功能强大的React应用，用于分析和可视化LoCoMo记忆检索评测结果。

## 功能特点

- 📊 **智能数据分析** - 自动解析results.json和metrics.json文件
- 📈 **可视化统计** - 准确率、F1/BLEU分数分布、类别性能对比
- 🔍 **详细题目探索** - 每道题的正误、记忆内容、证据对比
- 🎯 **证据覆盖分析** - 检查检索到的记忆是否包含应有的证据
- ⚡ **实时搜索筛选** - 按类别、正确性、关键词快速筛选
- 💻 **响应式设计** - 适配桌面端和移动端

## 快速开始

### 安装依赖

```bash
cd web-react
npm install
```

### 启动开发服务器

```bash
npm start
```

应用将在 http://localhost:3000 打开

### 使用方法

1. **上传文件**
   - Results文件：包含问题、答案、记忆等完整检索结果
   - Metrics文件：包含各项评分指标（BLEU、F1、LLM Judge）
   - 数据集文件（可选）：原始数据集，用于证据对比分析

2. **浏览结果**
   - 查看整体统计和各类别性能
   - 使用搜索和筛选功能快速定位题目
   - 点击展开查看详细记忆内容和证据分析

3. **分析表现**
   - 对比标准答案与生成答案
   - 分析检索到的记忆质量
   - 查看证据覆盖率和遗漏情况

## 文件格式要求

### Results.json 格式
```json
{
  "0": [
    {
      "question": "问题内容",
      "answer": "标准答案", 
      "response": "生成答案",
      "category": "1",
      "speaker_1_memories": [...],
      "speaker_1_memory_time": 0.123,
      "response_time": 0.456
    }
  ]
}
```

### Metrics.json 格式
```json
{
  "0": [
    {
      "question": "问题内容",
      "answer": "标准答案",
      "response": "生成答案", 
      "category": "1",
      "bleu_score": 0.75,
      "f1_score": 0.80,
      "llm_score": 1
    }
  ]
}
```

### 数据集格式（可选）
```json
[
  {
    "conversation": {...},
    "qa": [
      {
        "question": "问题内容",
        "answer": "标准答案",
        "evidence": ["证据1", "证据2"],
        "category": "1"
      }
    ]
  }
]
```

## 技术栈

- **React 18** - 现代化UI框架
- **TypeScript** - 类型安全开发
- **Tailwind CSS** - 现代化样式框架
- **Recharts** - 数据可视化图表
- **Lucide React** - 现代化图标库

## 项目结构

```
src/
├── components/          # React组件
│   ├── FileUpload.tsx      # 文件上传组件
│   ├── StatsDashboard.tsx  # 统计仪表板
│   ├── QuestionExplorer.tsx # 题目探索器
│   ├── MemoryViewer.tsx    # 记忆查看器
│   └── EvidenceComparison.tsx # 证据对比组件
├── types.ts             # TypeScript类型定义
├── App.tsx             # 主应用组件
├── App.css             # 样式文件
└── index.tsx           # 应用入口
```

## 部署

### 构建生产版本

```bash
npm run build
```

构建产物将在 `build/` 目录中生成。

### 部署到静态服务器

将 `build/` 目录的内容部署到任何静态文件服务器（如Nginx、Apache、GitHub Pages等）。

## 开发

### 代码规范

- 使用TypeScript进行类型安全开发
- 遵循React函数组件和Hooks最佳实践
- 使用Tailwind CSS进行样式开发
- 保持组件职责单一，便于维护

### 扩展功能

- 添加更多评分指标的可视化
- 实现数据导出功能（CSV/Excel）
- 添加更sophisticated的证据匹配算法
- 支持多种数据集格式

## 常见问题

**Q: 为什么上传文件后没有反应？**
A: 请检查文件格式是否正确，确保是有效的JSON文件。

**Q: 证据对比功能不准确怎么办？**
A: 当前使用简单的关键词匹配，可以通过上传完整数据集文件来提高准确性。

**Q: 如何查看原始对话内容？**
A: 在记忆详情中点击"显示原始对话"按钮可以展开完整的对话历史。

## 许可证

MIT License - 详见 LICENSE 文件
