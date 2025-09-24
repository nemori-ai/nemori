# Nemori 重构路线图

## 阶段 1（已完成）
- [x] 定义领域接口（仓库、索引、生成器、边界检测等）
- [x] 拆分 infrastructure/services 层并实现默认装配器
- [x] MemorySystem 支持依赖注入，默认绑定现有实现
- [x] 基础测试通过（`tests/test_facade.py`）

## 阶段 2：流水线与缓存服务重塑
- [x] 将 Episode → Semantic 生成改为事件/任务驱动（事件总线 + 发布订阅）
- [x] 引入独立的缓存服务（语义、向量、情节）
- [x] 统一任务调度与重试策略，抽象语义生成工作队列
- [x] 扩展测试覆盖事件驱动行为（包括 In-memory 路径）

## 阶段 3：存储与索引扩展
- [x] 为 Episode/Semantic 仓库提供 InMemory 与文件实现
- [x] 为向量/词法索引增加内存实现替代
- [x] 编写契约测试保证抽象接口的一致性
- [x] 提供配置层切换不同实现的示例

## 阶段 4：API 与开箱即用体验
- [x] 扩展 `NemoriMemory` 以支持后端切换
- [x] 编写快速上手示例与文档
- [x] 提供自动化初始化脚本
- [x] 提供异步搜索 API

## 阶段 5：性能与 CI
- [x] 构建基础性能基准脚本
- [x] 引入指标上报（LoggingMetricsReporter）
- [x] 提供 CI 脚本骨架（scripts/run_ci.sh）
- [x] 建立性能记录文档（docs/performance_log.md）

## 阶段 6：发布与维护
- [ ] 整理最终文档与迁移指南
- [ ] 版本化发布（tag、release note）
- [ ] 制定后续维护策略与贡献指南

---
> 当前进度：阶段 1 完成，下一步进入阶段 2。
