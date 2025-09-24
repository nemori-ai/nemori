# Nemori Quickstart

```python
from datetime import datetime
from nemori import NemoriMemory

with NemoriMemory.from_env() as memory:
    user_id = "demo"
    memory.add_messages(
        user_id,
        [
            {"role": "user", "content": "Nemori remembers facts", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "我会记住你喜欢测试。", "timestamp": datetime.now().isoformat()},
        ],
    )
    memory.flush(user_id)
    memory.wait_for_semantic(user_id)
    results = memory.search(user_id, "remembers")
    print(results)
```

- `NemoriMemory.from_env()` 会读取 `OPENAI_API_KEY` 等环境变量并初始化默认存储/索引。
- 通过配置项 `storage_backend`, `vector_index_backend`, `lexical_index_backend` 可切换为纯内存实现，适合本地快速测试：

```python
from nemori import NemoriMemory, MemoryConfig

config = MemoryConfig(
    storage_backend="memory",
    vector_index_backend="memory",
    lexical_index_backend="memory",
)
memory = NemoriMemory(config=config)
```

> 更多示例参见 `examples/quickstart.py`。

> 小贴士：默认配置要求一次至少 2 条消息才能生成情景记忆，因此示例中同时提供了用户与助手的消息。
