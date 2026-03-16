# Nemori Refactoring Design — PostgreSQL + Async + LLM Orchestration

**Date:** 2026-03-16
**Status:** Draft
**Scope:** Full production-grade refactoring of the Nemori memory system library

---

## 1. Context & Decisions

Nemori is an independent Python library for agent memory management. The current implementation uses JSONL file storage, ChromaDB for vector search, BM25 for lexical search, and synchronous threading for concurrency.

### Key Decisions

| Dimension | Decision |
|-----------|----------|
| Positioning | Standalone Python library |
| Storage | PostgreSQL only (remove JSON/file storage) |
| Vector search | pgvector (remove ChromaDB) |
| Full-text search | PostgreSQL tsvector (remove BM25) |
| LLM workflow | Unified orchestration layer |
| Interface design | Redesign domain interfaces around PG capabilities |
| Concurrency | async/await with asyncpg (remove ThreadPoolExecutor) |
| Approach | Phased incremental refactoring |

### Phases

1. **Phase 1** — Storage layer: asyncpg + pgvector, replace JSONL + ChromaDB + BM25
2. **Phase 2** — Interface layer: redesign domain protocols, unify Repository + Index into Store
3. **Phase 3** — LLM orchestration: introduce LLMOrchestrator for retry, rate limiting, token tracking
4. **Phase 4** — Public API async: facade to async, cleanup deprecated code

---

## 2. Project Structure

```
src/
├── api/
│   └── facade.py              # async NemoriMemory public interface
├── config.py                  # Simplified MemoryConfig
├── core/
│   ├── memory_system.py       # Core orchestrator (async)
│   └── message_buffer.py      # Message buffering (delegates to MessageBufferStore)
├── domain/
│   ├── models.py              # Episode, SemanticMemory, Message (merged)
│   ├── interfaces.py          # Redesigned abstract protocols
│   └── exceptions.py          # Exception hierarchy
├── db/
│   ├── connection.py          # asyncpg connection pool lifecycle
│   ├── migrations.py          # Schema versioning + auto-migration
│   ├── episode_store.py       # Episode CRUD + pgvector + full-text
│   └── semantic_store.py      # Semantic CRUD + pgvector + full-text
├── llm/
│   ├── orchestrator.py        # Unified LLM orchestration
│   ├── client.py              # async OpenAI client wrapper
│   ├── prompts.py             # Prompt templates
│   └── generators/
│       ├── episode.py         # Episode generation (prompt + parse only)
│       ├── semantic.py        # Semantic generation
│       └── segmenter.py       # Batch segmentation
├── search/
│   └── unified.py             # Unified search (delegates to db layer)
├── services/
│   ├── event_bus.py           # async event bus
│   └── embedding.py           # async embedding client
└── utils/
    ├── token_counter.py       # Token estimation
    └── text.py                # Text utilities
```

---

## 3. Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Schema versioning
CREATE TABLE schema_migrations (
    version     INT PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Episodes
CREATE TABLE episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(255) NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content,''))
                ) STORED,
    source_messages JSONB,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_episodes_user_id ON episodes(user_id);
CREATE INDEX idx_episodes_embedding ON episodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_episodes_tsv ON episodes USING gin(tsv);
CREATE INDEX idx_episodes_created_at ON episodes(user_id, created_at DESC);

-- Semantic Memories
CREATE TABLE semantic_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(255) NOT NULL,
    content         TEXT NOT NULL,
    memory_type     VARCHAR(50) NOT NULL,
    embedding       vector(1536),
    tsv             tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(content,''))
                    ) STORED,
    source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    confidence      FLOAT DEFAULT 1.0,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_semantic_user_id ON semantic_memories(user_id);
CREATE INDEX idx_semantic_embedding ON semantic_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_semantic_tsv ON semantic_memories USING gin(tsv);
CREATE INDEX idx_semantic_type ON semantic_memories(user_id, memory_type);

-- Message Buffer (persistent, crash-safe)
CREATE TABLE message_buffer (
    id          BIGSERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed   BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_buffer_user_unprocessed ON message_buffer(user_id) WHERE NOT processed;
```

### Design Decisions

- **Generated tsvector**: Auto-maintained full-text index, no manual sync needed
- **HNSW index**: Uses HNSW (not IVFFlat) for pgvector — HNSW works correctly on empty tables and does not require a training step, which is critical since users will always start with an empty database
- **ON DELETE SET NULL**: Deleting an episode nullifies the reference in semantic memories but preserves the semantic memory itself
- **Message buffer in PG**: Current in-memory buffer loses data on crash; PG makes it persistent. Processed messages are deleted after episode generation to prevent unbounded growth
- **JSONB metadata**: Flexible extension without schema changes
- **Configurable vector dimension**: Default 1536, adjustable via config. `migrations.py` must template the dimension from `config.embedding_dimension` when generating DDL. Changing the dimension after data exists requires a migration (ALTER COLUMN + re-embedding all rows)
- **COALESCE in tsvector**: Defensive NULL handling in generated tsvector columns prevents silent search failures if nullable text columns are added in future migrations

### Data Migration (from existing JSONL/ChromaDB)

Phase 1 includes a migration script `scripts/migrate_to_pg.py`:

1. **Read existing JSONL files** — parse `episodes/{user_id}_episodes.jsonl` and `semantic/{user_id}_semantic.jsonl`
2. **Re-embed all content** — existing ChromaDB embeddings are not extracted; instead, re-generate embeddings via the configured embedding provider. This ensures consistency with the new model/dimension.
3. **Bulk insert into PG** — use `asyncpg.copy_records_to_table()` for efficiency
4. **Idempotent** — script uses `ON CONFLICT(id) DO NOTHING`, safe to re-run
5. **Resumable** — tracks progress per user_id, can restart from last completed user

The migration script is a one-time tool, not part of the library's runtime. It reads from the old storage paths and writes to PG.

### Domain Model Field Mapping

#### Episode

| Current field | New PG column | Notes |
|---------------|---------------|-------|
| `episode_id` | `id` (UUID) | Renamed |
| `user_id` | `user_id` | Unchanged |
| `title` | `title` | Unchanged |
| `content` | `content` | Unchanged |
| `embedding` | `embedding` (vector) | Re-generated on migration |
| `messages` | `source_messages` (JSONB) | Renamed, stored as JSONB |
| `timestamp` | `created_at` | Mapped to TIMESTAMPTZ |
| `boundary_reason` | `metadata->>'boundary_reason'` | Moved to JSONB metadata |
| `message_count` | Derived from `jsonb_array_length(source_messages)` | No longer a stored field |
| `tags` | `metadata->'tags'` | Moved to JSONB metadata |

#### SemanticMemory

| Current field | New PG column | Notes |
|---------------|---------------|-------|
| `memory_id` | `id` (UUID) | Renamed |
| `user_id` | `user_id` | Unchanged |
| `content` | `content` | Unchanged |
| `knowledge_type` | `memory_type` | Renamed for clarity |
| `embedding` | `embedding` (vector) | Re-generated on migration |
| `source_episodes` (list[str]) | `source_episode_id` (single UUID FK) | **Simplified**: in the current generation flow, each semantic memory is always derived from exactly one episode. The old list type was over-general. |
| `confidence` | `confidence` | Unchanged |
| `revision_count` | Dropped | Not used in current logic; `updated_at` tracks changes |
| `tags` | `metadata->'tags'` | Moved to JSONB metadata |

---

## 4. Domain Interfaces

```python
# domain/interfaces.py

@runtime_checkable
class EpisodeStore(Protocol):
    """Unified episode persistence + search"""

    async def save(self, episode: Episode) -> None: ...
    async def get(self, episode_id: str) -> Episode | None: ...
    async def list_by_user(self, user_id: str, limit: int = 100, offset: int = 0) -> list[Episode]: ...
    async def delete(self, episode_id: str) -> None: ...
    async def delete_by_user(self, user_id: str) -> None: ...
    async def search_by_vector(self, user_id: str, embedding: list[float], top_k: int) -> list[Episode]: ...
    async def search_by_text(self, user_id: str, query: str, top_k: int) -> list[Episode]: ...
    async def search_hybrid(self, user_id: str, query: str, embedding: list[float], top_k: int) -> list[Episode]: ...


@runtime_checkable
class SemanticStore(Protocol):
    """Unified semantic memory persistence + search"""

    async def save(self, memory: SemanticMemory) -> None: ...
    async def save_batch(self, memories: list[SemanticMemory]) -> None: ...
    async def get(self, memory_id: str) -> SemanticMemory | None: ...
    async def list_by_user(self, user_id: str, memory_type: str | None = None) -> list[SemanticMemory]: ...
    async def delete(self, memory_id: str) -> None: ...
    async def delete_by_user(self, user_id: str) -> None: ...
    async def search_by_vector(self, user_id: str, embedding: list[float], top_k: int) -> list[SemanticMemory]: ...
    async def search_by_text(self, user_id: str, query: str, top_k: int) -> list[SemanticMemory]: ...
    async def search_hybrid(self, user_id: str, query: str, embedding: list[float], top_k: int) -> list[SemanticMemory]: ...


@runtime_checkable
class MessageBufferStore(Protocol):
    """Persistent message buffer"""

    async def push(self, user_id: str, messages: list[Message]) -> None: ...
    async def get_unprocessed(self, user_id: str) -> list[Message]: ...
    async def mark_processed(self, user_id: str, message_ids: list[int]) -> None: ...
    async def count_unprocessed(self, user_id: str) -> int: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, messages: list[dict], **kwargs) -> str: ...
```

### Key Changes from Current

| Current | After | Reason |
|---------|-------|--------|
| EpisodeRepository + VectorIndex + LexicalIndex | EpisodeStore | PG unifies persistence and search |
| SemanticRepository + VectorIndex + LexicalIndex | SemanticStore | Same |
| No buffer protocol | MessageBufferStore | Buffer now persistent in PG |
| DedupStrategy Protocol | Removed | Single strategy, inline in SemanticGenerator |
| LLMClient concrete class | LLMProvider Protocol + LLMOrchestrator | Decouple call protocol from orchestration |
| EmbeddingClient concrete class | EmbeddingProvider Protocol | Testable, replaceable |

---

## 5. LLM Orchestration Layer

### Architecture

```
Generator (defines prompt + parses response)
    ↓ submits LLMRequest
LLMOrchestrator (unified dispatch)
    ├── Retry strategy (exponential backoff + jitter)
    ├── Concurrency limiting (asyncio.Semaphore)
    ├── Token budget tracking
    ├── Structured logging (request_id, model, latency, tokens)
    └── Per-request timeout
    ↓ calls
LLMProvider (async complete)
```

### Core Types

```python
@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[dict, ...]         # tuple for true immutability
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    response_format: type | None = None
    timeout: float = 30.0
    retries: int = 3
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    request_id: str


class LLMOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        default_model: str,
        max_concurrent: int = 10,
        token_budget: int | None = None,
        logger: logging.Logger | None = None,
    ): ...

    async def execute(self, request: LLMRequest) -> LLMResponse: ...
    async def execute_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]: ...

    @property
    def stats(self) -> OrchestratorStats: ...
```

### Retry Strategy

- Retryable: 429, 500, 502, 503, 504
- Non-retryable: 400, 401, 403, 404
- Backoff: `min(base * 2^attempt + random(0, jitter), max_delay)`
- Defaults: base=1s, jitter=0.5s, max_delay=30s

### Generator Pattern

Generators become pure prompt-constructors + response-parsers. They receive an `LLMOrchestrator` and delegate all call mechanics to it:

```python
class EpisodeGenerator:
    def __init__(self, orchestrator: LLMOrchestrator, embedding: EmbeddingProvider): ...

    async def generate(self, user_id: str, messages: list[Message], boundary_reason: str) -> Episode:
        request = LLMRequest(
            messages=self._build_prompt(messages, boundary_reason),
            response_format=EpisodeSchema,
            metadata={"generator": "episode", "user_id": user_id},
        )
        response = await self.orchestrator.execute(request)
        return self._parse_response(user_id, response, messages)
```

---

## 6. Async Core & Public API

### MemorySystem

```python
class MemorySystem:
    def __init__(
        self,
        config: MemoryConfig,
        db: DatabaseManager,
        episode_store: EpisodeStore,
        semantic_store: SemanticStore,
        buffer_store: MessageBufferStore,
        orchestrator: LLMOrchestrator,
        embedding: EmbeddingProvider,
        episode_generator: EpisodeGenerator,
        semantic_generator: SemanticGenerator,
        event_bus: EventBus,
    ): ...

    async def add_messages(self, user_id: str, messages: list[Message]) -> None: ...
    async def flush(self, user_id: str) -> list[Episode]: ...
    async def search(self, user_id: str, query: str, ...) -> SearchResult: ...
    async def delete_episode(self, user_id: str, episode_id: str) -> None: ...
    async def delete_semantic(self, user_id: str, memory_id: str) -> None: ...
    async def delete_user(self, user_id: str) -> None: ...
    async def stats(self, user_id: str) -> SystemStats: ...
    async def drain(self, timeout: float = 30.0) -> None: ...
```

### Concurrency

- `asyncio.Lock` per user_id (replaces `threading.RLock`)
- `asyncio.create_task` for background semantic generation (replaces `ThreadPoolExecutor`)
- Lock creation is lazy: `dict[str, asyncio.Lock]` with LRU eviction (max 10,000 entries) to prevent unbounded memory growth
- All access must happen from a single event loop (documented requirement)

### Store Upsert Semantics

`save()` on both `EpisodeStore` and `SemanticStore` uses `INSERT ... ON CONFLICT(id) DO UPDATE` (upsert). This means:
- First call with a given `id` creates the record
- Subsequent calls with the same `id` update the record
- No separate `update()` method needed — `save()` handles both cases

### Public Facade

```python
class NemoriMemory:
    def __init__(self, config: MemoryConfig | None = None): ...

    async def __aenter__(self) -> "NemoriMemory": ...
    async def __aexit__(self, *exc) -> None: ...

    async def add_messages(self, user_id: str, messages: list[dict]) -> None: ...
    async def flush(self, user_id: str) -> list[dict]: ...
    async def search(self, user_id: str, query: str, **kwargs) -> dict: ...
    async def delete_episode(self, user_id: str, episode_id: str) -> None: ...
    async def delete_semantic(self, user_id: str, memory_id: str) -> None: ...
    async def delete_user(self, user_id: str) -> None: ...
    async def stats(self, user_id: str) -> dict: ...
    async def health(self) -> HealthResult: ...
```

`dsn` is read from `config.dsn` — no separate parameter to avoid ambiguity.

`health()` returns a structured `HealthResult` with separate `healthy: bool` (checks db/llm/embedding) and `diagnostics: dict` (pool_size, pool_free, etc.) to avoid the `all(status.values())` pitfall.

Usage:

```python
async with NemoriMemory(config=MemoryConfig(dsn="postgresql://localhost/nemori")) as memory:
    await memory.add_messages("user_1", [
        {"role": "user", "content": "I like hiking"},
        {"role": "assistant", "content": "Hiking is great exercise!"},
    ])
    await memory.flush("user_1")
    results = await memory.search("user_1", "outdoor activities")
```

---

## 7. Configuration

```python
@dataclass
class MemoryConfig:
    # Database
    dsn: str = "postgresql://localhost/nemori"
    db_pool_min: int = 5
    db_pool_max: int = 20

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_max_concurrent: int = 10
    llm_timeout: float = 30.0
    llm_retries: int = 3
    llm_token_budget: int | None = None

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str | None = None
    embedding_dimension: int = 1536

    # Buffer & Generation
    buffer_size_min: int = 2
    buffer_size_max: int = 25
    enable_batch_segmentation: bool = True
    batch_threshold: int = 20
    episode_min_messages: int = 2
    episode_max_messages: int = 25

    # Semantic Memory
    enable_semantic_memory: bool = True
    enable_prediction_correction: bool = True
    semantic_similarity_threshold: float = 0.85

    # Search
    search_top_k_episodes: int = 10
    search_top_k_semantic: int = 10
```

### Removed Settings

| Removed | Reason |
|---------|--------|
| `storage_backend` | PostgreSQL only |
| `vector_index_backend` | pgvector replaces ChromaDB |
| `lexical_index_backend` | PG tsvector replaces BM25 |
| `chroma_persist_directory` | No ChromaDB |
| `storage_path` | No file storage |
| `enable_cache` / `cache_size` / `cache_ttl` | Rely on PG, add app-level cache later by profiling |
| `enable_episode_merging` / `merge_*` | Simplified: dedup inline in semantic layer |
| `enable_norlift_ranking` / `norlift_*` | Add back in future iteration |
| `extract_semantic_per_episode` | Always per-episode in new flow; config flag unnecessary |
| `language` | Prompts consolidated to English; multi-language support deferred to future iteration |
| `enable_parallel_search` | async is inherently concurrent |
| `max_workers` / `semantic_generation_workers` | No thread pool |
| `batch_size` | PG handles batch operations natively |

---

## 8. Error Handling & Production Guarantees

### Exception Hierarchy

```python
class NemoriError(Exception): ...
class DatabaseError(NemoriError): ...
class LLMError(NemoriError): ...
class LLMRateLimitError(LLMError):
    retry_after: float | None = None
class LLMAuthError(LLMError): ...
class TokenBudgetExceeded(LLMError):
    used: int
    budget: int
class EmbeddingError(NemoriError): ...
class ConfigError(NemoriError): ...
class UserNotFoundError(NemoriError): ...
```

### Principles

- Orchestrator retries internally; raises specific exception after exhausting retries
- Generators have fallback logic (e.g., raw episode on generation failure); raise only if fallback also fails
- Facade transparently propagates all exceptions to caller
- All exceptions inherit `NemoriError` for catch-all handling

### Logging

- Standard `logging.getLogger("nemori")` — no handler configuration (caller's responsibility)
- Key log points: LLM requests/responses, episode generation, search queries, errors

### Graceful Shutdown

```python
async def __aexit__(self, *exc):
    await self._system.drain(timeout=30.0)  # wait for background tasks
    await self._db.close()                   # close connection pool
```

### Health Check

```python
@dataclass
class HealthResult:
    healthy: bool                  # True if db + llm + embedding all OK
    db: bool
    llm: bool
    embedding: bool
    diagnostics: dict              # pool_size, pool_free, etc.

async def health(self) -> HealthResult:
    db_ok = await self._db.ping()
    llm_ok = await self._check_llm()
    embed_ok = await self._check_embed()
    return HealthResult(
        healthy=db_ok and llm_ok and embed_ok,
        db=db_ok, llm=llm_ok, embedding=embed_ok,
        diagnostics={
            "pool_size": self._db.pool.get_size(),
            "pool_free": self._db.pool.get_idle_size(),
        },
    )
```

---

## 9. Removal List

### Directories to Delete

- `src/storage/` (entire directory)
- `src/infrastructure/` (entire directory)
- `src/models/` (merged into `domain/models.py`)

### Files to Delete

- `src/search/chroma_search.py`
- `src/search/bm25_search.py`
- `src/search/episode_original_message_search.py`
- `src/search/original_message_search.py`
- `src/services/cache.py`
- `src/services/providers.py`
- `src/services/task_manager.py`
- `src/services/metrics.py`
- `src/utils/performance.py`
- `src/generation/episode_merger.py`
- `src/generation/prediction_correction_engine.py`

### Files to Rewrite

- `src/core/memory_system.py` → async
- `src/core/message_buffer.py` → delegates to MessageBufferStore
- `src/api/facade.py` → async
- `src/config.py` → simplified
- `src/domain/interfaces.py` → new protocols
- `src/services/event_bus.py` → async (sync callbacks → async handlers)

### Note on prediction_correction_engine.py

The file `src/generation/prediction_correction_engine.py` is deleted, but `enable_prediction_correction` remains in config. The prediction-correction logic moves into `src/llm/generators/semantic.py` as an internal implementation detail of the semantic generator.

### Files to Move/Rename

- `src/generation/` → `src/llm/`
- `src/utils/llm_client.py` → `src/llm/client.py`
- `src/utils/embedding_client.py` → `src/services/embedding.py`
- `src/search/unified_search.py` → `src/search/unified.py`

### New Files

- `src/db/connection.py`
- `src/db/migrations.py`
- `src/db/episode_store.py`
- `src/db/semantic_store.py`
- `src/llm/orchestrator.py`
- `src/domain/exceptions.py`
- `src/domain/models.py`

### Dependency Changes

Remove: `chromadb`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `aiofiles`
Add: `asyncpg`, `pgvector`
Keep: `openai`, `tiktoken`, `pydantic`, `python-dotenv`

---

## 10. Usage Example (Post-Refactoring)

```python
import asyncio
from nemori import NemoriMemory, MemoryConfig

async def main():
    config = MemoryConfig(
        dsn="postgresql://user:pass@localhost/nemori",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )

    async with NemoriMemory(config=config) as memory:
        # Health check
        health = await memory.health()
        print(f"System healthy: {health.healthy}")

        # Add conversation
        await memory.add_messages("alice", [
            {"role": "user", "content": "I just moved to Tokyo last month"},
            {"role": "assistant", "content": "How exciting! How are you finding life in Tokyo?"},
            {"role": "user", "content": "Love it! The food is amazing, especially ramen"},
        ])

        # Force processing
        episodes = await memory.flush("alice")
        print(f"Created {len(episodes)} episodes")

        # Search
        results = await memory.search("alice", "Where does Alice live?")
        print(results)

        # Stats
        stats = await memory.stats("alice")
        print(stats)

asyncio.run(main())
```
