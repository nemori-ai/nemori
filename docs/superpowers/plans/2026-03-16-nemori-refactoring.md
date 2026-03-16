# Nemori Refactoring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Nemori from JSONL/ChromaDB/BM25/sync to PostgreSQL/pgvector/tsvector/async with unified LLM orchestration.

**Architecture:** 4-phase incremental refactoring — Phase 1 (DB + domain models), Phase 2 (domain interfaces + store implementations), Phase 3 (LLM orchestrator + generators), Phase 4 (async core + facade + cleanup). Each phase produces working, testable code.

**Tech Stack:** Python 3.10+, asyncpg, pgvector, OpenAI async client, Pydantic v2, pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-16-nemori-refactoring-design.md`

---

## File Structure

### New Files to Create
- `src/domain/models.py` — Unified Message, Episode, SemanticMemory dataclasses
- `src/domain/exceptions.py` — NemoriError hierarchy
- `src/domain/interfaces.py` — Protocol definitions (EpisodeStore, SemanticStore, etc.)
- `src/db/__init__.py` — Package init
- `src/db/connection.py` — DatabaseManager (asyncpg pool lifecycle)
- `src/db/migrations.py` — Schema versioning + auto-migration
- `src/db/episode_store.py` — PgEpisodeStore (CRUD + pgvector + tsvector)
- `src/db/semantic_store.py` — PgSemanticStore (CRUD + pgvector + tsvector)
- `src/db/buffer_store.py` — PgMessageBufferStore
- `src/llm/__init__.py` — Package init
- `src/llm/orchestrator.py` — LLMOrchestrator (retry, semaphore, token budget)
- `src/llm/client.py` — AsyncLLMClient implementing LLMProvider
- `src/llm/prompts.py` — Prompt templates (migrated from generation/)
- `src/llm/generators/__init__.py` — Package init
- `src/llm/generators/episode.py` — EpisodeGenerator (prompt + parse)
- `src/llm/generators/semantic.py` — SemanticGenerator (with prediction-correction)
- `src/llm/generators/segmenter.py` — BatchSegmenter
- `src/search/unified.py` — UnifiedSearch (delegates to stores)
- `src/services/embedding.py` — AsyncEmbeddingClient implementing EmbeddingProvider
- `src/services/event_bus.py` — AsyncEventBus (rewrite)
- `src/utils/token_counter.py` — Token estimation utility (rewrite: simplified)
- `src/utils/text.py` — Text utilities (renamed from text_utils.py)
- `tests/conftest.py` — Shared fixtures (PG test database, mock providers)
- `tests/test_models.py` — Domain model tests
- `tests/test_exceptions.py` — Exception hierarchy tests
- `tests/test_connection.py` — DatabaseManager tests
- `tests/test_migrations.py` — Migration tests
- `tests/test_episode_store.py` — PgEpisodeStore tests
- `tests/test_semantic_store.py` — PgSemanticStore tests
- `tests/test_buffer_store.py` — PgMessageBufferStore tests
- `tests/test_orchestrator.py` — LLMOrchestrator tests
- `tests/test_generators.py` — Generator tests
- `tests/test_search.py` — UnifiedSearch tests
- `tests/test_memory_system.py` — MemorySystem integration tests
- `tests/test_facade.py` — NemoriMemory facade tests

### Files to Rewrite
- `src/config.py` — Simplified MemoryConfig
- `src/core/memory_system.py` — Async MemorySystem
- `src/core/message_buffer.py` — Delegates to MessageBufferStore
- `src/api/facade.py` — Async NemoriMemory

### Files to Delete (Phase 4)
- `src/storage/` (entire directory)
- `src/infrastructure/` (entire directory)
- `src/models/` (entire directory — replaced by domain/models.py)
- `src/search/chroma_search.py`, `src/search/bm25_search.py`, `src/search/episode_original_message_search.py`, `src/search/original_message_search.py`
- `src/services/cache.py`, `src/services/providers.py`, `src/services/task_manager.py`, `src/services/metrics.py`
- `src/utils/performance.py`, `src/utils/llm_client.py`, `src/utils/embedding_client.py`
- `src/generation/` (entire directory — replaced by llm/)

---

## Chunk 1: Foundation (Domain Models, Exceptions, Config, DB Connection)

### Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

```toml
[project]
name = "nemori"
version = "0.2.0"
requires-python = ">=3.10"

dependencies = [
    "asyncpg>=0.29.0",
    "pgvector>=0.3.0",
    "openai>=1.0.0",
    "tiktoken>=0.5.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "black>=23.0",
]
eval = [
    "nltk",
    "bert-score",
    "rouge-score",
    "pandas",
    "spacy>=3.7.0",
]
```

Remove from dependencies: `chromadb`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `aiofiles`.

- [ ] **Step 2: Install updated dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful installation with asyncpg, pgvector, pytest-asyncio

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update dependencies for PostgreSQL + async refactoring"
```

---

### Task 2: Domain Exceptions

**Files:**
- Create: `src/domain/exceptions.py`
- Test: `tests/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exceptions.py
"""Tests for Nemori exception hierarchy."""
import pytest
from src.domain.exceptions import (
    NemoriError,
    DatabaseError,
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    TokenBudgetExceeded,
    EmbeddingError,
    ConfigError,
    UserNotFoundError,
)


def test_all_exceptions_inherit_nemori_error():
    """All custom exceptions should be catchable via NemoriError."""
    exceptions = [
        DatabaseError("db fail"),
        LLMError("llm fail"),
        LLMRateLimitError("rate limited"),
        LLMAuthError("auth fail"),
        TokenBudgetExceeded("budget exceeded", used=5000, budget=1000),
        EmbeddingError("embed fail"),
        ConfigError("config fail"),
        UserNotFoundError("user not found"),
    ]
    for exc in exceptions:
        assert isinstance(exc, NemoriError)


def test_llm_rate_limit_error_has_retry_after():
    exc = LLMRateLimitError("rate limited", retry_after=30.0)
    assert exc.retry_after == 30.0
    assert isinstance(exc, LLMError)


def test_llm_rate_limit_error_default_retry_after():
    exc = LLMRateLimitError("rate limited")
    assert exc.retry_after is None


def test_token_budget_exceeded_has_usage_info():
    exc = TokenBudgetExceeded("over budget", used=5000, budget=1000)
    assert exc.used == 5000
    assert exc.budget == 1000
    assert isinstance(exc, LLMError)


def test_nemori_error_is_exception():
    with pytest.raises(NemoriError):
        raise DatabaseError("test")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exceptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.domain.exceptions'`

- [ ] **Step 3: Write implementation**

```python
# src/domain/exceptions.py
"""Nemori exception hierarchy."""


class NemoriError(Exception):
    """Base exception for all Nemori errors."""


class DatabaseError(NemoriError):
    """Connection failure, query failure, migration failure."""


class LLMError(NemoriError):
    """Base for LLM call errors."""


class LLMRateLimitError(LLMError):
    """429 — caller can use retry_after to decide backoff."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """401/403 — invalid API key."""


class TokenBudgetExceeded(LLMError):
    """Token budget exhausted."""

    def __init__(self, message: str, used: int, budget: int):
        super().__init__(message)
        self.used = used
        self.budget = budget


class EmbeddingError(NemoriError):
    """Embedding generation failure."""


class ConfigError(NemoriError):
    """Configuration validation failure."""


class UserNotFoundError(NemoriError):
    """No data for the given user_id."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_exceptions.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/exceptions.py tests/test_exceptions.py
git commit -m "feat: add Nemori exception hierarchy"
```

---

### Task 3: Domain Models

**Files:**
- Create: `src/domain/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
"""Tests for domain models."""
import pytest
from datetime import datetime, timezone
from src.domain.models import Message, Episode, SemanticMemory, HealthResult


class TestMessage:
    def test_create_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.message_id  # auto-generated
        assert isinstance(msg.timestamp, datetime)

    def test_message_from_dict(self):
        data = {"role": "user", "content": "hi", "timestamp": "2024-01-01T10:00:00"}
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "hi"

    def test_message_to_dict(self):
        msg = Message(role="assistant", content="world")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "world"
        assert "message_id" in d
        assert "timestamp" in d


class TestEpisode:
    def test_create_episode(self):
        ep = Episode(
            user_id="u1",
            title="Test",
            content="Test content",
            source_messages=[{"role": "user", "content": "hi"}],
        )
        assert ep.id  # auto-generated UUID
        assert ep.user_id == "u1"
        assert ep.title == "Test"
        assert ep.metadata == {}

    def test_episode_from_dict(self):
        data = {
            "id": "abc-123",
            "user_id": "u1",
            "title": "Test",
            "content": "Content",
            "source_messages": [],
            "created_at": "2024-01-01T10:00:00",
        }
        ep = Episode.from_dict(data)
        assert ep.id == "abc-123"

    def test_episode_to_dict(self):
        ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
        d = ep.to_dict()
        assert "id" in d
        assert d["user_id"] == "u1"
        assert "created_at" in d

    def test_episode_metadata_stores_boundary_reason(self):
        ep = Episode(
            user_id="u1",
            title="T",
            content="C",
            source_messages=[],
            metadata={"boundary_reason": "topic_change"},
        )
        assert ep.metadata["boundary_reason"] == "topic_change"


class TestSemanticMemory:
    def test_create_semantic_memory(self):
        sm = SemanticMemory(
            user_id="u1",
            content="User likes hiking",
            memory_type="preference",
        )
        assert sm.id  # auto-generated
        assert sm.memory_type == "preference"
        assert sm.confidence == 1.0
        assert sm.source_episode_id is None

    def test_semantic_memory_with_source(self):
        sm = SemanticMemory(
            user_id="u1",
            content="User works at Google",
            memory_type="identity",
            source_episode_id="ep-123",
            confidence=0.9,
        )
        assert sm.source_episode_id == "ep-123"
        assert sm.confidence == 0.9

    def test_semantic_memory_to_dict(self):
        sm = SemanticMemory(
            user_id="u1", content="fact", memory_type="identity"
        )
        d = sm.to_dict()
        assert d["memory_type"] == "identity"
        assert "id" in d


class TestHealthResult:
    def test_healthy_when_all_ok(self):
        hr = HealthResult(db=True, llm=True, embedding=True, diagnostics={})
        assert hr.healthy is True

    def test_unhealthy_when_db_down(self):
        hr = HealthResult(db=False, llm=True, embedding=True, diagnostics={})
        assert hr.healthy is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/domain/models.py
"""Domain models for Nemori memory system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class Message:
    """A single conversation message."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=ts,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Episode:
    """An episodic memory derived from conversation messages."""

    user_id: str
    title: str
    content: str
    source_messages: list[dict[str, Any]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "source_messages": self.source_messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif created is None:
            created = datetime.now()
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        elif updated is None:
            updated = datetime.now()
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data["user_id"],
            title=data["title"],
            content=data["content"],
            source_messages=data.get("source_messages", []),
            metadata=data.get("metadata", {}),
            created_at=created,
            updated_at=updated,
        )


@dataclass
class SemanticMemory:
    """A semantic knowledge fact extracted from episodes."""

    user_id: str
    content: str
    memory_type: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None
    source_episode_id: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_episode_id": self.source_episode_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticMemory:
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        elif created is None:
            created = datetime.now()
        updated = data.get("updated_at")
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        elif updated is None:
            updated = datetime.now()
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data["user_id"],
            content=data["content"],
            memory_type=data.get("memory_type", data.get("knowledge_type", "")),
            source_episode_id=data.get("source_episode_id"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            created_at=created,
            updated_at=updated,
        )


@dataclass
class HealthResult:
    """Health check result for the Nemori system."""

    db: bool
    llm: bool
    embedding: bool
    diagnostics: dict[str, Any]

    @property
    def healthy(self) -> bool:
        return self.db and self.llm and self.embedding
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/test_models.py
git commit -m "feat: add new domain models (Message, Episode, SemanticMemory, HealthResult)"
```

---

### Task 4: Simplified MemoryConfig

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for simplified MemoryConfig."""
import os
import pytest
from src.config import MemoryConfig


def test_default_config():
    cfg = MemoryConfig()
    assert cfg.dsn == "postgresql://localhost/nemori"
    assert cfg.db_pool_min == 5
    assert cfg.db_pool_max == 20
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dimension == 1536
    assert cfg.buffer_size_min == 2
    assert cfg.search_top_k_episodes == 10


def test_config_reads_env_for_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    cfg = MemoryConfig()
    assert cfg.llm_api_key == "test-key-123"


def test_config_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")
    cfg = MemoryConfig()
    assert cfg.llm_api_key == "openai-fallback"


def test_config_custom_dsn():
    cfg = MemoryConfig(dsn="postgresql://user:pass@db:5432/mydb")
    assert cfg.dsn == "postgresql://user:pass@db:5432/mydb"


def test_config_no_removed_fields():
    """Ensure old fields are gone."""
    cfg = MemoryConfig()
    assert not hasattr(cfg, "storage_backend")
    assert not hasattr(cfg, "vector_index_backend")
    assert not hasattr(cfg, "lexical_index_backend")
    assert not hasattr(cfg, "chroma_persist_directory")
    assert not hasattr(cfg, "storage_path")
    assert not hasattr(cfg, "enable_episode_merging")
    assert not hasattr(cfg, "max_workers")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_config.py -v`
Expected: FAIL — old config has different fields

- [ ] **Step 3: Rewrite src/config.py**

```python
# src/config.py
"""Nemori configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _resolve_llm_key() -> str:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


def _resolve_embedding_key() -> str:
    return (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


@dataclass
class MemoryConfig:
    """Configuration for the Nemori memory system."""

    # Database
    dsn: str = "postgresql://localhost/nemori"
    db_pool_min: int = 5
    db_pool_max: int = 20

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = field(default_factory=_resolve_llm_key)
    llm_base_url: str | None = None
    llm_max_concurrent: int = 10
    llm_timeout: float = 30.0
    llm_retries: int = 3
    llm_token_budget: int | None = None

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = field(default_factory=_resolve_embedding_key)
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_config.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "refactor: simplify MemoryConfig for PostgreSQL-only backend"
```

---

### Task 5: Domain Interfaces (Protocols)

**Files:**
- Rewrite: `src/domain/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interfaces.py
"""Tests for domain interface protocols."""
import pytest
from typing import runtime_checkable
from src.domain.interfaces import (
    EpisodeStore,
    SemanticStore,
    MessageBufferStore,
    EmbeddingProvider,
    LLMProvider,
)


def test_protocols_are_runtime_checkable():
    """All protocols should be runtime_checkable."""
    assert hasattr(EpisodeStore, "__protocol_attrs__") or hasattr(
        EpisodeStore, "_is_runtime_protocol"
    )
    assert hasattr(SemanticStore, "__protocol_attrs__") or hasattr(
        SemanticStore, "_is_runtime_protocol"
    )
    assert hasattr(MessageBufferStore, "__protocol_attrs__") or hasattr(
        MessageBufferStore, "_is_runtime_protocol"
    )
    assert hasattr(EmbeddingProvider, "__protocol_attrs__") or hasattr(
        EmbeddingProvider, "_is_runtime_protocol"
    )
    assert hasattr(LLMProvider, "__protocol_attrs__") or hasattr(
        LLMProvider, "_is_runtime_protocol"
    )


def test_episode_store_has_required_methods():
    """EpisodeStore should define all CRUD + search methods."""
    required = [
        "save", "get", "list_by_user", "delete", "delete_by_user",
        "search_by_vector", "search_by_text", "search_hybrid",
    ]
    for method in required:
        assert hasattr(EpisodeStore, method), f"Missing: {method}"


def test_semantic_store_has_required_methods():
    required = [
        "save", "save_batch", "get", "list_by_user", "delete",
        "delete_by_user", "search_by_vector", "search_by_text", "search_hybrid",
    ]
    for method in required:
        assert hasattr(SemanticStore, method), f"Missing: {method}"


def test_message_buffer_store_has_required_methods():
    required = ["push", "get_unprocessed", "mark_processed", "count_unprocessed"]
    for method in required:
        assert hasattr(MessageBufferStore, method), f"Missing: {method}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_interfaces.py -v`
Expected: FAIL — old interfaces.py has different classes

- [ ] **Step 3: Rewrite src/domain/interfaces.py**

```python
# src/domain/interfaces.py
"""Domain protocols for the Nemori memory system."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain.models import Episode, SemanticMemory, Message


@runtime_checkable
class EpisodeStore(Protocol):
    """Unified episode persistence + search."""

    async def save(self, episode: Episode) -> None: ...
    async def get(self, episode_id: str) -> Episode | None: ...
    async def list_by_user(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]: ...
    async def delete(self, episode_id: str) -> None: ...
    async def delete_by_user(self, user_id: str) -> None: ...
    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[Episode]: ...
    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[Episode]: ...
    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[Episode]: ...


@runtime_checkable
class SemanticStore(Protocol):
    """Unified semantic memory persistence + search."""

    async def save(self, memory: SemanticMemory) -> None: ...
    async def save_batch(self, memories: list[SemanticMemory]) -> None: ...
    async def get(self, memory_id: str) -> SemanticMemory | None: ...
    async def list_by_user(
        self, user_id: str, memory_type: str | None = None
    ) -> list[SemanticMemory]: ...
    async def delete(self, memory_id: str) -> None: ...
    async def delete_by_user(self, user_id: str) -> None: ...
    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]: ...
    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]: ...
    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]: ...


@runtime_checkable
class MessageBufferStore(Protocol):
    """Persistent message buffer."""

    async def push(self, user_id: str, messages: list[Message]) -> None: ...
    async def get_unprocessed(self, user_id: str) -> list[Message]: ...
    async def mark_processed(self, user_id: str, message_ids: list[int]) -> None: ...
    async def count_unprocessed(self, user_id: str) -> int: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding generation protocol."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """LLM call protocol."""

    async def complete(self, messages: list[dict], **kwargs: object) -> str: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_interfaces.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/interfaces.py tests/test_interfaces.py
git commit -m "refactor: redesign domain interfaces as unified Store protocols"
```

---

### Task 6: DatabaseManager (Connection Pool)

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/connection.py`
- Test: `tests/test_connection.py`

- [ ] **Step 1: Create package init**

```python
# src/db/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_connection.py
"""Tests for DatabaseManager."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.db.connection import DatabaseManager
from src.domain.exceptions import DatabaseError


@pytest.mark.asyncio
async def test_init_creates_pool():
    """DatabaseManager.init() should create an asyncpg pool."""
    dm = DatabaseManager()
    mock_pool = AsyncMock()
    mock_pool.get_size.return_value = 5
    mock_pool.get_idle_size.return_value = 5

    with patch("src.db.connection.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        await dm.init("postgresql://localhost/test")
        assert dm.pool is not None
    await dm.close()


@pytest.mark.asyncio
async def test_close_releases_pool():
    dm = DatabaseManager()
    mock_pool = AsyncMock()
    with patch("src.db.connection.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        await dm.init("postgresql://localhost/test")
        await dm.close()
        mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_ping_returns_bool():
    dm = DatabaseManager()
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=1)
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch("src.db.connection.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        await dm.init("postgresql://localhost/test")
        result = await dm.ping()
        assert result is True
    await dm.close()


@pytest.mark.asyncio
async def test_operations_before_init_raise():
    dm = DatabaseManager()
    with pytest.raises(DatabaseError):
        await dm.ping()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_connection.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Write implementation**

```python
# src/db/connection.py
"""asyncpg connection pool lifecycle management."""
from __future__ import annotations

import logging
from typing import Any

import asyncpg

from src.domain.exceptions import DatabaseError

logger = logging.getLogger("nemori")


class DatabaseManager:
    """Manages asyncpg connection pool lifecycle."""

    def __init__(self) -> None:
        self.pool: asyncpg.Pool | None = None

    async def init(
        self,
        dsn: str,
        min_size: int = 5,
        max_size: int = 20,
    ) -> None:
        """Create the connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn, min_size=min_size, max_size=max_size
            )
            logger.info("Database pool created (min=%d, max=%d)", min_size, max_size)
        except Exception as e:
            raise DatabaseError(f"Failed to create connection pool: {e}") from e

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
            self.pool = None

    def _ensure_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise DatabaseError("DatabaseManager not initialized. Call init() first.")
        return self.pool

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return status."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a query and return rows."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Execute a query and return a single row."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return a single value."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def ping(self) -> bool:
        """Check database connectivity."""
        try:
            pool = self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except DatabaseError:
            raise
        except Exception:
            return False

    async def ensure_schema(self, migrations_sql: list[tuple[int, str, str]]) -> None:
        """Run pending migrations. Each tuple is (version, name, sql)."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            # Create migrations table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Get applied versions
            rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
            applied = {r["version"] for r in rows}

            for version, name, sql in sorted(migrations_sql):
                if version not in applied:
                    logger.info("Applying migration %d: %s", version, name)
                    async with conn.transaction():
                        await conn.execute(sql)
                        await conn.execute(
                            "INSERT INTO schema_migrations (version) VALUES ($1)",
                            version,
                        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_connection.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/db/__init__.py src/db/connection.py tests/test_connection.py
git commit -m "feat: add DatabaseManager for asyncpg connection pool"
```

---

### Task 7: Schema Migrations

**Files:**
- Create: `src/db/migrations.py`
- Test: `tests/test_migrations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migrations.py
"""Tests for schema migrations."""
import pytest
from src.db.migrations import get_migrations


def test_get_migrations_returns_list():
    migrations = get_migrations(embedding_dimension=1536)
    assert isinstance(migrations, list)
    assert len(migrations) >= 1


def test_migrations_are_ordered():
    migrations = get_migrations(embedding_dimension=1536)
    versions = [m[0] for m in migrations]
    assert versions == sorted(versions)


def test_migrations_contain_episodes_table():
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    assert "CREATE TABLE" in initial_sql
    assert "episodes" in initial_sql
    assert "semantic_memories" in initial_sql
    assert "message_buffer" in initial_sql


def test_migrations_use_configured_dimension():
    migrations = get_migrations(embedding_dimension=768)
    initial_sql = migrations[0][2]
    assert "vector(768)" in initial_sql
    assert "vector(1536)" not in initial_sql


def test_migrations_use_hnsw_not_ivfflat():
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    assert "hnsw" in initial_sql.lower()
    assert "ivfflat" not in initial_sql.lower()


def test_migrations_use_coalesce_in_tsvector():
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    assert "coalesce" in initial_sql.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_migrations.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/db/migrations.py
"""Schema versioning and auto-migration."""
from __future__ import annotations


def get_migrations(embedding_dimension: int = 1536) -> list[tuple[int, str, str]]:
    """Return ordered list of (version, name, sql) migrations.

    The vector dimension is templated from config so that different
    embedding models work out of the box.
    """
    dim = embedding_dimension

    initial_schema = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- Episodes
    CREATE TABLE IF NOT EXISTS episodes (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id     VARCHAR(255) NOT NULL,
        title       TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector({dim}),
        tsv         tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content,''))
                    ) STORED,
        source_messages JSONB,
        metadata    JSONB DEFAULT '{{}}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON episodes USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(user_id, created_at DESC);

    -- Semantic Memories
    CREATE TABLE IF NOT EXISTS semantic_memories (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         VARCHAR(255) NOT NULL,
        content         TEXT NOT NULL,
        memory_type     VARCHAR(50) NOT NULL,
        embedding       vector({dim}),
        tsv             tsvector GENERATED ALWAYS AS (
                            to_tsvector('simple', coalesce(content,''))
                        ) STORED,
        source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
        confidence      FLOAT DEFAULT 1.0,
        metadata        JSONB DEFAULT '{{}}'::jsonb,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_semantic_user_id ON semantic_memories(user_id);
    CREATE INDEX IF NOT EXISTS idx_semantic_embedding ON semantic_memories USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_semantic_tsv ON semantic_memories USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memories(user_id, memory_type);

    -- Message Buffer
    CREATE TABLE IF NOT EXISTS message_buffer (
        id          BIGSERIAL PRIMARY KEY,
        user_id     VARCHAR(255) NOT NULL,
        role        VARCHAR(20) NOT NULL,
        content     TEXT NOT NULL,
        timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed   BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_buffer_user_unprocessed
        ON message_buffer(user_id) WHERE NOT processed;
    """

    return [
        (1, "initial schema", initial_schema),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_migrations.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/migrations.py tests/test_migrations.py
git commit -m "feat: add schema migrations with configurable vector dimension"
```


---

## Chunk 2: PostgreSQL Store Implementations

### Task 8: PgEpisodeStore

**Files:**
- Create: `src/db/episode_store.py`
- Test: `tests/test_episode_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_episode_store.py
"""Tests for PgEpisodeStore.

These tests use mocked asyncpg to test SQL logic without a real database.
Integration tests with a real PG instance belong in tests/integration/.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from src.db.episode_store import PgEpisodeStore
from src.domain.models import Episode


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="INSERT 1")
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def store(mock_db):
    return PgEpisodeStore(mock_db)


@pytest.mark.asyncio
async def test_save_calls_upsert(store, mock_db):
    ep = Episode(
        user_id="u1", title="Test", content="Content",
        source_messages=[], embedding=[0.1] * 1536,
    )
    await store.save(ep)
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "INSERT INTO episodes" in call_sql
    assert "ON CONFLICT" in call_sql


@pytest.mark.asyncio
async def test_get_returns_episode(store, mock_db):
    mock_db.fetchrow.return_value = {
        "id": "abc", "user_id": "u1", "title": "T",
        "content": "C", "source_messages": [],
        "metadata": {}, "embedding": None,
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    ep = await store.get("abc")
    assert ep is not None
    assert ep.id == "abc"


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store, mock_db):
    mock_db.fetchrow.return_value = None
    ep = await store.get("missing")
    assert ep is None


@pytest.mark.asyncio
async def test_delete_calls_execute(store, mock_db):
    await store.delete("abc")
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "DELETE FROM episodes" in call_sql


@pytest.mark.asyncio
async def test_search_by_text_uses_tsquery(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_by_text("u1", "hiking trip", top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "ts_rank" in call_sql or "tsv" in call_sql


@pytest.mark.asyncio
async def test_search_by_vector_uses_cosine(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_by_vector("u1", [0.1] * 1536, top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "<=>" in call_sql  # cosine distance operator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_episode_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/db/episode_store.py
"""PostgreSQL implementation of EpisodeStore."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.db.connection import DatabaseManager
from src.domain.models import Episode

logger = logging.getLogger("nemori")


class PgEpisodeStore:
    """Episode persistence + search backed by PostgreSQL + pgvector."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, episode: Episode) -> None:
        await self._db.execute(
            """
            INSERT INTO episodes (id, user_id, title, content, embedding,
                                  source_messages, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                source_messages = EXCLUDED.source_messages,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            episode.id,
            episode.user_id,
            episode.title,
            episode.content,
            episode.embedding,
            json.dumps(episode.source_messages),
            json.dumps(episode.metadata),
            episode.created_at,
            episode.updated_at,
        )

    async def get(self, episode_id: str) -> Episode | None:
        row = await self._db.fetchrow(
            "SELECT * FROM episodes WHERE id = $1", episode_id
        )
        return self._row_to_episode(row) if row else None

    async def list_by_user(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT * FROM episodes WHERE user_id = $1
               ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
            user_id, limit, offset,
        )
        return [self._row_to_episode(r) for r in rows]

    async def delete(self, episode_id: str) -> None:
        await self._db.execute("DELETE FROM episodes WHERE id = $1", episode_id)

    async def delete_by_user(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM episodes WHERE user_id = $1", user_id)

    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT *, embedding <=> $2::vector AS distance
               FROM episodes
               WHERE user_id = $1 AND embedding IS NOT NULL
               ORDER BY distance ASC
               LIMIT $3""",
            user_id, str(embedding), top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[Episode]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $2)) AS rank
               FROM episodes
               WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
               ORDER BY rank DESC
               LIMIT $3""",
            user_id, query, top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[Episode]:
        """Reciprocal Rank Fusion of vector + text search."""
        rows = await self._db.fetch(
            """WITH vector_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $3::vector) AS vrank
                FROM episodes
                WHERE user_id = $1 AND embedding IS NOT NULL
                LIMIT $4 * 2
            ),
            text_results AS (
                SELECT id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank(tsv, plainto_tsquery('simple', $2)) DESC
                ) AS trank
                FROM episodes
                WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
                LIMIT $4 * 2
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(1.0 / (60 + v.vrank), 0) +
                       COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
                FROM vector_results v
                FULL OUTER JOIN text_results t ON v.id = t.id
                ORDER BY rrf_score DESC
                LIMIT $4
            )
            SELECT e.* FROM fused f JOIN episodes e ON f.id = e.id
            ORDER BY f.rrf_score DESC""",
            user_id, query, str(embedding), top_k,
        )
        return [self._row_to_episode(r) for r in rows]

    @staticmethod
    def _row_to_episode(row: Any) -> Episode:
        source_msgs = row["source_messages"]
        if isinstance(source_msgs, str):
            source_msgs = json.loads(source_msgs)
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Episode(
            id=str(row["id"]),
            user_id=row["user_id"],
            title=row["title"],
            content=row["content"],
            embedding=list(row["embedding"]) if row.get("embedding") else None,
            source_messages=source_msgs or [],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_episode_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/episode_store.py tests/test_episode_store.py
git commit -m "feat: add PgEpisodeStore with pgvector + tsvector search"
```

---

### Task 9: PgSemanticStore

**Files:**
- Create: `src/db/semantic_store.py`
- Test: `tests/test_semantic_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_store.py
"""Tests for PgSemanticStore."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from src.db.semantic_store import PgSemanticStore
from src.domain.models import SemanticMemory


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.execute = AsyncMock(return_value="INSERT 1")
    return db


@pytest.fixture
def store(mock_db):
    return PgSemanticStore(mock_db)


@pytest.mark.asyncio
async def test_save_calls_upsert(store, mock_db):
    sm = SemanticMemory(
        user_id="u1", content="User likes hiking",
        memory_type="preference", embedding=[0.1] * 1536,
    )
    await store.save(sm)
    mock_db.execute.assert_called_once()
    call_sql = mock_db.execute.call_args[0][0]
    assert "INSERT INTO semantic_memories" in call_sql
    assert "ON CONFLICT" in call_sql


@pytest.mark.asyncio
async def test_save_batch(store, mock_db):
    memories = [
        SemanticMemory(user_id="u1", content=f"fact {i}", memory_type="identity")
        for i in range(3)
    ]
    await store.save_batch(memories)
    assert mock_db.execute.call_count == 3


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store, mock_db):
    result = await store.get("missing-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_by_user_with_type_filter(store, mock_db):
    await store.list_by_user("u1", memory_type="preference")
    call_sql = mock_db.fetch.call_args[0][0]
    assert "memory_type" in call_sql


@pytest.mark.asyncio
async def test_search_hybrid(store, mock_db):
    mock_db.fetch.return_value = []
    await store.search_hybrid("u1", "hiking", [0.1] * 1536, top_k=5)
    call_sql = mock_db.fetch.call_args[0][0]
    assert "rrf_score" in call_sql or "<=>" in call_sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_semantic_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write implementation**

```python
# src/db/semantic_store.py
"""PostgreSQL implementation of SemanticStore."""
from __future__ import annotations

import json
import logging
from typing import Any

from src.db.connection import DatabaseManager
from src.domain.models import SemanticMemory

logger = logging.getLogger("nemori")


class PgSemanticStore:
    """Semantic memory persistence + search backed by PostgreSQL + pgvector."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def save(self, memory: SemanticMemory) -> None:
        await self._db.execute(
            """
            INSERT INTO semantic_memories
                (id, user_id, content, memory_type, embedding,
                 source_episode_id, confidence, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                memory_type = EXCLUDED.memory_type,
                embedding = EXCLUDED.embedding,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            memory.id,
            memory.user_id,
            memory.content,
            memory.memory_type,
            memory.embedding,
            memory.source_episode_id,
            memory.confidence,
            json.dumps(memory.metadata),
            memory.created_at,
            memory.updated_at,
        )

    async def save_batch(self, memories: list[SemanticMemory]) -> None:
        for memory in memories:
            await self.save(memory)

    async def get(self, memory_id: str) -> SemanticMemory | None:
        row = await self._db.fetchrow(
            "SELECT * FROM semantic_memories WHERE id = $1", memory_id
        )
        return self._row_to_memory(row) if row else None

    async def list_by_user(
        self, user_id: str, memory_type: str | None = None
    ) -> list[SemanticMemory]:
        if memory_type:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 AND memory_type = $2
                   ORDER BY created_at DESC""",
                user_id, memory_type,
            )
        else:
            rows = await self._db.fetch(
                """SELECT * FROM semantic_memories
                   WHERE user_id = $1 ORDER BY created_at DESC""",
                user_id,
            )
        return [self._row_to_memory(r) for r in rows]

    async def delete(self, memory_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE id = $1", memory_id
        )

    async def delete_by_user(self, user_id: str) -> None:
        await self._db.execute(
            "DELETE FROM semantic_memories WHERE user_id = $1", user_id
        )

    async def search_by_vector(
        self, user_id: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """SELECT *, embedding <=> $2::vector AS distance
               FROM semantic_memories
               WHERE user_id = $1 AND embedding IS NOT NULL
               ORDER BY distance ASC
               LIMIT $3""",
            user_id, str(embedding), top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    async def search_by_text(
        self, user_id: str, query: str, top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """SELECT *, ts_rank(tsv, plainto_tsquery('simple', $2)) AS rank
               FROM semantic_memories
               WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
               ORDER BY rank DESC
               LIMIT $3""",
            user_id, query, top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    async def search_hybrid(
        self, user_id: str, query: str, embedding: list[float], top_k: int
    ) -> list[SemanticMemory]:
        rows = await self._db.fetch(
            """WITH vector_results AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $3::vector) AS vrank
                FROM semantic_memories
                WHERE user_id = $1 AND embedding IS NOT NULL
                LIMIT $4 * 2
            ),
            text_results AS (
                SELECT id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank(tsv, plainto_tsquery('simple', $2)) DESC
                ) AS trank
                FROM semantic_memories
                WHERE user_id = $1 AND tsv @@ plainto_tsquery('simple', $2)
                LIMIT $4 * 2
            ),
            fused AS (
                SELECT COALESCE(v.id, t.id) AS id,
                       COALESCE(1.0 / (60 + v.vrank), 0) +
                       COALESCE(1.0 / (60 + t.trank), 0) AS rrf_score
                FROM vector_results v
                FULL OUTER JOIN text_results t ON v.id = t.id
                ORDER BY rrf_score DESC
                LIMIT $4
            )
            SELECT sm.* FROM fused f
            JOIN semantic_memories sm ON f.id = sm.id
            ORDER BY f.rrf_score DESC""",
            user_id, query, str(embedding), top_k,
        )
        return [self._row_to_memory(r) for r in rows]

    @staticmethod
    def _row_to_memory(row: Any) -> SemanticMemory:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return SemanticMemory(
            id=str(row["id"]),
            user_id=row["user_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            embedding=list(row["embedding"]) if row.get("embedding") else None,
            source_episode_id=str(row["source_episode_id"]) if row.get("source_episode_id") else None,
            confidence=row["confidence"],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_semantic_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/semantic_store.py tests/test_semantic_store.py
git commit -m "feat: add PgSemanticStore with pgvector + tsvector search"
```

---

### Task 10: PgMessageBufferStore

**Files:**
- Create: `src/db/buffer_store.py`
- Test: `tests/test_buffer_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_buffer_store.py
"""Tests for PgMessageBufferStore."""
import pytest
from unittest.mock import AsyncMock
from src.db.buffer_store import PgMessageBufferStore
from src.domain.models import Message


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.fetchval = AsyncMock(return_value=0)
    return db


@pytest.fixture
def store(mock_db):
    return PgMessageBufferStore(mock_db)


@pytest.mark.asyncio
async def test_push_inserts_messages(store, mock_db):
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    await store.push("u1", messages)
    assert mock_db.execute.call_count == 2


@pytest.mark.asyncio
async def test_count_unprocessed(store, mock_db):
    mock_db.fetchval.return_value = 5
    count = await store.count_unprocessed("u1")
    assert count == 5
    call_sql = mock_db.fetchval.call_args[0][0]
    assert "NOT processed" in call_sql or "processed = false" in call_sql.lower()


@pytest.mark.asyncio
async def test_mark_processed_and_delete(store, mock_db):
    await store.mark_processed("u1", [1, 2, 3])
    call_sql = mock_db.execute.call_args[0][0]
    assert "DELETE" in call_sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_buffer_store.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/db/buffer_store.py
"""PostgreSQL implementation of MessageBufferStore."""
from __future__ import annotations

import logging

from src.db.connection import DatabaseManager
from src.domain.models import Message

logger = logging.getLogger("nemori")


class PgMessageBufferStore:
    """Persistent message buffer backed by PostgreSQL."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def push(self, user_id: str, messages: list[Message]) -> None:
        for msg in messages:
            await self._db.execute(
                """INSERT INTO message_buffer (user_id, role, content, timestamp)
                   VALUES ($1, $2, $3, $4)""",
                user_id, msg.role, msg.content, msg.timestamp,
            )

    async def get_unprocessed(self, user_id: str) -> list[Message]:
        rows = await self._db.fetch(
            """SELECT id, role, content, timestamp
               FROM message_buffer
               WHERE user_id = $1 AND NOT processed
               ORDER BY timestamp ASC""",
            user_id,
        )
        result = []
        for row in rows:
            msg = Message(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
                metadata={"buffer_id": row["id"]},
            )
            result.append(msg)
        return result

    async def mark_processed(self, user_id: str, message_ids: list[int]) -> None:
        """Delete processed messages to prevent unbounded growth."""
        if not message_ids:
            return
        await self._db.execute(
            """DELETE FROM message_buffer
               WHERE user_id = $1 AND id = ANY($2)""",
            user_id, message_ids,
        )

    async def count_unprocessed(self, user_id: str) -> int:
        count = await self._db.fetchval(
            """SELECT COUNT(*) FROM message_buffer
               WHERE user_id = $1 AND NOT processed""",
            user_id,
        )
        return count or 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_buffer_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/buffer_store.py tests/test_buffer_store.py
git commit -m "feat: add PgMessageBufferStore with cleanup on mark_processed"
```


---

## Chunk 3: LLM Orchestration Layer

### Task 11: LLM Orchestrator

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Create package init**

```python
# src/llm/__init__.py
# src/llm/generators/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_orchestrator.py
"""Tests for LLMOrchestrator."""
import pytest
import asyncio
from unittest.mock import AsyncMock
from types import MappingProxyType
from src.llm.orchestrator import LLMOrchestrator, LLMRequest, LLMResponse, TokenUsage
from src.domain.exceptions import LLMError, LLMRateLimitError, TokenBudgetExceeded


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value="response text")
    return provider


@pytest.fixture
def orchestrator(mock_provider):
    return LLMOrchestrator(
        provider=mock_provider,
        default_model="gpt-4o-mini",
        max_concurrent=5,
    )


@pytest.mark.asyncio
async def test_execute_simple_request(orchestrator, mock_provider):
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    response = await orchestrator.execute(request)
    assert isinstance(response, LLMResponse)
    assert response.content == "response text"
    mock_provider.complete.assert_called_once()


@pytest.mark.asyncio
async def test_execute_uses_default_model(orchestrator, mock_provider):
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    await orchestrator.execute(request)
    call_kwargs = mock_provider.complete.call_args
    assert call_kwargs[1].get("model") == "gpt-4o-mini" or "gpt-4o-mini" in str(call_kwargs)


@pytest.mark.asyncio
async def test_execute_retries_on_server_error(mock_provider):
    call_count = 0
    async def flaky_complete(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise LLMError("500 server error")
        return "success"

    mock_provider.complete = flaky_complete
    orch = LLMOrchestrator(provider=mock_provider, default_model="gpt-4o-mini")
    request = LLMRequest(messages=({"role": "user", "content": "hi"},), retries=3)
    response = await orch.execute(request)
    assert response.content == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_execute_respects_concurrency_limit():
    """Only max_concurrent requests should run simultaneously."""
    active = 0
    max_active = 0

    async def slow_complete(messages, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "done"

    provider = AsyncMock()
    provider.complete = slow_complete
    orch = LLMOrchestrator(provider=provider, default_model="m", max_concurrent=2)

    requests = [LLMRequest(messages=({"role": "user", "content": f"{i}"},)) for i in range(5)]
    await orch.execute_batch(requests)
    assert max_active <= 2


@pytest.mark.asyncio
async def test_token_budget_exceeded(mock_provider):
    orch = LLMOrchestrator(
        provider=mock_provider, default_model="m", token_budget=100
    )
    # Simulate usage that exceeds budget
    orch._total_tokens = 95
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    with pytest.raises(TokenBudgetExceeded):
        await orch.execute(request)


@pytest.mark.asyncio
async def test_stats_tracking(orchestrator, mock_provider):
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    await orchestrator.execute(request)
    stats = orchestrator.stats
    assert stats.total_requests >= 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 4: Write implementation**

```python
# src/llm/orchestrator.py
"""Unified LLM call orchestration with retry, concurrency, and budget."""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from src.domain.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    TokenBudgetExceeded,
)
from src.domain.interfaces import LLMProvider

logger = logging.getLogger("nemori")


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[dict, ...]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    response_format: type | None = None
    timeout: float = 30.0
    retries: int = 3
    metadata: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    request_id: str


@dataclass
class OrchestratorStats:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0


# HTTP status codes
_RETRYABLE = {429, 500, 502, 503, 504}
_NON_RETRYABLE = {400, 401, 403, 404}


class LLMOrchestrator:
    """Unified LLM call orchestration."""

    def __init__(
        self,
        provider: LLMProvider,
        default_model: str,
        max_concurrent: int = 10,
        token_budget: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._token_budget = token_budget
        self._log = logger or logging.getLogger("nemori")
        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0

    async def execute(self, request: LLMRequest) -> LLMResponse:
        """Execute a single LLM request with retry and concurrency control."""
        # Budget check
        if self._token_budget and self._total_tokens >= self._token_budget:
            raise TokenBudgetExceeded(
                "Token budget exceeded",
                used=self._total_tokens,
                budget=self._token_budget,
            )

        model = request.model or self._default_model
        request_id = str(uuid.uuid4())[:8]
        last_error: Exception | None = None

        for attempt in range(request.retries):
            try:
                async with self._semaphore:
                    start = time.monotonic()
                    content = await asyncio.wait_for(
                        self._provider.complete(
                            list(request.messages),
                            model=model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                        ),
                        timeout=request.timeout,
                    )
                    latency = (time.monotonic() - start) * 1000

                self._total_requests += 1
                self._total_latency_ms += latency

                response = LLMResponse(
                    content=content,
                    model=model,
                    usage=TokenUsage(),
                    latency_ms=latency,
                    request_id=request_id,
                )
                self._log.debug(
                    "LLM request %s completed in %.0fms",
                    request_id, latency,
                )
                return response

            except (LLMAuthError, TokenBudgetExceeded):
                raise
            except Exception as e:
                last_error = e
                self._total_errors += 1
                if attempt < request.retries - 1:
                    delay = min(1.0 * (2 ** attempt) + random.uniform(0, 0.5), 30.0)
                    self._log.warning(
                        "LLM request %s attempt %d failed: %s. Retrying in %.1fs",
                        request_id, attempt + 1, e, delay,
                    )
                    await asyncio.sleep(delay)

        raise LLMError(f"All {request.retries} attempts failed: {last_error}") from last_error

    async def execute_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Execute multiple requests concurrently, respecting semaphore."""
        tasks = [self.execute(req) for req in requests]
        return await asyncio.gather(*tasks)

    @property
    def stats(self) -> OrchestratorStats:
        avg = (
            self._total_latency_ms / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return OrchestratorStats(
            total_requests=self._total_requests,
            total_tokens=self._total_tokens,
            total_errors=self._total_errors,
            avg_latency_ms=avg,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_orchestrator.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/__init__.py src/llm/orchestrator.py src/llm/generators/__init__.py tests/test_orchestrator.py
git commit -m "feat: add LLMOrchestrator with retry, concurrency, and token budget"
```

---

### Task 12: Async LLM Client

**Files:**
- Create: `src/llm/client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_client.py
"""Tests for AsyncLLMClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llm.client import AsyncLLMClient


@pytest.mark.asyncio
async def test_complete_returns_string():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete(
            [{"role": "user", "content": "hi"}], model="gpt-4o-mini"
        )
        assert result == "Hello!"


@pytest.mark.asyncio
async def test_complete_passes_model_and_params():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client.complete(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o", temperature=0.5, max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_llm_client.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/llm/client.py
"""Async OpenAI-compatible LLM client implementing LLMProvider."""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from src.domain.exceptions import LLMError, LLMAuthError, LLMRateLimitError

logger = logging.getLogger("nemori")


class AsyncLLMClient:
    """Async LLM client wrapping the OpenAI API."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """Call the chat completions endpoint."""
        model = kwargs.pop("model", "gpt-4o-mini")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 2000)

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "403" in error_str:
                raise LLMAuthError(f"Authentication failed: {e}") from e
            if "429" in error_str:
                raise LLMRateLimitError(f"Rate limited: {e}") from e
            raise LLMError(f"LLM call failed: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_llm_client.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_client.py
git commit -m "feat: add AsyncLLMClient implementing LLMProvider protocol"
```

---

### Task 13: Async Embedding Client

**Files:**
- Create: `src/services/embedding.py`
- Test: `tests/test_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
"""Tests for AsyncEmbeddingClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.embedding import AsyncEmbeddingClient


@pytest.mark.asyncio
async def test_embed_returns_float_list():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed("hello")
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_lists():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
```

- [ ] **Step 2: Run test to verify it fails, then write implementation**

```python
# src/services/embedding.py
"""Async embedding client implementing EmbeddingProvider."""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from src.domain.exceptions import EmbeddingError

logger = logging.getLogger("nemori")


class AsyncEmbeddingClient:
    """Async embedding generation via OpenAI-compatible API."""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}") from e
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_embedding.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/services/embedding.py tests/test_embedding.py
git commit -m "feat: add AsyncEmbeddingClient implementing EmbeddingProvider"
```

---

### Task 14: Prompt Templates + Generators

**Files:**
- Create: `src/llm/prompts.py` (migrated from src/generation/prompts.py, cleaned up)
- Create: `src/llm/generators/episode.py`
- Create: `src/llm/generators/semantic.py`
- Create: `src/llm/generators/segmenter.py`
- Test: `tests/test_generators.py`

- [ ] **Step 1: Migrate prompts.py**

Copy `src/generation/prompts.py` to `src/llm/prompts.py`. Remove merge-related prompts (MERGE_DECISION_PROMPT, MERGE_CONTENT_PROMPT) and their helper methods. Keep: EPISODE_GENERATION_PROMPT, PREDICTION_PROMPT, EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT, SEMANTIC_GENERATION_PROMPT, BATCH_SEGMENTATION_PROMPT, and their format helper methods.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_generators.py
"""Tests for LLM generators."""
import pytest
from unittest.mock import AsyncMock
from src.llm.orchestrator import LLMOrchestrator, LLMResponse, TokenUsage
from src.llm.generators.episode import EpisodeGenerator
from src.llm.generators.semantic import SemanticGenerator
from src.llm.generators.segmenter import BatchSegmenter
from src.domain.models import Message, Episode


@pytest.fixture
def mock_orchestrator():
    orch = AsyncMock(spec=LLMOrchestrator)
    return orch


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[0.1] * 1536)
    return emb


@pytest.mark.asyncio
async def test_episode_generator_returns_episode(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"title": "Test Episode", "content": "User discussed hiking.", "timestamp": "2024-01-01T10:00:00"}',
        model="gpt-4o-mini",
        usage=TokenUsage(),
        latency_ms=100,
        request_id="abc",
    ))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [
        Message(role="user", content="I love hiking"),
        Message(role="assistant", content="That's great!"),
    ]
    episode = await gen.generate("u1", messages, "topic_change")
    assert isinstance(episode, Episode)
    assert episode.title == "Test Episode"
    assert episode.user_id == "u1"
    assert episode.embedding is not None


@pytest.mark.asyncio
async def test_episode_generator_fallback_on_bad_json(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content="not valid json",
        model="gpt-4o-mini",
        usage=TokenUsage(),
        latency_ms=100,
        request_id="abc",
    ))
    gen = EpisodeGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    messages = [Message(role="user", content="test")]
    episode = await gen.generate("u1", messages, "fallback_test")
    # Should create a fallback episode with raw content
    assert isinstance(episode, Episode)
    assert episode.user_id == "u1"


@pytest.mark.asyncio
async def test_semantic_generator_returns_memories(mock_orchestrator, mock_embedding):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"statements": ["User likes hiking", "User works at Google"]}',
        model="gpt-4o-mini",
        usage=TokenUsage(),
        latency_ms=100,
        request_id="abc",
    ))
    gen = SemanticGenerator(orchestrator=mock_orchestrator, embedding=mock_embedding)
    episode = Episode(user_id="u1", title="T", content="C", source_messages=[])
    memories = await gen.generate("u1", episode, [], [])
    assert len(memories) == 2
    assert memories[0].content == "User likes hiking"


@pytest.mark.asyncio
async def test_segmenter_returns_groups(mock_orchestrator):
    mock_orchestrator.execute = AsyncMock(return_value=LLMResponse(
        content='{"episodes": [{"indices": [1, 2], "topic": "hiking"}, {"indices": [3, 4], "topic": "work"}]}',
        model="gpt-4o-mini",
        usage=TokenUsage(),
        latency_ms=100,
        request_id="abc",
    ))
    seg = BatchSegmenter(orchestrator=mock_orchestrator)
    messages = [
        Message(role="user", content=f"msg {i}") for i in range(4)
    ]
    groups = await seg.segment(messages)
    assert len(groups) == 2
    assert len(groups[0]["messages"]) == 2
```

- [ ] **Step 3: Write implementations**

Create `src/llm/generators/episode.py`, `src/llm/generators/semantic.py`, `src/llm/generators/segmenter.py`. Each generator:
1. Receives `LLMOrchestrator` and optionally `EmbeddingProvider`
2. Constructs an `LLMRequest` with the appropriate prompt
3. Parses the JSON response
4. Returns domain objects
5. Has fallback logic for malformed LLM responses

Key implementation details:
- `EpisodeGenerator.generate()`: builds prompt from messages, parses JSON {title, content, timestamp}, generates embedding, returns Episode
- `SemanticGenerator.generate()`: supports prediction-correction mode (two LLM calls) or direct extraction (one call), returns list[SemanticMemory]
- `BatchSegmenter.segment()`: sends all messages, parses JSON {episodes: [{indices, topic}]}, returns grouped message lists

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/test_generators.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/prompts.py src/llm/generators/ tests/test_generators.py
git commit -m "feat: add async LLM generators (episode, semantic, segmenter)"
```


---

## Chunk 4: Core System, Facade, Search, and Cleanup

### Task 15: Async EventBus

**Files:**
- Rewrite: `src/services/event_bus.py`
- Test: `tests/test_event_bus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_bus.py
"""Tests for async EventBus."""
import pytest
import asyncio
from src.services.event_bus import EventBus


@pytest.mark.asyncio
async def test_emit_calls_handler():
    bus = EventBus()
    called_with = {}

    async def handler(**kwargs):
        called_with.update(kwargs)

    bus.on("test_event", handler)
    await bus.emit("test_event", data="hello")
    await asyncio.sleep(0.05)  # let tasks complete
    assert called_with.get("data") == "hello"


@pytest.mark.asyncio
async def test_multiple_handlers():
    bus = EventBus()
    results = []

    async def h1(**kw): results.append("h1")
    async def h2(**kw): results.append("h2")

    bus.on("ev", h1)
    bus.on("ev", h2)
    await bus.emit("ev")
    await asyncio.sleep(0.05)
    assert "h1" in results and "h2" in results


@pytest.mark.asyncio
async def test_no_handler_no_error():
    bus = EventBus()
    await bus.emit("unknown_event")  # should not raise
```

- [ ] **Step 2: Write implementation**

```python
# src/services/event_bus.py
"""Async event bus for decoupling pipeline stages."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("nemori")


class EventBus:
    """Simple async publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    async def emit(self, event: str, **kwargs: Any) -> None:
        for handler in self._handlers.get(event, []):
            asyncio.create_task(handler(**kwargs))
```

- [ ] **Step 3: Run tests, commit**

Run: `python -m pytest tests/test_event_bus.py -v`

```bash
git add src/services/event_bus.py tests/test_event_bus.py
git commit -m "refactor: rewrite EventBus as async"
```

---

### Task 16: Unified Search

**Files:**
- Create: `src/search/unified.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
"""Tests for UnifiedSearch."""
import pytest
from unittest.mock import AsyncMock
from src.search.unified import UnifiedSearch, SearchMethod, SearchResult
from src.domain.models import Episode, SemanticMemory


@pytest.fixture
def mock_episode_store():
    store = AsyncMock()
    store.search_by_vector = AsyncMock(return_value=[])
    store.search_by_text = AsyncMock(return_value=[])
    store.search_hybrid = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_semantic_store():
    store = AsyncMock()
    store.search_by_vector = AsyncMock(return_value=[])
    store.search_by_text = AsyncMock(return_value=[])
    store.search_hybrid = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_embedding():
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[0.1] * 1536)
    return emb


@pytest.fixture
def search(mock_episode_store, mock_semantic_store, mock_embedding):
    return UnifiedSearch(mock_episode_store, mock_semantic_store, mock_embedding)


@pytest.mark.asyncio
async def test_hybrid_search(search, mock_episode_store, mock_semantic_store):
    result = await search.search("u1", "hiking", method=SearchMethod.HYBRID)
    assert isinstance(result, SearchResult)
    mock_episode_store.search_hybrid.assert_called_once()
    mock_semantic_store.search_hybrid.assert_called_once()


@pytest.mark.asyncio
async def test_vector_search(search, mock_episode_store, mock_semantic_store):
    result = await search.search("u1", "hiking", method=SearchMethod.VECTOR)
    mock_episode_store.search_by_vector.assert_called_once()


@pytest.mark.asyncio
async def test_text_search(search, mock_episode_store, mock_semantic_store):
    result = await search.search("u1", "hiking", method=SearchMethod.TEXT)
    mock_episode_store.search_by_text.assert_called_once()


@pytest.mark.asyncio
async def test_search_result_to_dict(search):
    result = await search.search("u1", "test")
    d = result.to_dict()
    assert "episodes" in d
    assert "semantic_memories" in d
```

- [ ] **Step 2: Write implementation**

```python
# src/search/unified.py
"""Unified search across episode and semantic stores."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.domain.models import Episode, SemanticMemory
from src.domain.interfaces import EpisodeStore, SemanticStore, EmbeddingProvider

logger = logging.getLogger("nemori")


class SearchMethod(Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    episodes: list[Episode] = field(default_factory=list)
    semantic_memories: list[SemanticMemory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "semantic_memories": [s.to_dict() for s in self.semantic_memories],
        }


class UnifiedSearch:
    """Delegates search to the appropriate store methods."""

    def __init__(
        self,
        episode_store: EpisodeStore,
        semantic_store: SemanticStore,
        embedding: EmbeddingProvider,
    ) -> None:
        self._episodes = episode_store
        self._semantics = semantic_store
        self._embedding = embedding

    async def search(
        self,
        user_id: str,
        query: str,
        top_k_episodes: int = 10,
        top_k_semantic: int = 10,
        method: SearchMethod = SearchMethod.HYBRID,
    ) -> SearchResult:
        embedding = await self._embedding.embed(query)

        if method == SearchMethod.VECTOR:
            ep_task = self._episodes.search_by_vector(user_id, embedding, top_k_episodes)
            sm_task = self._semantics.search_by_vector(user_id, embedding, top_k_semantic)
        elif method == SearchMethod.TEXT:
            ep_task = self._episodes.search_by_text(user_id, query, top_k_episodes)
            sm_task = self._semantics.search_by_text(user_id, query, top_k_semantic)
        else:  # HYBRID
            ep_task = self._episodes.search_hybrid(user_id, query, embedding, top_k_episodes)
            sm_task = self._semantics.search_hybrid(user_id, query, embedding, top_k_semantic)

        episodes, semantics = await asyncio.gather(ep_task, sm_task)
        return SearchResult(episodes=episodes, semantic_memories=semantics)
```

- [ ] **Step 3: Run tests, commit**

Run: `python -m pytest tests/test_search.py -v`

```bash
git add src/search/unified.py tests/test_search.py
git commit -m "feat: add UnifiedSearch delegating to store search methods"
```

---

### Task 17: Async MemorySystem

**Files:**
- Rewrite: `src/core/memory_system.py`
- Test: `tests/test_memory_system.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_system.py
"""Tests for async MemorySystem."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.core.memory_system import MemorySystem
from src.domain.models import Message, Episode, SemanticMemory
from src.search.unified import SearchResult
from src.config import MemoryConfig


@pytest.fixture
def deps():
    """Create mock dependencies."""
    return {
        "config": MemoryConfig(),
        "db": AsyncMock(),
        "episode_store": AsyncMock(),
        "semantic_store": AsyncMock(),
        "buffer_store": AsyncMock(),
        "orchestrator": AsyncMock(),
        "embedding": AsyncMock(),
        "episode_generator": AsyncMock(),
        "semantic_generator": AsyncMock(),
        "event_bus": AsyncMock(),
        "search": AsyncMock(),
    }


@pytest.fixture
def system(deps):
    deps["buffer_store"].count_unprocessed = AsyncMock(return_value=0)
    deps["buffer_store"].get_unprocessed = AsyncMock(return_value=[])
    deps["search"].search = AsyncMock(return_value=SearchResult())
    return MemorySystem(**deps)


@pytest.mark.asyncio
async def test_add_messages_pushes_to_buffer(system, deps):
    msgs = [Message(role="user", content="hi")]
    await system.add_messages("u1", msgs)
    deps["buffer_store"].push.assert_called_once_with("u1", msgs)


@pytest.mark.asyncio
async def test_flush_processes_buffer(system, deps):
    deps["buffer_store"].get_unprocessed.return_value = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    ep = Episode(user_id="u1", title="T", content="C", source_messages=[])
    deps["episode_generator"].generate = AsyncMock(return_value=ep)
    deps["semantic_generator"].generate = AsyncMock(return_value=[])

    result = await system.flush("u1")
    assert len(result) >= 1
    deps["episode_store"].save.assert_called()


@pytest.mark.asyncio
async def test_search_delegates_to_unified_search(system, deps):
    await system.search("u1", "hiking")
    deps["search"].search.assert_called_once()


@pytest.mark.asyncio
async def test_delete_episode(system, deps):
    await system.delete_episode("u1", "ep-1")
    deps["episode_store"].delete.assert_called_once_with("ep-1")


@pytest.mark.asyncio
async def test_delete_user(system, deps):
    await system.delete_user("u1")
    deps["episode_store"].delete_by_user.assert_called_once_with("u1")
    deps["semantic_store"].delete_by_user.assert_called_once_with("u1")


@pytest.mark.asyncio
async def test_drain_waits_for_tasks(system):
    await system.drain(timeout=1.0)  # Should not raise
```

- [ ] **Step 2: Write implementation**

Rewrite `src/core/memory_system.py` as async. Key changes:
- All methods are `async def`
- User locks: `dict[str, asyncio.Lock]` with LRU eviction
- Background semantic generation via `asyncio.create_task`
- `drain()` method to wait for background tasks
- Receives all dependencies via constructor (no factory)

- [ ] **Step 3: Run tests, commit**

```bash
git add src/core/memory_system.py tests/test_memory_system.py
git commit -m "refactor: rewrite MemorySystem as async with asyncio.Lock"
```

---

### Task 18: Async NemoriMemory Facade

**Files:**
- Rewrite: `src/api/facade.py`
- Test: `tests/test_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_facade.py
"""Tests for async NemoriMemory facade."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.api.facade import NemoriMemory
from src.config import MemoryConfig


@pytest.mark.asyncio
async def test_facade_context_manager():
    """NemoriMemory should work as async context manager."""
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db_instance = AsyncMock()
        MockDB.return_value = mock_db_instance

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            assert memory is not None
        mock_db_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_facade_add_messages():
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        MockDB.return_value = AsyncMock()

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            await memory.add_messages("u1", [{"role": "user", "content": "hi"}])
            memory._system.add_messages.assert_called_once()


@pytest.mark.asyncio
async def test_facade_health():
    with patch("src.api.facade.DatabaseManager") as MockDB, \
         patch("src.api.facade.NemoriMemory._build_system", new_callable=AsyncMock):
        mock_db = AsyncMock()
        mock_db.ping = AsyncMock(return_value=True)
        mock_db.pool = MagicMock()
        mock_db.pool.get_size.return_value = 10
        mock_db.pool.get_idle_size.return_value = 8
        MockDB.return_value = mock_db

        config = MemoryConfig(dsn="postgresql://localhost/test", llm_api_key="test")
        async with NemoriMemory(config=config) as memory:
            memory._system = AsyncMock()
            health = await memory.health()
            assert health.db is True
```

- [ ] **Step 2: Write implementation**

Rewrite `src/api/facade.py`:
- `__init__(self, config: MemoryConfig | None = None)`
- `async __aenter__`: init DB, ensure_schema, build_system
- `async __aexit__`: drain + close
- All public methods delegate to `self._system`
- `_build_system()`: wires all dependencies together
- `health()`: returns `HealthResult`

- [ ] **Step 3: Run tests, commit**

```bash
git add src/api/facade.py tests/test_facade.py
git commit -m "refactor: rewrite NemoriMemory facade as async context manager"
```

---

### Task 19: Utils — Token Counter & Text Utilities

**Files:**
- Rewrite: `src/utils/token_counter.py`
- Create: `src/utils/text.py` (renamed from text_utils.py)

- [ ] **Step 1: Rewrite token_counter.py**

Simplify to a stateless utility (remove singleton pattern, remove category tracking). Keep only the `estimate_tokens(text)` function using tiktoken:

```python
# src/utils/token_counter.py
"""Token estimation utilities."""
from __future__ import annotations

import tiktoken


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count for a given text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: ~4 chars per token for English
        return len(text) // 4
```

- [ ] **Step 2: Create src/utils/text.py**

Rename and simplify from text_utils.py:

```python
# src/utils/text.py
"""Text utilities."""
from __future__ import annotations


def estimate_token_count(text: str) -> int:
    """Quick token estimate based on character heuristics."""
    if not text:
        return 0
    # Detect if text is primarily CJK
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cjk_count > len(text) * 0.3:
        return int(len(text) / 1.5)
    return len(text) // 4
```

- [ ] **Step 3: Delete old text_utils.py**

```bash
rm -f src/utils/text_utils.py
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/token_counter.py src/utils/text.py
git commit -m "refactor: simplify token_counter and rename text_utils to text"
```

---

### Task 20: Update Package Exports (was 19)

**Files:**
- Modify: `nemori/__init__.py`
- Modify: `src/utils/__init__.py`

- [ ] **Step 1: Update nemori/__init__.py**

```python
# nemori/__init__.py
"""Nemori — Self-organizing agent memory system."""
from src.api.facade import NemoriMemory
from src.config import MemoryConfig
from src.domain.models import Message, Episode, SemanticMemory, HealthResult
from src.domain.exceptions import NemoriError, DatabaseError, LLMError
from src.search.unified import SearchMethod, SearchResult

__all__ = [
    "NemoriMemory",
    "MemoryConfig",
    "Message",
    "Episode",
    "SemanticMemory",
    "HealthResult",
    "NemoriError",
    "DatabaseError",
    "LLMError",
    "SearchMethod",
    "SearchResult",
]
```

- [ ] **Step 2: Commit**

```bash
git add nemori/__init__.py src/utils/__init__.py
git commit -m "refactor: update package exports for new architecture"
```

---

### Task 21: Delete Old Code

**Files:** All files listed in spec Section 9 "Removal List"

- [ ] **Step 1: Delete old storage layer**

```bash
rm -rf src/storage/
rm -rf src/infrastructure/
rm -rf src/models/
```

- [ ] **Step 2: Delete old search implementations**

```bash
rm -f src/search/chroma_search.py
rm -f src/search/bm25_search.py
rm -f src/search/episode_original_message_search.py
rm -f src/search/original_message_search.py
rm -f src/search/unified_search.py
```

- [ ] **Step 3: Delete old services**

```bash
rm -f src/services/cache.py
rm -f src/services/providers.py
rm -f src/services/task_manager.py
rm -f src/services/metrics.py
```

- [ ] **Step 4: Delete old utils**

```bash
rm -f src/utils/performance.py
rm -f src/utils/llm_client.py
rm -f src/utils/embedding_client.py
```

- [ ] **Step 5: Delete old generation module**

```bash
rm -rf src/generation/
```

- [ ] **Step 6: Note on message_buffer.py**

`src/core/message_buffer.py` is no longer needed — MemorySystem uses `MessageBufferStore` directly. Delete it:

```bash
rm -f src/core/message_buffer.py
```

This is a deliberate simplification from the spec: the thin wrapper added no value since MemorySystem interacts with the protocol directly.

- [ ] **Step 7: Run all tests to ensure nothing is broken**

Run: `python -m pytest tests/ -v`
Expected: All new tests PASS, no import errors

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove old JSONL/ChromaDB/BM25/sync code"
```

---

### Task 22: Update Examples

**Files:**
- Modify: `examples/quickstart.py`

- [ ] **Step 1: Rewrite quickstart.py**

```python
# examples/quickstart.py
"""Nemori quickstart example."""
import asyncio
from nemori import NemoriMemory, MemoryConfig


async def main():
    config = MemoryConfig(
        dsn="postgresql://localhost/nemori",
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


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add examples/quickstart.py
git commit -m "docs: update quickstart example for async API"
```

---

### Task 23: Final Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Type check**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -m mypy src/ --ignore-missing-imports`
Expected: No errors (or only known third-party stubs)

- [ ] **Step 3: Verify imports work**

Run: `cd /Users/nanjiayan/Desktop/Nemori/nemori && python -c "from nemori import NemoriMemory, MemoryConfig, SearchMethod; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "refactor: complete Nemori PostgreSQL + async refactoring"
```

