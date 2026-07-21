# src/config.py
"""Nemori configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _resolve_llm_key() -> str:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("AGENT_API_KEY")
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


def _resolve_dsn() -> str:
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("DSN")
        or "postgresql://nemori:nemori@localhost:5432/nemori"
    )


def _resolve_base_url(env_key: str) -> str | None:
    if env_key == "LLM_BASE_URL":
        value = os.getenv(env_key) or os.getenv("AGENT_BASE_URL") or None
    else:
        value = os.getenv(env_key) or None
    if not value:
        return None
    normalized = value.strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in ("/chat/completions", "/embeddings", "/responses"):
        if lowered.endswith(suffix):
            return normalized[: -len(suffix)].rstrip("/")
    return normalized


@dataclass
class MemoryConfig:
    """Configuration for the Nemori memory system."""

    # Database
    dsn: str = field(default_factory=_resolve_dsn)
    db_pool_min: int = 5
    db_pool_max: int = 20

    # Multi-tenant isolation
    agent_id: str = "default"

    # LLM
    llm_model: str = field(
        default_factory=lambda: os.getenv("NEMORI_LLM_MODEL")
        or os.getenv("AGENT_MODEL")
        or "gpt-4o-mini"
    )
    llm_api_key: str = field(default_factory=_resolve_llm_key)
    llm_base_url: str | None = field(default_factory=lambda: _resolve_base_url("LLM_BASE_URL"))
    llm_max_concurrent: int = 10
    llm_timeout: float = 30.0
    llm_retries: int = 3
    llm_token_budget: int | None = None

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = field(default_factory=_resolve_embedding_key)
    embedding_base_url: str | None = field(default_factory=lambda: _resolve_base_url("EMBEDDING_BASE_URL"))
    embedding_dimension: int = 1536

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL") or "localhost")
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection_prefix: str = "nemori"

    # Buffer & Generation
    buffer_size_min: int = 2
    buffer_size_max: int = 25
    enable_batch_segmentation: bool = True
    batch_threshold: int = 20
    # Paper terminology: the observation window ``w`` used for each
    # segmentation request.  Nemori's reported LoCoMo experiments use w=20.
    observation_window: int = 20
    episode_min_messages: int = 2
    episode_max_messages: int = 25

    # Semantic Memory
    enable_semantic_memory: bool = True
    enable_prediction_correction: bool = True
    semantic_similarity_threshold: float = 0.70
    # Native management parameters reported in arXiv:2508.03341v4.
    semantic_context_top_k: int = 10
    semantic_context_similarity_threshold: float = 0.70
    semantic_consolidation_top_k: int = 5
    enable_semantic_consolidation: bool = True

    # Episode Merging
    enable_episode_merging: bool = True
    merge_similarity_threshold: float = 0.85
    merge_top_k: int = 5

    # Search
    search_top_k_episodes: int = 10
    search_top_k_semantic: int = 20

    def __post_init__(self) -> None:
        from nemori.domain.exceptions import ConfigError
        if self.db_pool_min < 1:
            raise ConfigError("db_pool_min must be >= 1")
        if self.db_pool_max < self.db_pool_min:
            raise ConfigError("db_pool_max must be >= db_pool_min")
        if self.buffer_size_min < 1:
            raise ConfigError("buffer_size_min must be >= 1")
        if self.buffer_size_max < self.buffer_size_min:
            raise ConfigError("buffer_size_max must be >= buffer_size_min")
        if self.llm_timeout <= 0:
            raise ConfigError("llm_timeout must be > 0")
        if self.llm_retries < 1:
            raise ConfigError("llm_retries must be >= 1")
        if self.batch_threshold < 1:
            raise ConfigError("batch_threshold must be >= 1")
        if self.observation_window < 1:
            raise ConfigError("observation_window must be >= 1")
        if self.merge_top_k < 1:
            raise ConfigError("merge_top_k must be >= 1")
        if self.semantic_context_top_k < 1:
            raise ConfigError("semantic_context_top_k must be >= 1")
        if not 0 <= self.semantic_context_similarity_threshold <= 1:
            raise ConfigError(
                "semantic_context_similarity_threshold must be between 0 and 1"
            )
        if self.semantic_consolidation_top_k < 1:
            raise ConfigError("semantic_consolidation_top_k must be >= 1")
        if self.search_top_k_episodes < 1:
            raise ConfigError("search_top_k_episodes must be >= 1")
        if self.search_top_k_semantic < 1:
            raise ConfigError("search_top_k_semantic must be >= 1")
