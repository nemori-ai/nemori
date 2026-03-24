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

    # Multi-tenant isolation
    agent_id: str = "default"

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

    # Qdrant
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection_prefix: str = "nemori"

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

    # Episode Merging
    enable_episode_merging: bool = True
    merge_similarity_threshold: float = 0.85
    merge_top_k: int = 5

    # Search
    search_top_k_episodes: int = 10
    search_top_k_semantic: int = 10

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
        if self.search_top_k_episodes < 1:
            raise ConfigError("search_top_k_episodes must be >= 1")
        if self.search_top_k_semantic < 1:
            raise ConfigError("search_top_k_semantic must be >= 1")
