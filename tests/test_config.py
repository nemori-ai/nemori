"""Tests for simplified MemoryConfig."""
import pytest
from nemori.config import MemoryConfig


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
    cfg = MemoryConfig()
    assert not hasattr(cfg, "storage_backend")
    assert not hasattr(cfg, "vector_index_backend")
    assert not hasattr(cfg, "lexical_index_backend")
    assert not hasattr(cfg, "chroma_persist_directory")
    assert not hasattr(cfg, "storage_path")
    assert not hasattr(cfg, "enable_episode_merging")
    assert not hasattr(cfg, "max_workers")
