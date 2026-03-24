"""Tests for simplified MemoryConfig."""
import pytest
from nemori.config import MemoryConfig
from nemori.domain.exceptions import ConfigError


def test_default_config():
    cfg = MemoryConfig()
    assert cfg.dsn == "postgresql://localhost/nemori"
    assert cfg.db_pool_min == 5
    assert cfg.db_pool_max == 20
    assert cfg.agent_id == "default"
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


def test_config_custom_agent_id():
    cfg = MemoryConfig(agent_id="my-agent")
    assert cfg.agent_id == "my-agent"


def test_config_no_removed_fields():
    cfg = MemoryConfig()
    assert not hasattr(cfg, "storage_backend")
    assert not hasattr(cfg, "vector_index_backend")
    assert not hasattr(cfg, "lexical_index_backend")
    assert not hasattr(cfg, "chroma_persist_directory")
    assert not hasattr(cfg, "storage_path")
    assert hasattr(cfg, "enable_episode_merging")
    assert not hasattr(cfg, "max_workers")


def test_config_invalid_db_pool_min():
    with pytest.raises(ConfigError, match="db_pool_min"):
        MemoryConfig(db_pool_min=0)


def test_config_invalid_db_pool_max():
    with pytest.raises(ConfigError, match="db_pool_max"):
        MemoryConfig(db_pool_min=10, db_pool_max=5)


def test_config_invalid_buffer_size_min():
    with pytest.raises(ConfigError, match="buffer_size_min"):
        MemoryConfig(buffer_size_min=0)


def test_config_invalid_buffer_size_max():
    with pytest.raises(ConfigError, match="buffer_size_max"):
        MemoryConfig(buffer_size_min=10, buffer_size_max=5)


def test_config_invalid_embedding_dimension():
    with pytest.raises(ConfigError, match="embedding_dimension"):
        MemoryConfig(embedding_dimension=0)


def test_config_invalid_search_top_k_episodes():
    with pytest.raises(ConfigError, match="search_top_k_episodes"):
        MemoryConfig(search_top_k_episodes=0)


def test_config_invalid_search_top_k_semantic():
    with pytest.raises(ConfigError, match="search_top_k_semantic"):
        MemoryConfig(search_top_k_semantic=0)
