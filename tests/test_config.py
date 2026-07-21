"""Tests for simplified MemoryConfig."""
import pytest
from nemori.config import MemoryConfig
from nemori.domain.exceptions import ConfigError


def test_default_config(monkeypatch):
    for name in (
        "LLM_API_KEY",
        "AGENT_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "NEMORI_LLM_MODEL",
        "AGENT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = MemoryConfig()
    assert "nemori" in cfg.dsn  # DSN resolved from env or default
    assert cfg.db_pool_min == 5
    assert cfg.db_pool_max == 20
    assert cfg.agent_id == "default"
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dimension == 1536
    assert cfg.buffer_size_min == 2
    assert cfg.observation_window == 20
    assert cfg.search_top_k_episodes == 10
    assert cfg.search_top_k_semantic == 20
    assert cfg.semantic_context_top_k == 10
    assert cfg.semantic_context_similarity_threshold == 0.70
    assert cfg.semantic_consolidation_top_k == 5


def test_config_reads_env_for_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    cfg = MemoryConfig()
    assert cfg.llm_api_key == "test-key-123"


def test_config_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
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


def test_config_invalid_observation_window():
    with pytest.raises(ConfigError, match="observation_window"):
        MemoryConfig(observation_window=0)


def test_config_invalid_llm_runtime_settings():
    with pytest.raises(ConfigError, match="llm_timeout"):
        MemoryConfig(llm_timeout=0)
    with pytest.raises(ConfigError, match="llm_retries"):
        MemoryConfig(llm_retries=0)


def test_config_has_qdrant_defaults(monkeypatch):
    for name in (
        "QDRANT_URL",
        "QDRANT_PORT",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = MemoryConfig()
    assert cfg.qdrant_url == "localhost"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_api_key is None
    assert cfg.qdrant_collection_prefix == "nemori"


def test_config_invalid_search_top_k_episodes():
    with pytest.raises(ConfigError, match="search_top_k_episodes"):
        MemoryConfig(search_top_k_episodes=0)


def test_config_invalid_search_top_k_semantic():
    with pytest.raises(ConfigError, match="search_top_k_semantic"):
        MemoryConfig(search_top_k_semantic=0)


def test_config_invalid_native_semantic_management_settings():
    with pytest.raises(ConfigError, match="semantic_context_top_k"):
        MemoryConfig(semantic_context_top_k=0)
    with pytest.raises(ConfigError, match="semantic_context_similarity_threshold"):
        MemoryConfig(semantic_context_similarity_threshold=1.1)
    with pytest.raises(ConfigError, match="semantic_consolidation_top_k"):
        MemoryConfig(semantic_consolidation_top_k=0)
