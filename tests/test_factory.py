"""Tests for storage factory functionality."""

import pytest

from nemori.storage import (
    DuckDBEpisodicMemoryRepository,
    DuckDBRawDataRepository,
    MemoryEpisodicMemoryRepository,
    MemoryRawDataRepository,
    PostgreSQLEpisodicMemoryRepository,
    PostgreSQLRawDataRepository,
    StorageConfig,
    StorageError,
    UnsupportedBackendError,
    create_duckdb_config,
    create_episodic_memory_repository,
    create_memory_config,
    create_postgresql_config,
    create_raw_data_repository,
    create_repositories,
    get_supported_backends,
    validate_config,
)


class TestStorageFactory:
    """Test storage factory functions."""

    def test_get_supported_backends(self):
        """Test getting supported backend types."""
        backends = get_supported_backends()
        assert isinstance(backends, list)
        assert "memory" in backends
        assert "duckdb" in backends
        assert "postgresql" in backends

    def test_create_memory_config(self):
        """Test creating memory storage configuration."""
        config = create_memory_config(batch_size=500, cache_size=1000)

        assert config.backend_type == "memory"
        assert config.connection_string is None
        assert config.batch_size == 500
        assert config.cache_size == 1000

    def test_create_duckdb_config(self):
        """Test creating DuckDB storage configuration."""
        config = create_duckdb_config(db_path="/tmp/test.duckdb", batch_size=1000)

        assert config.backend_type == "duckdb"
        assert config.connection_string == "/tmp/test.duckdb"
        assert config.batch_size == 1000

    def test_create_duckdb_config_no_path(self):
        """Test creating DuckDB configuration without path (in-memory)."""
        config = create_duckdb_config()

        assert config.backend_type == "duckdb"
        assert config.connection_string is None

    def test_create_postgresql_config_with_password(self):
        """Test creating PostgreSQL configuration with password."""
        config = create_postgresql_config(
            host="localhost", port=5432, database="test_db", username="test_user", password="test_pass", batch_size=2000
        )

        assert config.backend_type == "postgresql"
        assert "postgresql+asyncpg://" in config.connection_string
        assert "test_user:test_pass@localhost:5432/test_db" in config.connection_string
        assert config.batch_size == 2000

    def test_create_postgresql_config_no_password(self):
        """Test creating PostgreSQL configuration without password."""
        config = create_postgresql_config(host="localhost", database="test_db", username="test_user")

        assert config.backend_type == "postgresql"
        assert "postgresql+asyncpg://" in config.connection_string
        assert "test_user@localhost:5432/test_db" in config.connection_string

    def test_validate_config_memory(self):
        """Test validating memory configuration."""
        config = create_memory_config()
        validate_config(config)  # Should not raise

    def test_validate_config_duckdb(self):
        """Test validating DuckDB configuration."""
        config = create_duckdb_config(db_path="/tmp/test.duckdb")
        validate_config(config)  # Should not raise

    def test_validate_config_postgresql(self):
        """Test validating PostgreSQL configuration."""
        config = create_postgresql_config(
            host="localhost", database="test_db", username="test_user", password="test_pass"
        )
        validate_config(config)  # Should not raise

    def test_validate_config_invalid_backend(self):
        """Test validation with invalid backend type."""
        config = StorageConfig(backend_type="invalid")

        with pytest.raises(UnsupportedBackendError):
            validate_config(config)

    def test_validate_config_no_backend_type(self):
        """Test validation with missing backend type."""
        config = StorageConfig(backend_type="")

        with pytest.raises(StorageError):
            validate_config(config)

    def test_validate_config_postgresql_no_connection_string(self):
        """Test validation of PostgreSQL config without connection string."""
        config = StorageConfig(backend_type="postgresql")

        with pytest.raises(StorageError, match="PostgreSQL backend requires a connection string"):
            validate_config(config)

    def test_validate_config_postgresql_invalid_connection_string(self):
        """Test validation of PostgreSQL config with invalid connection string."""
        config = StorageConfig(backend_type="postgresql", connection_string="mysql://user:pass@localhost/db")

        with pytest.raises(StorageError, match="must start with 'postgresql://'"):
            validate_config(config)

    def test_validate_config_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        config = StorageConfig(backend_type="memory", batch_size=0)

        with pytest.raises(StorageError, match="Batch size must be positive"):
            validate_config(config)

    def test_validate_config_negative_cache_size(self):
        """Test validation with negative cache size."""
        config = StorageConfig(backend_type="memory", cache_size=-1)

        with pytest.raises(StorageError, match="Cache size cannot be negative"):
            validate_config(config)

    def test_validate_config_invalid_embedding_dimensions(self):
        """Test validation with invalid embedding dimensions."""
        config = StorageConfig(backend_type="memory", embedding_dimensions=0)

        with pytest.raises(StorageError, match="Embedding dimensions must be positive"):
            validate_config(config)

    def test_create_repositories_memory(self):
        """Test creating memory repositories."""
        config = create_memory_config()
        raw_repo, episode_repo = create_repositories(config)

        assert isinstance(raw_repo, MemoryRawDataRepository)
        assert isinstance(episode_repo, MemoryEpisodicMemoryRepository)

    def test_create_repositories_duckdb(self):
        """Test creating DuckDB repositories."""
        config = create_duckdb_config()
        raw_repo, episode_repo = create_repositories(config)

        assert isinstance(raw_repo, DuckDBRawDataRepository)
        assert isinstance(episode_repo, DuckDBEpisodicMemoryRepository)

    def test_create_repositories_postgresql(self):
        """Test creating PostgreSQL repositories."""
        config = create_postgresql_config(
            host="localhost", database="test_db", username="test_user", password="test_pass"
        )
        raw_repo, episode_repo = create_repositories(config)

        assert isinstance(raw_repo, PostgreSQLRawDataRepository)
        assert isinstance(episode_repo, PostgreSQLEpisodicMemoryRepository)

    def test_create_repositories_unsupported_backend(self):
        """Test creating repositories with unsupported backend."""
        config = StorageConfig(backend_type="unsupported")

        with pytest.raises(UnsupportedBackendError):
            create_repositories(config)

    def test_create_raw_data_repository_memory(self):
        """Test creating memory raw data repository."""
        config = create_memory_config()
        repo = create_raw_data_repository(config)

        assert isinstance(repo, MemoryRawDataRepository)

    def test_create_raw_data_repository_duckdb(self):
        """Test creating DuckDB raw data repository."""
        config = create_duckdb_config()
        repo = create_raw_data_repository(config)

        assert isinstance(repo, DuckDBRawDataRepository)

    def test_create_raw_data_repository_postgresql(self):
        """Test creating PostgreSQL raw data repository."""
        config = create_postgresql_config(host="localhost", database="test_db", username="test_user")
        repo = create_raw_data_repository(config)

        assert isinstance(repo, PostgreSQLRawDataRepository)

    def test_create_episodic_memory_repository_memory(self):
        """Test creating memory episodic memory repository."""
        config = create_memory_config()
        repo = create_episodic_memory_repository(config)

        assert isinstance(repo, MemoryEpisodicMemoryRepository)

    def test_create_episodic_memory_repository_duckdb(self):
        """Test creating DuckDB episodic memory repository."""
        config = create_duckdb_config()
        repo = create_episodic_memory_repository(config)

        assert isinstance(repo, DuckDBEpisodicMemoryRepository)

    def test_create_episodic_memory_repository_postgresql(self):
        """Test creating PostgreSQL episodic memory repository."""
        config = create_postgresql_config(host="localhost", database="test_db", username="test_user")
        repo = create_episodic_memory_repository(config)

        assert isinstance(repo, PostgreSQLEpisodicMemoryRepository)

    def test_factory_error_handling(self):
        """Test factory error handling."""
        # Test with configuration that would cause an error during creation
        config = StorageConfig(backend_type="memory")

        # Simulate an error by providing invalid config to a function that expects it
        with pytest.raises(Exception, match="Unsupported backend type: invalid"):
            # This should trigger the error handling in the factory
            config.backend_type = "invalid"
            create_repositories(config)
