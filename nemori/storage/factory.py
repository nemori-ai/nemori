"""
Storage factory for creating storage backends.

This module provides factory functions to create appropriate storage
repositories based on configuration.
"""

from .duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from .jsonl_storage import JSONLEpisodicMemoryRepository, JSONLRawDataRepository
from .memory_storage import MemoryEpisodicMemoryRepository, MemoryRawDataRepository
from .postgresql_storage import PostgreSQLEpisodicMemoryRepository, PostgreSQLRawDataRepository
from .repository import EpisodicMemoryRepository, RawDataRepository
from .storage_types import StorageConfig


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class UnsupportedBackendError(StorageError):
    """Raised when an unsupported storage backend is requested."""

    pass


def create_repositories(config: StorageConfig) -> tuple[RawDataRepository, EpisodicMemoryRepository]:
    """
    Create storage repositories based on configuration.

    Args:
        config: Storage configuration specifying backend type and settings

    Returns:
        Tuple of (RawDataRepository, EpisodicMemoryRepository)

    Raises:
        UnsupportedBackendError: If the backend type is not supported
        StorageError: If there's an error creating the repositories
    """
    backend_type = config.backend_type.lower()

    try:
        if backend_type == "memory":
            raw_repo = MemoryRawDataRepository(config)
            episode_repo = MemoryEpisodicMemoryRepository(config)

        elif backend_type == "duckdb":
            raw_repo = DuckDBRawDataRepository(config)
            episode_repo = DuckDBEpisodicMemoryRepository(config)

        elif backend_type == "postgresql":
            raw_repo = PostgreSQLRawDataRepository(config)
            episode_repo = PostgreSQLEpisodicMemoryRepository(config)

        elif backend_type == "jsonl":
            raw_repo = JSONLRawDataRepository(config)
            episode_repo = JSONLEpisodicMemoryRepository(config)

        else:
            raise UnsupportedBackendError(f"Unsupported backend type: {backend_type}")

        return raw_repo, episode_repo

    except Exception as e:
        if isinstance(e, UnsupportedBackendError):
            raise
        raise StorageError(f"Failed to create repositories for backend '{backend_type}': {e}") from e


def create_raw_data_repository(config: StorageConfig) -> RawDataRepository:
    """
    Create a raw data repository based on configuration.

    Args:
        config: Storage configuration

    Returns:
        RawDataRepository instance

    Raises:
        UnsupportedBackendError: If the backend type is not supported
        StorageError: If there's an error creating the repository
    """
    backend_type = config.backend_type.lower()

    try:
        if backend_type == "memory":
            return MemoryRawDataRepository(config)
        elif backend_type == "duckdb":
            return DuckDBRawDataRepository(config)
        elif backend_type == "postgresql":
            return PostgreSQLRawDataRepository(config)
        elif backend_type == "jsonl":
            return JSONLRawDataRepository(config)
        else:
            raise UnsupportedBackendError(f"Unsupported backend type: {backend_type}")

    except Exception as e:
        if isinstance(e, UnsupportedBackendError):
            raise
        raise StorageError(f"Failed to create raw data repository for backend '{backend_type}': {e}") from e


def create_episodic_memory_repository(config: StorageConfig) -> EpisodicMemoryRepository:
    """
    Create an episodic memory repository based on configuration.

    Args:
        config: Storage configuration

    Returns:
        EpisodicMemoryRepository instance

    Raises:
        UnsupportedBackendError: If the backend type is not supported
        StorageError: If there's an error creating the repository
    """
    backend_type = config.backend_type.lower()

    try:
        if backend_type == "memory":
            return MemoryEpisodicMemoryRepository(config)
        elif backend_type == "duckdb":
            return DuckDBEpisodicMemoryRepository(config)
        elif backend_type == "postgresql":
            return PostgreSQLEpisodicMemoryRepository(config)
        elif backend_type == "jsonl":
            return JSONLEpisodicMemoryRepository(config)
        else:
            raise UnsupportedBackendError(f"Unsupported backend type: {backend_type}")

    except Exception as e:
        if isinstance(e, UnsupportedBackendError):
            raise
        raise StorageError(f"Failed to create episodic memory repository for backend '{backend_type}': {e}") from e


def get_supported_backends() -> list[str]:
    """
    Get list of supported storage backends.

    Returns:
        List of supported backend names
    """
    return ["memory", "duckdb", "postgresql", "jsonl"]


def validate_config(config: StorageConfig) -> None:
    """
    Validate storage configuration.

    Args:
        config: Storage configuration to validate

    Raises:
        StorageError: If the configuration is invalid
    """
    if not config.backend_type:
        raise StorageError("Backend type is required")

    if config.backend_type.lower() not in get_supported_backends():
        raise UnsupportedBackendError(f"Unsupported backend type: {config.backend_type}")

    # Backend-specific validation
    backend_type = config.backend_type.lower()

    if backend_type == "postgresql":
        if not config.connection_string:
            raise StorageError("PostgreSQL backend requires a connection string")

        # Basic connection string validation
        if not config.connection_string.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise StorageError(
                "PostgreSQL connection string must start with 'postgresql://' or 'postgresql+asyncpg://'"
            )

    elif backend_type == "duckdb":
        # DuckDB can work with or without a connection string (file path)
        pass

    elif backend_type == "memory":
        # Memory backend doesn't require a connection string
        pass

    elif backend_type == "jsonl":
        # JSONL backend can work with or without a connection string (directory path)
        pass

    # Validate other settings
    if config.batch_size <= 0:
        raise StorageError("Batch size must be positive")

    if config.cache_size < 0:
        raise StorageError("Cache size cannot be negative")

    if config.embedding_dimensions <= 0:
        raise StorageError("Embedding dimensions must be positive")


def create_postgresql_config(
    host: str = "localhost",
    port: int = 5432,
    database: str = "nemori",
    username: str = "postgres",
    password: str | None = None,
    **kwargs,
) -> StorageConfig:
    """
    Create a PostgreSQL storage configuration.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        username: Username
        password: Password (optional)
        **kwargs: Additional StorageConfig parameters

    Returns:
        StorageConfig configured for PostgreSQL
    """
    # Build connection string
    if password:
        connection_string = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    else:
        connection_string = f"postgresql+asyncpg://{username}@{host}:{port}/{database}"

    return StorageConfig(backend_type="postgresql", connection_string=connection_string, **kwargs)


def create_duckdb_config(db_path: str | None = None, **kwargs) -> StorageConfig:
    """
    Create a DuckDB storage configuration.

    Args:
        db_path: Path to DuckDB file (optional, defaults to in-memory)
        **kwargs: Additional StorageConfig parameters

    Returns:
        StorageConfig configured for DuckDB
    """
    return StorageConfig(backend_type="duckdb", connection_string=db_path, **kwargs)


def create_memory_config(**kwargs) -> StorageConfig:
    """
    Create an in-memory storage configuration.

    Args:
        **kwargs: Additional StorageConfig parameters

    Returns:
        StorageConfig configured for in-memory storage
    """
    return StorageConfig(backend_type="memory", **kwargs)


def create_jsonl_config(data_dir: str | None = None, **kwargs) -> StorageConfig:
    """
    Create a JSONL storage configuration.

    Args:
        data_dir: Directory path for JSONL files (optional, defaults to "nemori_data")
        **kwargs: Additional StorageConfig parameters

    Returns:
        StorageConfig configured for JSONL storage
    """
    return StorageConfig(backend_type="jsonl", connection_string=data_dir, **kwargs)
