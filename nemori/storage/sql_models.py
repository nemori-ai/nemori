"""
SQLModel definitions for Nemori storage layer.

This module provides type-safe SQLModel classes for database operations,
supporting both DuckDB and future PostgreSQL/MySQL implementations.
"""

from datetime import datetime

from sqlmodel import Column, Field, SQLModel, Text


class RawDataTable(SQLModel, table=True):
    """SQLModel for raw event data storage."""

    __tablename__ = "raw_data"

    data_id: str = Field(primary_key=True, max_length=255)
    data_type: str = Field(max_length=50, index=True)
    content: str = Field(sa_column=Column(Text))
    source: str = Field(max_length=255, index=True)
    timestamp: datetime = Field(index=True)
    duration: float | None = None
    timezone: str | None = Field(default=None, max_length=50)
    precision: str | None = Field(default="second", max_length=20)
    event_metadata: str = Field(sa_column=Column(Text))
    processed: bool = Field(default=False, index=True)
    processing_version: str = Field(default="1.0", max_length=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now())


class EpisodeTable(SQLModel, table=True):
    """SQLModel for episode storage."""

    __tablename__ = "episodes"

    episode_id: str = Field(primary_key=True, max_length=255)
    owner_id: str = Field(max_length=255, index=True)
    episode_type: str = Field(max_length=50, index=True)
    level: int = Field(index=True)
    title: str = Field(max_length=500)
    content: str = Field(sa_column=Column(Text))
    summary: str = Field(sa_column=Column(Text))
    timestamp: datetime = Field(index=True)
    duration: float | None = None
    timezone: str | None = Field(default=None, max_length=50)
    precision: str | None = Field(default="second", max_length=20)
    event_metadata: str = Field(sa_column=Column(Text))
    structured_data: str = Field(sa_column=Column(Text))
    search_keywords: str = Field(sa_column=Column(Text))
    embedding_vector: str | None = Field(default=None, sa_column=Column(Text))
    recall_count: int = Field(default=0)
    importance_score: float = Field(default=0.0, index=True)
    last_accessed: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now())


class EpisodeRawDataTable(SQLModel, table=True):
    """SQLModel for episode-raw data relationships."""

    __tablename__ = "episode_raw_data"

    episode_id: str = Field(primary_key=True, max_length=255)
    raw_data_id: str = Field(primary_key=True, max_length=255)


class BaseSQLRepository:
    """Base class for SQL-based repositories with common security methods."""

    def __init__(self, connection):
        self.connection = connection

    def validate_id(self, id_value: str) -> str:
        """Validate and sanitize ID values to prevent SQL injection."""
        if not isinstance(id_value, str):
            raise ValueError("ID must be a string")

        if not id_value.strip():
            raise ValueError("ID cannot be empty")

        # Remove any potentially dangerous characters
        if any(char in id_value for char in [";", "--", "/*", "*/", "xp_"]):
            raise ValueError("ID contains invalid characters")

        if len(id_value) > 255:
            raise ValueError("ID too long")

        return id_value.strip()

    def validate_limit_offset(self, limit: int | None, offset: int | None) -> tuple[int | None, int | None]:
        """Validate and sanitize limit and offset parameters."""
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("Limit must be a non-negative integer")
            if limit > 10000:  # Reasonable upper bound
                raise ValueError("Limit too large")

        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("Offset must be a non-negative integer")
            if offset > 1000000:  # Reasonable upper bound
                raise ValueError("Offset too large")

        return limit, offset

    def validate_enum_value(self, value: str, allowed_values: set[str], field_name: str) -> str:
        """Validate enum values."""
        if value not in allowed_values:
            raise ValueError(f"Invalid {field_name}: {value}")
        return value

    def sanitize_search_term(self, term: str) -> str:
        """Sanitize search terms to prevent SQL injection."""
        if not isinstance(term, str):
            raise ValueError("Search term must be a string")

        # Remove SQL metacharacters
        dangerous_chars = [";", "--", "/*", "*/", "xp_", "sp_", "exec", "execute"]
        term_lower = term.lower()

        for char in dangerous_chars:
            if char in term_lower:
                raise ValueError(f"Search term contains invalid pattern: {char}")

        return term.strip()
