"""Tests for schema migrations."""
import pytest
from nemori.db.migrations import get_migrations


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


def test_migrations_no_pgvector_columns():
    """After Qdrant migration, no pgvector columns in initial schema."""
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    # No pgvector extension or embedding columns (tsvector is fine)
    assert "CREATE EXTENSION IF NOT EXISTS vector" not in initial_sql
    assert "embedding   vector(" not in initial_sql
    assert "hnsw" not in initial_sql.lower()


def test_migrations_use_coalesce_in_tsvector():
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    assert "coalesce" in initial_sql.lower()


def test_migration_3_adds_agent_id():
    migrations = get_migrations(embedding_dimension=1536)
    assert len(migrations) >= 3
    version, name, sql = migrations[2]
    assert version == 3
    assert "agent_id" in name.lower() or "agent_id" in sql
    assert "agent_id" in sql
    assert "episodes" in sql
    assert "semantic_memories" in sql
    assert "message_buffer" in sql


def test_initial_schema_has_agent_id():
    migrations = get_migrations(embedding_dimension=1536)
    initial_sql = migrations[0][2]
    assert "agent_id" in initial_sql


def test_migration_4_removes_pgvector():
    """Migration 4 should drop embedding columns and vector extension."""
    migrations = get_migrations(embedding_dimension=1536)
    assert len(migrations) >= 4
    version, name, sql = migrations[3]
    assert version == 4
    assert "DROP" in sql
    assert "embedding" in sql
