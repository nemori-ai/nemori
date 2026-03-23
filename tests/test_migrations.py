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
