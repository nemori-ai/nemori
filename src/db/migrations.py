"""Schema versioning and auto-migration."""
from __future__ import annotations


def get_migrations(embedding_dimension: int = 1536) -> list[tuple[int, str, str]]:
    dim = embedding_dimension

    initial_schema = f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    CREATE TABLE IF NOT EXISTS episodes (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id     VARCHAR(255) NOT NULL,
        title       TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector({dim}),
        tsv         tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content,''))
                    ) STORED,
        source_messages JSONB,
        metadata    JSONB DEFAULT '{{}}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON episodes USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS semantic_memories (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         VARCHAR(255) NOT NULL,
        content         TEXT NOT NULL,
        memory_type     VARCHAR(50) NOT NULL,
        embedding       vector({dim}),
        tsv             tsvector GENERATED ALWAYS AS (
                            to_tsvector('simple', coalesce(content,''))
                        ) STORED,
        source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
        confidence      FLOAT DEFAULT 1.0,
        metadata        JSONB DEFAULT '{{}}'::jsonb,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_semantic_user_id ON semantic_memories(user_id);
    CREATE INDEX IF NOT EXISTS idx_semantic_embedding ON semantic_memories USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_semantic_tsv ON semantic_memories USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memories(user_id, memory_type);

    CREATE TABLE IF NOT EXISTS message_buffer (
        id          BIGSERIAL PRIMARY KEY,
        user_id     VARCHAR(255) NOT NULL,
        role        VARCHAR(20) NOT NULL,
        content     TEXT NOT NULL,
        timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed   BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_buffer_user_unprocessed
        ON message_buffer(user_id) WHERE NOT processed;
    """

    return [
        (1, "initial schema", initial_schema),
    ]
