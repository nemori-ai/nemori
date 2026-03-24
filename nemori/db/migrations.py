"""Schema versioning and auto-migration."""
from __future__ import annotations


def get_migrations(embedding_dimension: int = 1536) -> list[tuple[int, str, str]]:
    initial_schema = """
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    CREATE TABLE IF NOT EXISTS episodes (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id     VARCHAR(255) NOT NULL,
        agent_id    VARCHAR(255) NOT NULL DEFAULT 'default',
        title       TEXT NOT NULL,
        content     TEXT NOT NULL,
        tsv         tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(content,''))
                    ) STORED,
        source_messages JSONB,
        metadata    JSONB DEFAULT '{}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_agent_user ON episodes(agent_id, user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON episodes USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_episodes_agent_user_created ON episodes(agent_id, user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS semantic_memories (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         VARCHAR(255) NOT NULL,
        agent_id        VARCHAR(255) NOT NULL DEFAULT 'default',
        content         TEXT NOT NULL,
        memory_type     VARCHAR(50) NOT NULL,
        tsv             tsvector GENERATED ALWAYS AS (
                            to_tsvector('simple', coalesce(content,''))
                        ) STORED,
        source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
        confidence      FLOAT DEFAULT 1.0,
        metadata        JSONB DEFAULT '{}'::jsonb,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_semantic_agent_user ON semantic_memories(agent_id, user_id);
    CREATE INDEX IF NOT EXISTS idx_semantic_tsv ON semantic_memories USING gin(tsv);
    CREATE INDEX IF NOT EXISTS idx_semantic_agent_user_type ON semantic_memories(agent_id, user_id, memory_type);

    CREATE TABLE IF NOT EXISTS message_buffer (
        id          BIGSERIAL PRIMARY KEY,
        user_id     VARCHAR(255) NOT NULL,
        agent_id    VARCHAR(255) NOT NULL DEFAULT 'default',
        role        VARCHAR(20) NOT NULL,
        content     TEXT NOT NULL,
        timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        processed   BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_buffer_agent_user_unprocessed
        ON message_buffer(agent_id, user_id) WHERE NOT processed;
    """

    return [
        (1, "initial schema", initial_schema),
        (2, "multimodal buffer content", """
    ALTER TABLE message_buffer ALTER COLUMN content TYPE JSONB USING to_jsonb(content);
"""),
        (3, "add agent_id for multi-tenant isolation", """
    ALTER TABLE episodes ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NOT NULL DEFAULT 'default';
    ALTER TABLE semantic_memories ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NOT NULL DEFAULT 'default';
    ALTER TABLE message_buffer ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255) NOT NULL DEFAULT 'default';

    CREATE INDEX IF NOT EXISTS idx_episodes_agent_user ON episodes(agent_id, user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_agent_user_created ON episodes(agent_id, user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_semantic_agent_user ON semantic_memories(agent_id, user_id);
    CREATE INDEX IF NOT EXISTS idx_semantic_agent_user_type ON semantic_memories(agent_id, user_id, memory_type);
    CREATE INDEX IF NOT EXISTS idx_buffer_agent_user_unprocessed ON message_buffer(agent_id, user_id) WHERE NOT processed;

    DROP INDEX IF EXISTS idx_episodes_user_id;
    DROP INDEX IF EXISTS idx_episodes_created_at;
    DROP INDEX IF EXISTS idx_semantic_user_id;
    DROP INDEX IF EXISTS idx_semantic_type;
    DROP INDEX IF EXISTS idx_buffer_user_unprocessed;
"""),
        (4, "remove pgvector columns for Qdrant migration", """
    DROP INDEX IF EXISTS idx_episodes_embedding;
    DROP INDEX IF EXISTS idx_semantic_embedding;
    ALTER TABLE episodes DROP COLUMN IF EXISTS embedding;
    ALTER TABLE semantic_memories DROP COLUMN IF EXISTS embedding;
"""),
    ]
