"""Factory for assembling MemorySystem components."""
from __future__ import annotations

from nemori.config import MemoryConfig
from nemori.db.connection import DatabaseManager
from nemori.db.episode_store import PgEpisodeStore
from nemori.db.semantic_store import PgSemanticStore
from nemori.db.buffer_store import PgMessageBufferStore
from nemori.llm.client import AsyncLLMClient
from nemori.llm.orchestrator import LLMOrchestrator
from nemori.llm.generators.episode import EpisodeGenerator
from nemori.llm.generators.semantic import SemanticGenerator
from nemori.services.embedding import AsyncEmbeddingClient
from nemori.services.event_bus import EventBus
from nemori.search.unified import UnifiedSearch
from nemori.core.memory_system import MemorySystem


async def create_memory_system(config: MemoryConfig, db: DatabaseManager) -> MemorySystem:
    """Assemble a fully-wired MemorySystem from config and database manager."""
    episode_store = PgEpisodeStore(db)
    semantic_store = PgSemanticStore(db)
    buffer_store = PgMessageBufferStore(db)

    llm_client = AsyncLLMClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
    orchestrator = LLMOrchestrator(
        provider=llm_client,
        default_model=config.llm_model,
        max_concurrent=config.llm_max_concurrent,
        token_budget=config.llm_token_budget,
    )
    embedding = AsyncEmbeddingClient(
        api_key=config.embedding_api_key,
        model=config.embedding_model,
        base_url=config.embedding_base_url,
        dimensions=config.embedding_dimension,
    )
    episode_gen = EpisodeGenerator(orchestrator=orchestrator, embedding=embedding)
    semantic_gen = SemanticGenerator(
        orchestrator=orchestrator,
        embedding=embedding,
        enable_prediction_correction=config.enable_prediction_correction,
    )
    event_bus = EventBus()
    search = UnifiedSearch(episode_store, semantic_store, embedding)

    return MemorySystem(
        config=config,
        db=db,
        episode_store=episode_store,
        semantic_store=semantic_store,
        buffer_store=buffer_store,
        orchestrator=orchestrator,
        embedding=embedding,
        episode_generator=episode_gen,
        semantic_generator=semantic_gen,
        event_bus=event_bus,
        search=search,
        agent_id=config.agent_id,
    )
