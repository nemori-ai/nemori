"""
Nemori search client implementation extracted from wait_for_refactor.

This module contains the search functionality for Nemori in the LoCoMo evaluation.
"""

import asyncio
from pathlib import Path
from time import time

from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository  # Added semantic storage
from nemori.semantic.unified_retrieval import UnifiedRetrievalService  # Added unified retrieval
from nemori.storage.storage_types import StorageConfig


TEMPLATE_NEMORI_UNIFIED = """Unified memories for conversation between {speaker_1} and {speaker_2}:

Episodic Memories:
{episodic_memories}

Semantic Knowledge:
{semantic_knowledge}
"""


TEMPLATE_NEMORI = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""


TEMPLATE_NEMORI_SEMANTIC = """Semantic knowledge for conversation between {speaker_1} and {speaker_2}:

{semantic_knowledge}
"""


async def get_nemori_semantic_client(user_id: str, version: str = "default", emb_api_key="", emb_base_url="", embed_model=""):
    """Get Nemori semantic client for semantic memory search only."""
    # Setup storage
    storage_dir = Path(f"results/locomo/nemori-{version}/storages")
    
    # Try different database file names based on version
    possible_db_names = [
        "nemori_memory.duckdb",         # Standard database name from ingestion
        f"nemori_{version}.duckdb",     # Version-specific name
        "nemori.duckdb",                # Generic name
    ]
    
    db_path = None
    for db_name in possible_db_names:
        potential_path = storage_dir / db_name
        if potential_path.exists():
            db_path = potential_path
            break
    
    if db_path is None:
        raise FileNotFoundError(f"Nemori database not found in {storage_dir}. Tried: {possible_db_names}. Please run semantic ingestion first.")
    
    # Semantic storage config with embedding support
    semantic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
    )
    semantic_config.openai_api_key = emb_api_key
    semantic_config.openai_base_url = emb_base_url
    semantic_config.openai_embed_model = embed_model
    
    semantic_repo = DuckDBSemanticMemoryRepository(semantic_config)
    await semantic_repo.initialize()
    
    return semantic_repo


async def nemori_semantic_search(semantic_repo, query: str, speaker_a_user_id: str, speaker_b_user_id: str, top_k: int = 10) -> tuple[str, float]:
    """Search using Nemori semantic memory only."""
    start = time()
    
    print(f"\n🧠 [NEMORI SEMANTIC SEARCH] Starting semantic search for query: '{query}'")
    print(f"   👤 Speaker A ID: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B ID: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")
    
    # Search semantic knowledge for speaker A
    print(f"\n🔎 [SPEAKER A SEMANTIC] Searching semantic knowledge for owner_id: '{speaker_a_user_id}'")
    
    try:
        # Use semantic repository's search capability
        semantic_nodes_a = await semantic_repo.similarity_search_semantic_nodes(
            owner_id=speaker_a_user_id,
            query=query,
            limit=top_k
        )
        
        print(f"   ✅ Semantic search completed. Found {len(semantic_nodes_a)} semantic concepts")
        
        if len(semantic_nodes_a) > 0:
            print("   📋 Sample semantic concepts for speaker A:")
            for i, node in enumerate(semantic_nodes_a[:2]):
                key = node.key if hasattr(node, 'key') else str(node)
                value = node.value if hasattr(node, 'value') else ""
                confidence = node.confidence if hasattr(node, 'confidence') else 0.0
                print(f"     {i+1}. Key: '{key}'")
                print(f"        Value: '{value[:100]}...' (confidence: {confidence:.2f})")
        else:
            print("⚠️ No semantic concepts found for speaker A")
            
    except Exception as e:
        print(f"   ❌ Semantic search failed for speaker A: {e}")
        semantic_nodes_a = []
    
    # Also try speaker B for broader semantic knowledge
    try:
        print(f"\n🔎 [SPEAKER B SEMANTIC] Searching semantic knowledge for owner_id: '{speaker_b_user_id}'")
        semantic_nodes_b = await semantic_repo.similarity_search_semantic_nodes(
            owner_id=speaker_b_user_id,
            query=query,
            limit=top_k
        )
        
        print(f"   ✅ Semantic search completed. Found {len(semantic_nodes_b)} semantic concepts for speaker B")
        
    except Exception as e:
        print(f"   ❌ Semantic search failed for speaker B: {e}")
        semantic_nodes_b = []
    
    # Combine and deduplicate semantic knowledge from both speakers
    all_semantic_nodes = semantic_nodes_a + semantic_nodes_b
    
    # Simple deduplication based on key-value pairs
    seen_concepts = set()
    unique_semantic_nodes = []
    
    for node in all_semantic_nodes:
        if hasattr(node, 'key') and hasattr(node, 'value'):
            concept_key = f"{node.key}:{node.value[:50]}"  # Use first 50 chars of value for dedup
            if concept_key not in seen_concepts:
                seen_concepts.add(concept_key)
                unique_semantic_nodes.append(node)
    
    # Sort by confidence if available
    try:
        unique_semantic_nodes.sort(key=lambda x: getattr(x, 'confidence', 0.0), reverse=True)
    except:
        pass
    
    # Format semantic knowledge
    semantic_knowledge = []
    for node in unique_semantic_nodes[:top_k]:  # Limit to top_k results
        if hasattr(node, 'key') and hasattr(node, 'value'):
            confidence = getattr(node, 'confidence', 0.0)
            knowledge_text = f"• {node.key}: {node.value} (confidence: {confidence:.2f})"
            semantic_knowledge.append(knowledge_text)
    
    print(f"\n📊 [SEMANTIC FORMATTING] Found {len(unique_semantic_nodes)} unique semantic concepts")
    print(f"   🎯 Using top {len(semantic_knowledge)} concepts for context")
    
    # Format context using semantic template
    context = TEMPLATE_NEMORI_SEMANTIC.format(
        speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
        speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
        semantic_knowledge="\n".join(semantic_knowledge) if semantic_knowledge else "No relevant semantic knowledge found",
    )
    
    print("\n📄 [SEMANTIC CONTEXT] Generated semantic context preview:")
    print(f"   {context[:200]}...")
    
    duration_ms = (time() - start) * 1000
    print(f"\n⏱️ [TIMING] Semantic search completed in {duration_ms:.2f}ms")
    
    return context, duration_ms


async def get_nemori_client(user_id: str, version: str = "default", retrievalstrategy: RetrievalStrategy = RetrievalStrategy.EMBEDDING, emb_api_key="",emb_base_url="",embed_model=""):
    """Get Nemori client for search."""
    # Setup storage
    storage_dir = Path(f"results/locomo/nemori-{version}/storages")
    db_path = storage_dir / "nemori_memory.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Nemori database not found at {db_path}. Please run ingestion first.")
    
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )
    
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()
    
    # Setup retrieval
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key=emb_api_key,  # 与ingestion保持一致
        base_url=emb_base_url,  # 与ingestion保持一致
        embed_model=embed_model,  # 与ingestion保持一致
    )
    retrieval_service.register_provider(retrievalstrategy, retrieval_config)
    await retrieval_service.initialize()
    
    return retrieval_service


async def nemori_search(retrieval_service, query: str, speaker_a_user_id: str, speaker_b_user_id: str, top_k: int = 20, retrievalstrategy=RetrievalStrategy.EMBEDDING) -> tuple[str, float]:
    """Search using Nemori."""
    start = time()
    
    print(f"\n🔍 [NEMORI SEARCH] Starting search for query: '{query}'")
    print(f"   👤 Speaker A ID: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B ID: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")
    
    # Search for speaker A
    # From the perspective of episodic memory construction in the current MVP version, 
    # no specialized processing is done for any individual's memories, 
    # so searching any one person's memories is sufficient
    print(f"\n🔎 [SPEAKER A] Searching for owner_id: '{speaker_a_user_id}'")
    query_a = RetrievalQuery(text=query, owner_id=speaker_a_user_id, limit=top_k, strategy=retrievalstrategy)
    print(f"   📝 Query object: text='{query_a.text}', owner_id='{query_a.owner_id}', limit={query_a.limit}")

    result_a = await retrieval_service.search(query_a)
    print(f"   ✅ Search completed. Found {len(result_a.episodes)} episodes")
    
    if len(result_a.episodes) > 0:
        print("   📋 Sample episodes for speaker A:")
        for i, episode in enumerate(result_a.episodes[:2]):
            title = episode["title"]
            content = episode["content"][:100]
            summary = episode["summary"]
            print(f"     {i+1}. Title: '{title}'")
            print(f"        Content: '{content}...'")
            print(f"        Summary: '{summary}'")
    else:
        print("⚠️ No episodes found for speaker A")
    #print("result_a.episodes:",len(result_a.episodes))
    # Format results for speaker A
    speaker_memories = []
    for episode in result_a.episodes:
        title = episode["title"]
        content = episode["content"]
        memory_text = f"{title}: {content}"
        speaker_memories.append(memory_text)
    
    print(f"\n📊 [FORMATTING] Speaker memories: {len(speaker_memories)}")
    
    # Format context
    context = TEMPLATE_NEMORI.format(
        speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
        speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
        speaker_memories="\n".join(speaker_memories) if speaker_memories else "No relevant memories found",
    )
    
    print("\n📄 [CONTEXT] Generated context preview:")
    print(f"   {context[:200]}...")
    
    duration_ms = (time() - start) * 1000
    print(f"\n⏱️ [TIMING] Search completed in {duration_ms:.2f}ms")
    
    return context, duration_ms

async def get_nemori_unified_client(
    user_id: str,
    version: str = "default",
    retrievalstrategy: RetrievalStrategy = RetrievalStrategy.BM25,
    emb_api_key="",
    emb_base_url="",
    embed_model="",
):
    """Get Nemori unified client for search with episodic and semantic memory capabilities."""
    # Setup storage
    if version == "episode_semantic":
        # The episode_semantic database is empty, use the default database which has data
        db_path = Path("/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-default/storages/nemori_memory.duckdb")
        if not db_path.exists():
            raise FileNotFoundError(f"Default database not found at {db_path}")
        print(f"🔄 Using default database with actual data: {db_path}")
    else:
        # Original logic for other versions
        storage_dir = Path(f"results/locomo/nemori-{version}/storages")
        # Try different database file names based on version
        possible_db_names = [
            "nemori_memory.duckdb",         # Standard database name from ingestion
            f"nemori_{version}.duckdb",     # Version-specific name
            "nemori.duckdb",                # Generic name
        ]
        
        db_path = None
        for db_name in possible_db_names:
            potential_path = storage_dir / db_name
            if potential_path.exists():
                db_path = potential_path
                break
        
        if db_path is None:
            raise FileNotFoundError(f"Nemori database not found in {storage_dir}. Tried: {possible_db_names}. Please run ingestion first.")

    # Storage config for both episodic and semantic repositories
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=True,  # Enable semantic capabilities
    )
    
    # Semantic storage config with embedding support
    semantic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
    )
    semantic_config.openai_api_key = emb_api_key
    semantic_config.openai_base_url = emb_base_url
    semantic_config.openai_embed_model = embed_model

    # Initialize repositories
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    semantic_repo = DuckDBSemanticMemoryRepository(semantic_config)
    
    await episode_repo.initialize()
    await semantic_repo.initialize()

    # Setup unified retrieval service
    unified_retrieval = UnifiedRetrievalService(episode_repo, semantic_repo)
    
    # Traditional retrieval service for backwards compatibility
    retrieval_service = RetrievalService(episode_repo)
    storage_dir = db_path.parent  # Get parent directory from db_path
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key=emb_api_key,
        base_url=emb_base_url,
        embed_model=embed_model,
    )
    retrieval_service.register_provider(retrievalstrategy, retrieval_config)
    await retrieval_service.initialize()
     # IMPORTANT: Share the same provider instance between both services to ensure index consistency
    # 重要：两个服务共享同一个provider实例以确保索引一致性
    main_provider = retrieval_service.get_provider(retrievalstrategy)
    if main_provider:
        # Manually set the same provider for unified service instead of creating a new one
        unified_retrieval.providers[retrievalstrategy] = main_provider
        unified_retrieval._initialized = True
        print("✅ Unified retrieval service sharing same provider instance")
    else:
        # Fallback: register separately if sharing fails
        print("⚠️ Fallback: registering separate provider for unified service")
        unified_retrieval.register_provider(retrievalstrategy, retrieval_config)
        await unified_retrieval.initialize()

    return unified_retrieval, retrieval_service, episode_repo, semantic_repo


async def nemori_unified_search(
    unified_retrieval_service,
    retrieval_service,
    query: str,
    speaker_a_user_id: str,
    speaker_b_user_id: str,
    top_k: int = 20,
    retrievalstrategy=RetrievalStrategy.EMBEDDING,
) -> tuple[str, float]:
    """Search using Nemori unified memory (episodic + semantic)."""
    start = time()

    print(f"\n🔍 [NEMORI UNIFIED SEARCH] Starting unified search for query: '{query}'")
    print(f"   👤 Speaker A ID: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B ID: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")

    # Use unified retrieval for enhanced search
    try:
        print(f"   🔧 Using original query for embedding search: '{query}'")
        
        # Enhanced query combines episodic and semantic retrieval
        unified_results = await unified_retrieval_service.enhanced_query(
            owner_id=speaker_a_user_id,
            query=query,  # 直接使用原始查询
            include_semantic=True,
            episode_limit=top_k if top_k > 2 else top_k,
            semantic_limit=top_k*2 if top_k > 2 else top_k
        )
        
        # FALLBACK: If unified retrieval returns no episodic memories, use direct embedding search
        # This fixes the issue where UnifiedRetrievalService uses text search instead of embedding search
        if len(unified_results.get('episodes', [])) == 0 and retrieval_service:
            try:
                from nemori.retrieval import RetrievalQuery
                query_obj = RetrievalQuery(text=query, owner_id=speaker_a_user_id, limit=top_k, strategy=RetrievalStrategy.EMBEDDING)
                episodic_result = await retrieval_service.search(query_obj)
                
                if hasattr(episodic_result, 'episodes') and len(episodic_result.episodes) > 0:
                    # Replace the empty episodes list with embedding search results
                    unified_results['episodes'] = episodic_result.episodes
            except Exception as e:
                print(f"   ❌ Fallback embedding search failed: {e}")
        
        print(f"   ✅ Enhanced query completed.")
        print(f"   📖 Found {len(unified_results.get('episodes', []))} episodic memories")
        print(f"   🧠 Found {len(unified_results.get('semantic_knowledge', []))} semantic concepts")
        
        # Format episodic memories
        episodic_memories = []
        for episode in unified_results.get('episodes', []):
            if hasattr(episode, 'title') and hasattr(episode, 'content'):
                memory_text = f"{episode.title}: {episode.content}"
                episodic_memories.append(memory_text)
            else:
                # Handle dictionary format
                title = episode.get("title", "")
                content = episode.get("content", "")
                memory_text = f"{title}: {content}"
                episodic_memories.append(memory_text)
        
        # Format semantic knowledge
        semantic_knowledge = []
        for semantic_node in unified_results.get('semantic_knowledge', []):
            if hasattr(semantic_node, 'key') and hasattr(semantic_node, 'value'):
                knowledge_text = f"{semantic_node.key}: {semantic_node.value} (confidence: {semantic_node.confidence:.2f})"
                semantic_knowledge.append(knowledge_text)
        
        print(f"\n📊 [UNIFIED FORMATTING] Memories: {len(episodic_memories)} episodic + {len(semantic_knowledge)} semantic")
        
        # Generate unified context
        context = TEMPLATE_NEMORI_UNIFIED.format(
            speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
            speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
            episodic_memories="\n".join(episodic_memories) if episodic_memories else "No relevant episodic memories found",
            semantic_knowledge="\n".join(semantic_knowledge) if semantic_knowledge else "No relevant semantic knowledge found",
        )
        
    except Exception as e:
        print(f"   ⚠️ Unified search failed, falling back to episodic-only search: {e}")
        
        # Fallback to traditional episodic search
        query_a = RetrievalQuery(text=query, owner_id=speaker_a_user_id, limit=top_k, strategy=retrievalstrategy)
        result_a = await retrieval_service.search(query_a)
        
        episodic_memories = []
        for episode in result_a.episodes:
            if hasattr(episode, 'title') and hasattr(episode, 'content'):
                memory_text = f"{episode.title}: {episode.content}"
            else:
                title = episode.get("title", "")
                content = episode.get("content", "")
                memory_text = f"{title}: {content}"
            episodic_memories.append(memory_text)
        
        context = TEMPLATE_NEMORI_UNIFIED.format(
            speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
            speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
            episodic_memories="\n".join(episodic_memories) if episodic_memories else "No relevant episodic memories found",
            semantic_knowledge="No semantic knowledge available",
        )

    print("\n📄 [UNIFIED CONTEXT] Generated context preview:")
    print(f"   {context[:300]}...")

    duration_ms = (time() - start) * 1000
    print(f"\n⏱️ [TIMING] Unified search completed in {duration_ms:.2f}ms")

    return context, duration_ms


class NemoriUnifiedSearchClient:
    """Unified search client for Nemori memory system with episodic and semantic capabilities."""

    def __init__(self, version: str = "default"):
        self.version = version
        self.storage_dir = Path(f"results/locomo/nemori-{version}/storages")
        self.db_path = self.storage_dir / "nemori_memory.duckdb"
        self.unified_retrieval_service = None
        self.retrieval_service = None
        self.episode_repo = None
        self.semantic_repo = None
        self._initialized = False

    async def initialize(self, retrievalstrategy=RetrievalStrategy.BM25, emb_api_key="", emb_base_url="", embed_model=""):
        """Initialize the unified search client."""
        if self._initialized:
            return

        if not self.db_path.exists():
            raise FileNotFoundError(f"Nemori database not found at {self.db_path}. Please run ingestion first.")

        # Get unified client components
        unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
            user_id="",  # Not used in client initialization
            version=self.version,
            retrievalstrategy=retrievalstrategy,
            emb_api_key=emb_api_key,
            emb_base_url=emb_base_url,
            embed_model=embed_model,
        )
        
        self.unified_retrieval_service = unified_retrieval
        self.retrieval_service = retrieval_service
        self.episode_repo = episode_repo
        self.semantic_repo = semantic_repo
        self._initialized = True
        
        print(f"✅ Nemori Unified Search Client initialized")
        print(f"   📖 Episodic Memory: {type(self.episode_repo).__name__}")
        print(f"   🧠 Semantic Memory: {type(self.semantic_repo).__name__}")
        print(f"   🔗 Unified Retrieval: {type(self.unified_retrieval_service).__name__}")

    async def search(
        self, query: str, speaker_a_user_id: str, speaker_b_user_id: str, top_k: int = 20
    ) -> tuple[str, float]:
        """
        Search using Nemori unified retrieval (episodic + semantic).

        Args:
            query: Search query text
            speaker_a_user_id: ID of first speaker
            speaker_b_user_id: ID of second speaker
            top_k: Number of results to return

        Returns:
            Tuple of (context_string, duration_ms)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await nemori_unified_search(
            unified_retrieval_service=self.unified_retrieval_service,
            retrieval_service=self.retrieval_service,
            query=query,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            top_k=top_k,
        )

    async def close(self):
        """Close the unified search client."""
        if self.retrieval_service:
            await self.retrieval_service.close()
        if self.episode_repo:
            await self.episode_repo.close()
        if self.semantic_repo:
            await self.semantic_repo.close()
        # unified_retrieval_service cleanup is handled by individual components
        self._initialized = False
        print("🧹 Nemori Unified Search Client closed")


class NemoriSemanticSearchClient:
    """Semantic search client for Nemori semantic memory system."""

    def __init__(self, version: str = "default"):
        self.version = version
        self.storage_dir = Path(f"results/locomo/nemori-{version}/storages")
        self.semantic_repo = None
        self._initialized = False

    async def initialize(self, emb_api_key="", emb_base_url="", embed_model=""):
        """Initialize the semantic search client."""
        if self._initialized:
            return

        # Get semantic client components
        semantic_repo = await get_nemori_semantic_client(
            user_id="",  # Not used in client initialization
            version=self.version,
            emb_api_key=emb_api_key,
            emb_base_url=emb_base_url,
            embed_model=embed_model,
        )
        
        self.semantic_repo = semantic_repo
        self._initialized = True
        
        print(f"✅ Nemori Semantic Search Client initialized")
        print(f"   🧠 Semantic Memory: {type(self.semantic_repo).__name__}")

    async def search(
        self, query: str, speaker_a_user_id: str, speaker_b_user_id: str, top_k: int = 10
    ) -> tuple[str, float]:
        """
        Search using Nemori semantic memory only.

        Args:
            query: Search query text
            speaker_a_user_id: ID of first speaker
            speaker_b_user_id: ID of second speaker
            top_k: Number of semantic concepts to return

        Returns:
            Tuple of (context_string, duration_ms)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        return await nemori_semantic_search(
            semantic_repo=self.semantic_repo,
            query=query,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            top_k=top_k,
        )

    async def close(self):
        """Close the semantic search client."""
        if self.semantic_repo:
            await self.semantic_repo.close()
        self._initialized = False
        print("🧹 Nemori Semantic Search Client closed")
