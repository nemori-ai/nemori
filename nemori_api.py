"""
Nemori API - A unified interface for Nemori episodic and semantic memory system

This module provides a simple API to interact with the Nemori memory system,
supporting both episodic and semantic memory functionality.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.storage.duckdb_storage import (
    DuckDBEpisodicMemoryRepository,
    DuckDBRawDataRepository,
)
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig


class NemoriAPI:
    """
    Unified API for Nemori episodic and semantic memory system.
    
    This class provides a simple interface to:
    - Process conversations into episodic memories
    - Extract semantic knowledge from conversations
    - Search both episodic and semantic memories
    - Manage the complete memory lifecycle
    
    Example:
        api = NemoriAPI(db_path="memory.db", version="production")
        
        # Initialize with LLM provider
        await api.initialize(
            llm_api_key="your-api-key",
            llm_base_url="https://api.openai.com/v1",
            llm_model="gpt-4o-mini",
            embed_api_key="your-embed-key",
            embed_base_url="http://localhost:6007/v1",
            embed_model="text-embedding-3-small"
        )
        
        # Process a conversation
        conversation = {
            "conversation_id": "conv_001",
            "messages": [
                {
                    "speaker": "Alice",
                    "text": "I'm learning Python machine learning",
                    "timestamp": "2024-01-20T10:00:00Z"
                },
                {
                    "speaker": "Bob", 
                    "text": "Try scikit-learn and pandas for data processing",
                    "timestamp": "2024-01-20T10:02:00Z"
                }
            ]
        }
        
        episodes = await api.process_conversation(conversation)
        
        # Search memories
        results = await api.search("machine learning", owner_id="alice_conv_001", top_k=10)
        
        # Search semantic knowledge
        semantic_results = await api.search_semantic("Python libraries", owner_id="alice_conv_001")
    """
    
    def __init__(
        self, 
        db_path: Optional[str] = None, 
        version: str = "default",
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.BM25,
        max_concurrency: int = 1,
        enable_semantic: bool = True
    ):
        """
        Initialize NemoriAPI.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses default path
            version: Version identifier for organizing storage
            retrieval_strategy: Default retrieval strategy (BM25, EMBEDDING)
            max_concurrency: Maximum concurrent operations
            enable_semantic: Whether to enable semantic memory features
        """
        self.version = version
        self.max_concurrency = max_concurrency
        self.enable_semantic = enable_semantic
        self.retrieval_strategy = retrieval_strategy
        
        # Set up database path
        if db_path is None:
            self.db_dir = Path(f"nemori_data/{version}")
            self.db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.db_dir / "nemori_memory.duckdb"
        else:
            self.db_path = Path(db_path)
            self.db_dir = self.db_path.parent
            self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Components (initialized in initialize())
        self.llm_provider = None
        self.raw_data_repo = None
        self.episode_repo = None
        self.semantic_repo = None
        self.retrieval_service = None
        self.unified_retrieval_service = None
        self.episode_manager = None
        self.discovery_engine = None
        self.semantic_manager = None
        
        self._initialized = False
    
    async def initialize(
        self,
        llm_api_key: str,
        llm_base_url: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4o-mini",
        embed_api_key: Optional[str] = None,
        embed_base_url: str = "http://localhost:6007/v1", 
        embed_model: str = "text-embedding-3-small",
        temperature: float = 0.1,
        max_tokens: int = 16384
    ) -> bool:
        """
        Initialize the Nemori system with LLM and embedding providers.
        
        Args:
            llm_api_key: API key for LLM provider
            llm_base_url: Base URL for LLM API
            llm_model: LLM model name
            embed_api_key: API key for embedding provider (defaults to llm_api_key)
            embed_base_url: Base URL for embedding API
            embed_model: Embedding model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for LLM responses
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            print("🤖 Initializing Nemori API...")
            
            # Initialize LLM provider
            self.llm_provider = OpenAIProvider(
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            
            if not await self.llm_provider.test_connection():
                print("❌ LLM connection failed")
                return False
            print(f"✅ LLM connected: {llm_model}")
            
            # Set up storage
            await self._setup_storage(embed_api_key or llm_api_key, embed_base_url, embed_model)
            
            # Set up retrieval services  
            await self._setup_retrieval(embed_api_key or llm_api_key, embed_base_url, embed_model)
            
            # Set up episode manager
            await self._setup_episode_manager()
            
            self._initialized = True
            print("✅ Nemori API initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def _setup_storage(self, embed_api_key: str, embed_base_url: str, embed_model: str):
        """Set up storage repositories."""
        # Create storage configurations
        storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(self.db_path),
            batch_size=100,
            cache_size=1000,
            enable_semantic_search=False,
        )
        
        semantic_config = StorageConfig(
            backend_type="duckdb", 
            connection_string=str(self.db_path),
            batch_size=100,
            cache_size=1000,
        )
        semantic_config.openai_api_key = embed_api_key
        semantic_config.openai_base_url = embed_base_url
        semantic_config.openai_embed_model = embed_model
        
        # Initialize repositories
        self.raw_data_repo = DuckDBRawDataRepository(storage_config)
        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
        
        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()
        
        if self.enable_semantic:
            self.semantic_repo = DuckDBSemanticMemoryRepository(semantic_config)
            await self.semantic_repo.initialize()
            
        print(f"✅ Storage initialized: {self.db_path}")
    
    async def _setup_retrieval(self, embed_api_key: str, embed_base_url: str, embed_model: str):
        """Set up retrieval services."""
        self.retrieval_service = RetrievalService(self.episode_repo)
        
        if self.enable_semantic:
            self.unified_retrieval_service = UnifiedRetrievalService(
                self.episode_repo, self.semantic_repo
            )
        
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.db_dir)},
            api_key=embed_api_key,
            base_url=embed_base_url,
            embed_model=embed_model,
        )
        
        self.retrieval_service.register_provider(self.retrieval_strategy, retrieval_config)
        await self.retrieval_service.initialize()
        
        if self.enable_semantic:
            # Share provider instance for consistency
            main_provider = self.retrieval_service.get_provider(self.retrieval_strategy)
            if main_provider:
                self.unified_retrieval_service.providers[self.retrieval_strategy] = main_provider
                self.unified_retrieval_service._initialized = True
            else:
                self.unified_retrieval_service.register_provider(self.retrieval_strategy, retrieval_config)
                await self.unified_retrieval_service.initialize()
                
        print("✅ Retrieval services initialized")
    
    async def _setup_episode_manager(self):
        """Set up episode manager with semantic capabilities."""
        builder_registry = EpisodeBuilderRegistry()
        
        if self.enable_semantic:
            # Set up semantic discovery components
            self.discovery_engine = ContextAwareSemanticDiscoveryEngine(
                self.llm_provider, self.unified_retrieval_service
            )
            self.semantic_manager = SemanticEvolutionManager(
                self.semantic_repo, self.discovery_engine, self.unified_retrieval_service
            )
            
            # Use enhanced builder with semantic capabilities
            conversation_builder = EnhancedConversationEpisodeBuilder(
                llm_provider=self.llm_provider,
                semantic_manager=self.semantic_manager
            )
        else:
            # Use basic builder
            conversation_builder = ConversationEpisodeBuilder(
                llm_provider=self.llm_provider
            )
        
        builder_registry.register(conversation_builder)
        
        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=self.retrieval_service,
        )
        
        print("✅ Episode manager initialized")
    
    def _ensure_initialized(self):
        """Ensure API is initialized before operations."""
        if not self._initialized:
            raise RuntimeError("API not initialized. Call initialize() first.")
    
    def _convert_conversation_to_raw_data(self, conversation: Dict[str, Any]) -> RawEventData:
        """Convert conversation dictionary to RawEventData format."""
        conversation_id = conversation.get("conversation_id", "unknown")
        messages = conversation.get("messages", [])
        
        nemori_messages = []
        for msg in messages:
            speaker_name = msg["speaker"]
            speaker_id = f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"
            
            # Ensure timestamp is in correct format
            timestamp = msg["timestamp"]
            if isinstance(timestamp, str) and timestamp.endswith("Z"):
                timestamp = timestamp.replace("Z", "+00:00")
            
            nemori_messages.append({
                "speaker_id": speaker_id,
                "user_name": speaker_name,
                "content": msg["text"],
                "timestamp": timestamp,
            })
        
        if not nemori_messages:
            raise ValueError("Conversation has no messages")
        
        first_timestamp = datetime.fromisoformat(nemori_messages[0]["timestamp"])
        last_timestamp = datetime.fromisoformat(nemori_messages[-1]["timestamp"])
        duration = (last_timestamp - first_timestamp).total_seconds()
        
        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=nemori_messages,
            source="nemori_api",
            temporal_info=TemporalInfo(
                timestamp=first_timestamp,
                duration=duration,
                timezone="UTC",
            ),
            metadata={"conversation_id": conversation_id},
        )
    
    async def process_conversation(
        self, 
        conversation: Dict[str, Any], 
        enable_semantic_discovery: bool = None
    ) -> List[Any]:
        """
        Process a conversation into episodic memories and optionally extract semantic knowledge.
        
        Args:
            conversation: Conversation dictionary with format:
                {
                    "conversation_id": "unique_id",
                    "messages": [
                        {
                            "speaker": "Speaker Name",
                            "text": "Message content", 
                            "timestamp": "2024-01-20T10:00:00Z"
                        },
                        ...
                    ]
                }
            enable_semantic_discovery: Whether to extract semantic knowledge (defaults to API setting)
            
        Returns:
            List of created episodes
        """
        self._ensure_initialized()
        
        if enable_semantic_discovery is None:
            enable_semantic_discovery = self.enable_semantic
        
        print(f"🔄 Processing conversation: {conversation.get('conversation_id', 'unknown')}")
        
        # Convert to RawEventData format
        raw_data = self._convert_conversation_to_raw_data(conversation)
        
        # Get speaker IDs for episode creation
        speakers = {
            msg["speaker_id"] for msg in raw_data.content 
            if isinstance(msg, dict) and "speaker_id" in msg
        }
        
        all_episodes = []
        
        # Process episodes for each speaker
        for speaker_id in speakers:
            episode = await self.episode_manager.process_raw_data(raw_data, speaker_id)
            if episode:
                all_episodes.append(episode)
                
                # Perform semantic discovery if enabled
                if enable_semantic_discovery and self.semantic_manager:
                    try:
                        conversation_data = ConversationData(raw_data)
                        original_conversation = "\n".join([
                            f"{msg.speaker_id}: {msg.content}" 
                            for msg in conversation_data.messages
                        ])
                        
                        discovered_nodes = await self.semantic_manager.process_episode_for_semantics(
                            episode=episode,
                            original_content=original_conversation,
                        )
                        
                        if discovered_nodes:
                            print(f"✅ Discovered {len(discovered_nodes)} semantic concepts for {speaker_id}")
                        
                    except Exception as e:
                        print(f"⚠️ Semantic discovery failed for {speaker_id}: {e}")
        
        print(f"✅ Created {len(all_episodes)} episodes")
        return all_episodes
    
    async def search(
        self, 
        query: str, 
        owner_id: str, 
        top_k: int = 10,
        strategy: Optional[RetrievalStrategy] = None
    ) -> Dict[str, Any]:
        """
        Search episodic memories.
        
        Args:
            query: Search query text
            owner_id: Owner ID to search within
            top_k: Number of results to return
            strategy: Retrieval strategy to use (defaults to API default)
            
        Returns:
            Dictionary with search results and metadata
        """
        self._ensure_initialized()
        
        if strategy is None:
            strategy = self.retrieval_strategy
        
        retrieval_query = RetrievalQuery(
            text=query,
            owner_id=owner_id,
            limit=top_k,
            strategy=strategy
        )
        
        results = await self.retrieval_service.search(retrieval_query)
        
        return {
            "query": query,
            "owner_id": owner_id,
            "episodes": results.episodes if hasattr(results, 'episodes') else [],
            "total_results": len(results.episodes) if hasattr(results, 'episodes') else 0,
            "strategy": strategy.value
        }
    
    async def search_unified(
        self,
        query: str,
        owner_id: str,
        episode_limit: int = 10,
        semantic_limit: int = 5,
        include_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Search both episodic and semantic memories using unified retrieval.
        
        Args:
            query: Search query text
            owner_id: Owner ID to search within
            episode_limit: Maximum episodic results
            semantic_limit: Maximum semantic results  
            include_semantic: Whether to include semantic knowledge
            
        Returns:
            Dictionary with unified search results
        """
        self._ensure_initialized()
        
        if not self.enable_semantic or not self.unified_retrieval_service:
            # Fall back to episodic search only
            return await self.search(query, owner_id, episode_limit)
        
        results = await self.unified_retrieval_service.enhanced_query(
            owner_id=owner_id,
            query=query,
            include_semantic=include_semantic,
            episode_limit=episode_limit,
            semantic_limit=semantic_limit
        )
        
        return {
            "query": query,
            "owner_id": owner_id,
            "episodes": results.get('episodes', []),
            "semantic_knowledge": results.get('semantic_knowledge', []),
            "total_episodes": len(results.get('episodes', [])),
            "total_semantic": len(results.get('semantic_knowledge', [])),
        }
    
    async def search_semantic(
        self,
        query: str,
        owner_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search semantic knowledge only.
        
        Args:
            query: Search query text
            owner_id: Owner ID to search within
            top_k: Number of semantic concepts to return
            
        Returns:
            Dictionary with semantic search results
        """
        self._ensure_initialized()
        
        if not self.enable_semantic or not self.semantic_repo:
            return {
                "query": query,
                "owner_id": owner_id,
                "semantic_nodes": [],
                "total_results": 0,
                "error": "Semantic search not enabled or available"
            }
        
        try:
            semantic_nodes = await self.semantic_repo.similarity_search_semantic_nodes(
                owner_id=owner_id,
                query=query,
                limit=top_k
            )
            
            return {
                "query": query,
                "owner_id": owner_id,
                "semantic_nodes": semantic_nodes,
                "total_results": len(semantic_nodes)
            }
            
        except Exception as e:
            return {
                "query": query,
                "owner_id": owner_id, 
                "semantic_nodes": [],
                "total_results": 0,
                "error": str(e)
            }
    
    async def get_owner_stats(self, owner_id: str) -> Dict[str, Any]:
        """
        Get statistics for an owner's memories.
        
        Args:
            owner_id: Owner ID to get stats for
            
        Returns:
            Dictionary with memory statistics
        """
        self._ensure_initialized()
        
        stats = {
            "owner_id": owner_id,
            "episodic_count": 0,
            "semantic_count": 0,
            "total_memories": 0
        }
        
        try:
            # Get episodic memory count
            episode_result = await self.episode_repo.get_episodes_by_owner(owner_id)
            episodes = episode_result.episodes if hasattr(episode_result, 'episodes') else episode_result
            stats["episodic_count"] = len(episodes) if episodes else 0
            
            # Get semantic memory count
            if self.enable_semantic and self.semantic_repo:
                semantic_nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner_id)
                stats["semantic_count"] = len(semantic_nodes) if semantic_nodes else 0
            
            stats["total_memories"] = stats["episodic_count"] + stats["semantic_count"]
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    async def list_owners(self) -> List[str]:
        """
        List all owner IDs in the system.
        
        Returns:
            List of owner IDs
        """
        self._ensure_initialized()
        
        # This is a simplified implementation
        # In practice, you might want to query the database for all unique owner_ids
        owners = set()
        
        try:
            # Get sample episodes to extract owner IDs
            # This is not efficient for large datasets - consider adding a proper query
            sample_result = await self.episode_repo.search_episodes(
                query="", limit=1000  # Get a sample to find owners
            )
            
            if hasattr(sample_result, 'episodes'):
                for episode in sample_result.episodes:
                    if hasattr(episode, 'owner_id'):
                        owners.add(episode.owner_id)
            
        except Exception as e:
            print(f"⚠️ Error listing owners: {e}")
        
        return list(owners)
    
    async def close(self):
        """Close all resources and connections."""
        if not self._initialized:
            return
        
        print("🧹 Closing Nemori API...")
        
        try:
            if self.retrieval_service:
                await self.retrieval_service.close()
            if self.raw_data_repo:
                await self.raw_data_repo.close()
            if self.episode_repo:
                await self.episode_repo.close()
            if self.semantic_repo:
                await self.semantic_repo.close()
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
        
        self._initialized = False
        print("✅ Nemori API closed")
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb  # Silence unused variable warnings
        await self.close()


# Convenience functions for quick usage
async def create_nemori_api(
    db_path: Optional[str] = None,
    version: str = "default",
    llm_api_key: Optional[str] = None,
    llm_base_url: str = "https://api.openai.com/v1",
    llm_model: str = "gpt-4o-mini",
    embed_api_key: Optional[str] = None,
    embed_base_url: str = "http://localhost:6007/v1",
    embed_model: str = "text-embedding-3-small",
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.BM25,
    enable_semantic: bool = True,
    **kwargs
) -> NemoriAPI:
    """
    Create and initialize a NemoriAPI instance with default configuration.
    
    Args:
        db_path: Database path (uses default if None)
        version: Version identifier
        llm_api_key: LLM API key (uses environment variable if None)
        llm_base_url: LLM API base URL
        llm_model: LLM model name
        embed_api_key: Embedding API key (uses llm_api_key if None)
        embed_base_url: Embedding API base URL
        embed_model: Embedding model name
        retrieval_strategy: Default retrieval strategy
        enable_semantic: Enable semantic memory
        **kwargs: Additional arguments for NemoriAPI constructor
        
    Returns:
        Initialized NemoriAPI instance
        
    Example:
        api = await create_nemori_api(
            db_path="my_memory.db",
            llm_api_key="your-api-key"
        )
    """
    # Use environment variables as fallback
    if llm_api_key is None:
        llm_api_key = os.getenv('OPENAI_API_KEY')
        if not llm_api_key:
            raise ValueError("LLM API key required (pass llm_api_key or set OPENAI_API_KEY)")
    
    api = NemoriAPI(
        db_path=db_path,
        version=version,
        retrieval_strategy=retrieval_strategy,
        enable_semantic=enable_semantic,
        **kwargs
    )
    
    success = await api.initialize(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_api_key=embed_api_key or llm_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model
    )
    
    if not success:
        raise RuntimeError("Failed to initialize Nemori API")
    
    return api


# Example usage
async def example_usage():
    """Example of how to use the Nemori API."""
    
    # Clean up any existing database to avoid conflicts
    db_path = Path("example_memory.db")
    if db_path.exists():
        db_path.unlink()
    
    # Method 1: Manual initialization
    api = NemoriAPI(
        db_path="example_memory.db",
        version="example",
        enable_semantic=True
    )
    api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
    base_url = "https://jeniya.cn/v1"
    model = "gpt-4.1-mini"
    embed_api_key = "EMPTY"
    embed_base_url = "http://localhost:6007/v1"
    embed_model = "qwen3-emb"
    await api.initialize(
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=model,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_model=embed_model
    )
    
    # Process a conversation
    conversation = {
        "conversation_id": "example_conv_001",
        "messages": [
            {
                "speaker": "Alice",
                "text": "I'm working on a machine learning project using Python",
                "timestamp": "2024-01-20T10:00:00Z"
            },
            {
                "speaker": "Bob",
                "text": "Great! I recommend using scikit-learn for beginners and pandas for data manipulation",
                "timestamp": "2024-01-20T10:02:00Z"
            },
            {
                "speaker": "Alice", 
                "text": "What about deep learning? Should I use TensorFlow or PyTorch?",
                "timestamp": "2024-01-20T10:04:00Z"
            },
            {
                "speaker": "Bob",
                "text": "PyTorch is more intuitive for learning, but TensorFlow is better for production",
                "timestamp": "2024-01-20T10:06:00Z"
            }
        ]
    }
    
    # Process the conversation
    episodes = await api.process_conversation(conversation)
    print(f"Created {len(episodes)} episodes")
    
    # Search episodic memories
    alice_id = "alice_example_conv_001"
    search_results = await api.search(
        query="machine learning Python",
        owner_id=alice_id,
        top_k=5
    )
    print(f"Found {search_results['total_results']} episodic memories")
    
    # Search with unified retrieval (episodic + semantic)
    unified_results = await api.search_unified(
        query="Python libraries recommendations",
        owner_id=alice_id,
        episode_limit=5,
        semantic_limit=3
    )
    print(f"Found {unified_results['total_episodes']} episodes and {unified_results['total_semantic']} semantic concepts")
    
    # Search semantic knowledge only
    semantic_results = await api.search_semantic(
        query="deep learning frameworks",
        owner_id=alice_id,
        top_k=3
    )
    print(f"Found {semantic_results['total_results']} semantic concepts")
    
    # Get owner statistics
    stats = await api.get_owner_stats(alice_id)
    print(f"Owner {alice_id} has {stats['total_memories']} total memories")
    
    # List all owners
    owners = await api.list_owners()
    print(f"System has {len(owners)} owners: {owners}")
    
    # Clean up
    await api.close()


async def simple_test():
    """Simple test to verify API functionality."""
    print("🧪 Running simple API test...")
    
    # Clean up any existing database to avoid conflicts
    import shutil
    import tempfile
    
    # Use temporary directory to avoid conflicts
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_memory.db"
    
    try:
        api = NemoriAPI(
            db_path=str(db_path),
            version="test",
            enable_semantic=True
        )
        
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"
        model = "gpt-4o-mini"
        embed_api_key = "EMPTY"
        embed_base_url = "http://localhost:6007/v1"
        embed_model = "qwen3-emb"
        
        print("🔧 Initializing API...")
        success = await api.initialize(
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_model=model,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model
        )
        
        if not success:
            print("❌ Failed to initialize API")
            return
            
        print("✅ API initialized successfully")
        
        # Test simple conversation
        conversation = {
            "conversation_id": "test_001",
            "messages": [
                {
                    "speaker": "Alice",
                    "text": "Hello, how are you?",
                    "timestamp": "2024-01-20T10:00:00+00:00"
                },
                {
                    "speaker": "Bob",
                    "text": "I'm good, thanks for asking!",
                    "timestamp": "2024-01-20T10:01:00+00:00"
                }
            ]
        }
        
        print("🔄 Processing conversation...")
        episodes = await api.process_conversation(conversation)
        print(f"✅ Created {len(episodes)} episodes")
        
        # Test search
        alice_id = "alice_test_001"
        print(f"🔍 Testing search for owner: {alice_id}")
        
        search_results = await api.search(
            query="hello",
            owner_id=alice_id,
            top_k=5
        )
        print(f"✅ Found {search_results['total_results']} search results")
        
        # Clean up
        await api.close()
        shutil.rmtree(temp_dir)
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run simple test instead of full example to avoid database conflicts
    asyncio.run(simple_test())