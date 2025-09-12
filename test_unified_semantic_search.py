#!/usr/bin/env python3
"""
Test UnifiedRetrievalService semantic search functionality
"""

import asyncio
from pathlib import Path
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig
from nemori.semantic.unified_retrieval import UnifiedRetrievalService

async def test_unified_retrieval_semantic_search():
    """Test UnifiedRetrievalService semantic search"""
    print("🧪 Testing UnifiedRetrievalService Semantic Search")
    print("=" * 60)
    
    # Setup storage config
    db_path = Path("/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-test/storages/nemori_memory.duckdb")
    
    # Configure episodic storage
    episodic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )
    
    # Configure semantic storage with embedding support
    semantic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
    )
    
    semantic_config.openai_api_key = "EMPTY" 
    semantic_config.openai_base_url = "http://localhost:6007/v1"
    semantic_config.openai_embed_model = "qwen3-emb"
    
    print(f"📄 Database: {db_path}")
    print(f"🔧 Embedding API: {semantic_config.openai_base_url}")
    print(f"🤖 Model: {semantic_config.openai_embed_model}")
    
    # Initialize repositories
    episodic_repo = DuckDBEpisodicMemoryRepository(episodic_config)
    semantic_repo = DuckDBSemanticMemoryRepository(semantic_config)
    
    await episodic_repo.initialize()
    await semantic_repo.initialize()
    
    print(f"✅ Repositories initialized")
    
    # Create UnifiedRetrievalService
    unified_service = UnifiedRetrievalService(episodic_repo, semantic_repo)
    await unified_service.initialize()
    
    print(f"✅ UnifiedRetrievalService initialized")
    
    # Test semantic search for each owner
    test_queries = ["机器学习", "Python", "PyTorch", "数据预处理"]
    test_owners = ["李四_0", "张三_0"]
    
    for owner_id in test_owners:
        print(f"\n👤 Testing owner: {owner_id}")
        
        for query in test_queries:
            print(f"\n  🔍 Query: '{query}'")
            try:
                results = await unified_service.search_semantic_memories(owner_id, query, limit=3)
                
                if results:
                    print(f"    ✅ Found {len(results)} results via UnifiedRetrievalService:")
                    for i, node in enumerate(results):
                        print(f"      {i+1}. {node.key}: {node.value[:60]}...")
                else:
                    print(f"    ❌ No results found via UnifiedRetrievalService")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Clean up
    await unified_service.close()
    await episodic_repo.close() 
    await semantic_repo.close()
    
    print(f"\n🎉 Test complete!")

if __name__ == "__main__":
    asyncio.run(test_unified_retrieval_semantic_search())