#!/usr/bin/env python3
"""
Test to understand why related_semantic_memories is empty
"""

import asyncio
from pathlib import Path
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig

async def test_semantic_search_progression():
    """Test semantic search results as data is progressively added"""
    print("🧪 Testing Semantic Search Progression")
    print("=" * 60)
    
    # Setup storage config
    db_path = Path("/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-test/storages/nemori_memory.duckdb")
    
    config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
    )
    
    # Add embedding configuration
    config.openai_api_key = "EMPTY"
    config.openai_base_url = "http://localhost:6007/v1"
    config.openai_embed_model = "qwen3-emb"
    
    # Initialize repository
    semantic_repo = DuckDBSemanticMemoryRepository(config)
    await semantic_repo.initialize()
    
    # Check existing semantic data
    from sqlmodel import Session, select
    from nemori.storage.sql_models import SemanticNodeTable
    
    with Session(semantic_repo.engine) as session:
        owners = session.exec(
            select(SemanticNodeTable.owner_id).distinct()
        ).all()
        
        print(f"📊 Current owners in database: {owners}")
        
        for owner_id in owners:
            nodes = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.owner_id == owner_id)
            ).all()
            print(f"   🧠 {owner_id}: {len(nodes)} semantic nodes")
    
    # Test search for new owner (should be empty) 
    print(f"\n🔍 Testing search for NEW owner 'test_user' (should return empty):")
    results_new = await semantic_repo.similarity_search_semantic_nodes("test_user", "machine learning", limit=5)
    print(f"   📄 Results for test_user: {len(results_new)}")
    
    # Test search for existing owner (should return results)
    if owners:
        existing_owner = owners[0]
        print(f"\n🔍 Testing search for EXISTING owner '{existing_owner}' (should return results):")
        results_existing = await semantic_repo.similarity_search_semantic_nodes(existing_owner, "machine learning", limit=5)
        print(f"   📄 Results for {existing_owner}: {len(results_existing)}")
        
        for i, node in enumerate(results_existing):
            print(f"      {i+1}. {node.key}: {node.value[:50]}...")
    
    await semantic_repo.close()
    print(f"\n🎉 Test complete!")

if __name__ == "__main__":
    asyncio.run(test_semantic_search_progression())