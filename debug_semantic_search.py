#!/usr/bin/env python3
"""
Debug script to test semantic memory search functionality
"""

import asyncio
from pathlib import Path
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig

async def debug_semantic_search():
    """Debug semantic memory search functionality"""
    print("🔍 Starting Semantic Memory Search Debug")
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
    
    print(f"📄 Database path: {db_path}")
    print(f"🔑 API key: {config.openai_api_key}")
    print(f"🌐 Base URL: {config.openai_base_url}")
    print(f"🤖 Model: {config.openai_embed_model}")
    
    # Initialize repository
    semantic_repo = DuckDBSemanticMemoryRepository(config)
    await semantic_repo.initialize()
    
    print(f"✅ Repository initialized")
    print(f"🔧 OpenAI client: {semantic_repo.openai_client is not None}")
    print(f"📋 Embed model: {semantic_repo.embed_model}")
    
    # Check for existing owners
    from sqlmodel import Session, select
    from nemori.storage.sql_models import SemanticNodeTable
    
    with Session(semantic_repo.engine) as session:
        owners = session.exec(
            select(SemanticNodeTable.owner_id).distinct()
        ).all()
        
        print(f"\n👥 Found owners: {owners}")
        
        for owner_id in owners:
            node_count = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.owner_id == owner_id)
            ).all()
            print(f"   🧠 {owner_id}: {len(node_count)} semantic nodes")
            
            # Show sample nodes
            for i, node in enumerate(node_count[:2]):
                print(f"      📝 Node {i+1}: {node.key} = {node.value[:50]}...")
    
    # Test search for each owner
    test_queries = ["机器学习", "Python", "PyTorch", "AI", "编程"]
    
    for owner_id in owners[:2]:  # Test first 2 owners
        print(f"\n🔍 Testing semantic search for owner: {owner_id}")
        
        for query in test_queries:
            print(f"  📋 Query: '{query}'")
            try:
                results = await semantic_repo.similarity_search_semantic_nodes(owner_id, query, limit=3)
                if results:
                    print(f"    ✅ Found {len(results)} results:")
                    for i, node in enumerate(results):
                        print(f"      {i+1}. {node.key}: {node.value[:60]}...")
                else:
                    print(f"    ❌ No results found")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Test embedding generation directly
    print(f"\n🧪 Testing direct embedding generation...")
    try:
        test_embedding = await semantic_repo._generate_query_embedding("测试查询")
        if test_embedding:
            print(f"    ✅ Embedding generated successfully: {len(test_embedding)} dimensions")
        else:
            print(f"    ❌ Embedding generation failed")
    except Exception as e:
        print(f"    ❌ Embedding error: {e}")
    
    await semantic_repo.close()
    print(f"\n🎉 Debug complete!")

if __name__ == "__main__":
    asyncio.run(debug_semantic_search())