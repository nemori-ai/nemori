#!/usr/bin/env python3
"""
Test script to debug semantic memory search functionality
"""
import asyncio
import sys
from pathlib import Path

# Add the nemori package to the path
sys.path.insert(0, str(Path(__file__).parent))

from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig

async def test_semantic_search():
    """Test semantic memory search with the actual database"""
    
    # Use the actual database path
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-test/storages/nemori_memory.duckdb"
    
    print(f"🔧 Initializing DuckDB semantic storage with database: {db_path}")
    
    # Create storage config
    config = StorageConfig(connection_string=db_path)
    
    # Initialize semantic storage
    semantic_storage = DuckDBSemanticMemoryRepository(config)
    await semantic_storage.initialize()
    
    print(f"✅ Storage initialized successfully")
    
    # Test parameters from the actual discovery process
    owner_id = "张三_0"  # From database analysis
    query = "Discussion on Python Machine Learning Libraries scikit-learn pandas numpy TensorFlow"  # Similar to episode title/content
    limit = 5
    
    print(f"\n🧪 Testing semantic search:")
    print(f"   Owner ID: {owner_id}")
    print(f"   Query: {query}")
    print(f"   Limit: {limit}")
    print()
    
    try:
        # This should trigger all our debug logging
        results = await semantic_storage.similarity_search_semantic_nodes(
            owner_id=owner_id,
            query=query, 
            limit=limit
        )
        
        print(f"\n📊 Search Results:")
        print(f"   Found {len(results)} semantic nodes")
        
        for i, node in enumerate(results, 1):
            print(f"   Node {i}:")
            print(f"     - Key: {node.key}")
            print(f"     - Value: {node.value[:100]}...")
            print(f"     - Confidence: {node.confidence}")
            print()
            
    except Exception as e:
        print(f"❌ Error during semantic search: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await semantic_storage.close()
        print("🔚 Storage connection closed")

if __name__ == "__main__":
    asyncio.run(test_semantic_search())