#!/usr/bin/env python3
"""
Test script to debug semantic text search functionality
"""
import asyncio
import sys
from pathlib import Path

# Add the nemori package to the path
sys.path.insert(0, str(Path(__file__).parent))

from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig, SemanticNodeQuery, SortOrder

async def test_semantic_text_search():
    """Test semantic text search directly"""
    
    # Use the actual database path
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-test/storages/nemori_memory.duckdb"
    
    print(f"🔧 Initializing DuckDB semantic storage")
    
    # Create storage config
    config = StorageConfig(connection_string=db_path)
    
    # Initialize semantic storage
    semantic_storage = DuckDBSemanticMemoryRepository(config)
    await semantic_storage.initialize()
    
    print(f"✅ Storage initialized")
    
    # Test parameters
    owner_id = "张三_0"
    
    # Test different search terms
    test_queries = [
        "Discussion on Python Machine Learning Libraries scikit-learn pandas numpy TensorFlow",
        "scikit-learn",
        "numpy",
        "TensorFlow",
        "pandas",
        "machine learning",
        "Python",
        "learning"
    ]
    
    for test_query in test_queries:
        print(f"\n🧪 Testing text search with query: '{test_query}'")
        
        # Create semantic node query for text search
        search_query = SemanticNodeQuery(
            owner_id=owner_id, 
            text_search=test_query, 
            limit=5, 
            sort_by="confidence", 
            sort_order=SortOrder.DESC
        )
        
        try:
            result = await semantic_storage.search_semantic_nodes(search_query)
            print(f"   📊 Found {len(result.semantic_nodes)} nodes")
            
            for i, node in enumerate(result.semantic_nodes, 1):
                print(f"     Node {i}: {node.key} -> {node.value[:50]}...")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test direct database query to verify data
    print(f"\n🔍 Direct database verification:")
    print(f"   Checking all semantic nodes for owner '{owner_id}':")
    
    try:
        # Get all nodes for the owner
        all_nodes_query = SemanticNodeQuery(owner_id=owner_id, limit=10)
        all_result = await semantic_storage.search_semantic_nodes(all_nodes_query)
        
        print(f"   📊 Total nodes found: {len(all_result.semantic_nodes)}")
        for i, node in enumerate(all_result.semantic_nodes, 1):
            print(f"     Node {i}:")
            print(f"       Key: {node.key}")
            print(f"       Value: {node.value[:100]}...")
            print(f"       Context: {node.context[:100] if node.context else 'None'}...")
            print()
            
    except Exception as e:
        print(f"   ❌ Error getting all nodes: {e}")
        import traceback
        traceback.print_exc()
    
    await semantic_storage.close()
    print("🔚 Test completed")

if __name__ == "__main__":
    asyncio.run(test_semantic_text_search())