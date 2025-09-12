#!/usr/bin/env python3
"""Check semantic data in the DuckDB database to understand why only some characters have semantic knowledge"""

import asyncio
import sys
from pathlib import Path

# Add the nemori package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository

async def main():
    db_path = Path("evaluation/memos/evaluation/results/locomo/nemori-test3/storages/nemori_full_semantic.duckdb")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    # Setup semantic repository
    config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path)
    )
    
    semantic_repo = DuckDBSemanticMemoryRepository(config)
    await semantic_repo.initialize()
    
    print("=== Semantic Node Counts by Owner ===")
    
    # Check semantic nodes for each character
    characters = ["Caroline", "Melanie", "Audrey", "James", "Joanna", "John", "Jon", "Tim"]
    
    total_nodes = 0
    for character in characters:
        try:
            nodes = await semantic_repo.get_all_semantic_nodes_for_owner(character)
            node_count = len(nodes)
            total_nodes += node_count
            
            print(f"{character}: {node_count} semantic nodes")
            
            if node_count > 0:
                # Show sample nodes
                print(f"  Sample nodes for {character}:")
                for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
                    print(f"    {i+1}. {node.key}: {node.value[:50]}...")
                print()
                
        except Exception as e:
            print(f"Error checking {character}: {e}")
    
    print(f"\nTotal semantic nodes across all characters: {total_nodes}")
    
    # Close the repository
    await semantic_repo.close()

if __name__ == "__main__":
    asyncio.run(main())