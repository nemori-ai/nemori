#!/usr/bin/env python3
"""Check episode data to understand data distribution"""

import asyncio
import sys
from pathlib import Path

# Add the nemori package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository

async def main():
    db_path = Path("evaluation/memos/evaluation/results/locomo/nemori-test3/storages/nemori_full_semantic.duckdb")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    # Setup episodic repository
    config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path)
    )
    
    episode_repo = DuckDBEpisodicMemoryRepository(config)
    await episode_repo.initialize()
    
    print("=== Episode Counts by Owner ===")
    
    # Check episodes for each character
    characters = ["Caroline", "Melanie", "Audrey", "James", "Joanna", "John", "Jon", "Tim"]
    
    total_episodes = 0
    for character in characters:
        try:
            result = await episode_repo.get_episodes_by_owner(character)
            episodes = result.episodes if hasattr(result, "episodes") else result
            episode_count = len(episodes)
            total_episodes += episode_count
            
            print(f"{character}: {episode_count} episodes")
            
            if episode_count > 0:
                # Show sample episode titles
                print(f"  Sample episodes for {character}:")
                for i, episode in enumerate(episodes[:3]):  # Show first 3 episodes
                    print(f"    {i+1}. {episode.title[:60]}...")
                print()
                
        except Exception as e:
            print(f"Error checking {character}: {e}")
    
    print(f"\nTotal episodes across all characters: {total_episodes}")
    
    # Close the repository
    await episode_repo.close()

if __name__ == "__main__":
    asyncio.run(main())