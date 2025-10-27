#!/usr/bin/env python3
"""
Debug script to check database content directly.
"""

import asyncio
from pathlib import Path

from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.storage.storage_types import StorageConfig


async def check_database_content():
    """Check what's actually in the database."""
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Checking database: {db_path}")
    
    # Check episodic memories
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=db_path,
        batch_size=100,
        cache_size=1000,
    )
    
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()
    
    # Count total episodes
    try:
        # Get connection and execute raw SQL
        import duckdb
        conn = duckdb.connect(db_path)
        
        # Check tables
        tables_result = conn.execute("SHOW TABLES").fetchall()
        print(f"\n📊 Tables in database: {[table[0] for table in tables_result]}")
        
        # Check episodes
        episodes_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        print(f"📖 Total episodes: {episodes_count}")
        
        if episodes_count > 0:
            # Show sample episodes
            sample_episodes = conn.execute("""
                SELECT user_id, title, content, summary 
                FROM episodes 
                LIMIT 5
            """).fetchall()
            
            print(f"\n📋 Sample episodes:")
            for i, (user_id, title, content, summary) in enumerate(sample_episodes):
                print(f"  {i+1}. User: {user_id}")
                print(f"     Title: {title}")
                print(f"     Content: {content[:100]}...")
                print(f"     Summary: {summary[:100]}...")
                print()
        
        # Check unique users
        users_result = conn.execute("SELECT DISTINCT user_id FROM episodes").fetchall()
        print(f"👥 Unique users: {[user[0] for user in users_result]}")
        
        # Check semantic nodes
        try:
            semantic_count = conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
            print(f"🧠 Total semantic nodes: {semantic_count}")
            
            if semantic_count > 0:
                # Show sample semantic nodes
                sample_semantic = conn.execute("""
                    SELECT owner_id, key, value, confidence 
                    FROM semantic_nodes 
                    LIMIT 5
                """).fetchall()
                
                print(f"\n🔍 Sample semantic nodes:")
                for i, (owner_id, key, value, confidence) in enumerate(sample_semantic):
                    print(f"  {i+1}. Owner: {owner_id}")
                    print(f"     Key: {key}")
                    print(f"     Value: {value[:100]}...")
                    print(f"     Confidence: {confidence}")
                    print()
                
                # Check unique semantic owners
                semantic_users = conn.execute("SELECT DISTINCT owner_id FROM semantic_nodes").fetchall()
                print(f"🧠 Semantic owners: {[user[0] for user in semantic_users]}")
                
        except Exception as e:
            print(f"⚠️ No semantic_nodes table or error: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await episode_repo.close()


if __name__ == "__main__":
    asyncio.run(check_database_content())