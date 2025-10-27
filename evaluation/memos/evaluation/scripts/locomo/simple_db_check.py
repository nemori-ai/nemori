#!/usr/bin/env python3
"""
Simple database check script.
"""

import duckdb
from pathlib import Path


def check_database():
    """Check database content directly with DuckDB."""
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Checking database: {db_path}")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Check tables
        tables_result = conn.execute("SHOW TABLES").fetchall()
        print(f"\n📊 Tables: {[table[0] for table in tables_result]}")
        
        # Check episodes
        episodes_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        print(f"📖 Episodes: {episodes_count}")
        
        if episodes_count > 0:
            # Show unique users
            users = conn.execute("SELECT DISTINCT owner_id FROM episodes").fetchall()
            print(f"👥 Episode users: {[user[0] for user in users]}")
            
            # Show sample episodes
            samples = conn.execute("""
                SELECT owner_id, title, LEFT(content, 50) 
                FROM episodes 
                LIMIT 3
            """).fetchall()
            
            print(f"\n📋 Sample episodes:")
            for owner_id, title, content in samples:
                print(f"  • {owner_id}: {title} - {content}...")
        
        # Check semantic nodes
        try:
            semantic_count = conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
            print(f"🧠 Semantic nodes: {semantic_count}")
            
            if semantic_count > 0:
                semantic_users = conn.execute("SELECT DISTINCT owner_id FROM semantic_nodes").fetchall()
                print(f"🧠 Semantic owners: {[user[0] for user in semantic_users]}")
                
                # Show sample semantic nodes
                semantic_samples = conn.execute("""
                    SELECT owner_id, key, LEFT(value, 50), confidence 
                    FROM semantic_nodes 
                    LIMIT 3
                """).fetchall()
                
                print(f"\n🔍 Sample semantic nodes:")
                for owner_id, key, value, confidence in semantic_samples:
                    print(f"  • {owner_id}: {key} = {value}... (conf: {confidence})")
                    
        except Exception as e:
            print(f"⚠️ Semantic check failed: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")


if __name__ == "__main__":
    check_database()