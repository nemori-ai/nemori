#!/usr/bin/env python3
"""
More thorough database structure check.
"""

import duckdb
import pandas as pd

def check_database_structure():
    """Check database tables and structure."""
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    print(f"Checking database: {db_path}")
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables: {[table[0] for table in tables]}")
        
        # Check episodes table structure
        if any('episodes' in str(table[0]) for table in tables):
            episodes_schema = conn.execute("DESCRIBE episodes").fetchall()
            print(f"\nEpisodes table schema:")
            for col in episodes_schema:
                print(f"  {col[0]}: {col[1]}")
            
            # Check episodes count and sample
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            print(f"\nEpisodes count: {count}")
            
            if count > 0:
                # Check columns that might contain owner info
                sample = conn.execute("SELECT * FROM episodes LIMIT 1").fetchall()
                print(f"Sample episode columns: {len(sample[0]) if sample else 0}")
                
                # Try different potential owner column names
                potential_owner_cols = ['owner_id', 'user_id', 'speaker_id', 'participant_id']
                for col in potential_owner_cols:
                    try:
                        result = conn.execute(f"SELECT DISTINCT {col} FROM episodes LIMIT 10").fetchall()
                        print(f"  {col}: {[r[0] for r in result]}")
                    except:
                        print(f"  {col}: column not found")
        
        # Check semantic_nodes table structure
        if any('semantic_nodes' in str(table[0]) for table in tables):
            semantic_schema = conn.execute("DESCRIBE semantic_nodes").fetchall()
            print(f"\nSemantic_nodes table schema:")
            for col in semantic_schema:
                print(f"  {col[0]}: {col[1]}")
            
            # Check semantic_nodes count
            count = conn.execute("SELECT COUNT(*) FROM semantic_nodes").fetchone()[0]
            print(f"\nSemantic_nodes count: {count}")
            
            if count > 0:
                # Try different potential owner column names
                potential_owner_cols = ['owner_id', 'user_id', 'speaker_id', 'participant_id']
                for col in potential_owner_cols:
                    try:
                        result = conn.execute(f"SELECT DISTINCT {col} FROM semantic_nodes LIMIT 10").fetchall()
                        print(f"  {col}: {[r[0] for r in result]}")
                    except:
                        print(f"  {col}: column not found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def check_user_3_specifically():
    """Check locomo data for user 3."""
    data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    locomo_df = pd.read_json(data_path)
    
    print(f"\n=== User 3 Details ===")
    conversation = locomo_df["conversation"].iloc[3]
    qa_set = locomo_df["qa"].iloc[3]
    
    print(f"Speakers: {conversation.get('speaker_a')} & {conversation.get('speaker_b')}")
    print(f"QA count: {len(qa_set)}")
    print(f"First 3 questions:")
    for i, qa in enumerate(qa_set[:3]):
        print(f"  {i+1}. {qa.get('question')}")

if __name__ == "__main__":
    check_database_structure()
    check_user_3_specifically()