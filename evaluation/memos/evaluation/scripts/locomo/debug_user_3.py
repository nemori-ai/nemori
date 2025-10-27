#!/usr/bin/env python3
"""
Debug script to check user 3 (Joanna & Nate) data and search issues.
"""

import asyncio
import pandas as pd
from pathlib import Path

def check_user_mapping_for_user_3():
    """Check user mapping specifically for user 3."""
    import duckdb
    
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    # Load locomo data to see what user 3 should be
    data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    locomo_df = pd.read_json(data_path)
    
    conversation = locomo_df["conversation"].iloc[3]  # User 3
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    
    print(f"User 3 conversation:")
    print(f"  Speaker A: {speaker_a}")
    print(f"  Speaker B: {speaker_b}")
    
    # Check database content
    conn = duckdb.connect(db_path, read_only=True)
    
    # Get all users in database
    episode_users = conn.execute("SELECT DISTINCT owner_id FROM episodes").fetchall()
    episode_users = [user[0] for user in episode_users]
    
    semantic_users = conn.execute("SELECT DISTINCT owner_id FROM semantic_nodes").fetchall()
    semantic_users = [user[0] for user in semantic_users]
    
    print(f"\nDatabase users:")
    print(f"  Episode users: {sorted(episode_users)}")
    print(f"  Semantic users: {sorted(semantic_users)}")
    
    # Look for potential matches
    potential_matches = []
    for user in episode_users:
        if speaker_a.lower() in user.lower() or speaker_b.lower() in user.lower():
            potential_matches.append(user)
    
    print(f"\nPotential matches for {speaker_a} & {speaker_b}: {potential_matches}")
    
    # Check specific content for potential matches
    for user_id in potential_matches:
        episode_count = conn.execute("SELECT COUNT(*) FROM episodes WHERE owner_id = ?", [user_id]).fetchone()[0]
        semantic_count = conn.execute("SELECT COUNT(*) FROM semantic_nodes WHERE owner_id = ?", [user_id]).fetchone()[0]
        
        print(f"\nUser {user_id}:")
        print(f"  Episodes: {episode_count}")
        print(f"  Semantic nodes: {semantic_count}")
        
        if episode_count > 0:
            sample_episodes = conn.execute("""
                SELECT title, LEFT(content, 100) 
                FROM episodes 
                WHERE owner_id = ? 
                LIMIT 2
            """, [user_id]).fetchall()
            
            print(f"  Sample episodes:")
            for i, (title, content) in enumerate(sample_episodes):
                print(f"    {i+1}. {title}")
                print(f"       {content}...")
    
    conn.close()


if __name__ == "__main__":
    check_user_mapping_for_user_3()