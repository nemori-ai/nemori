#!/usr/bin/env python3
"""
Quick test for user 2 with fresh search
"""

import asyncio
from locomo_search import process_user_nemori
import pandas as pd

async def test_user_2_fresh():
    locomo_df = pd.read_json('/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json')
    
    print("🧪 Testing user 2 with fresh search...")
    
    # Test user 2 processing
    results = await process_user_nemori(2, locomo_df, 'nemori', 'episode_semantic', top_k=5)
    
    # Check results
    for conv_id, search_results in results.items():
        print(f'\n📝 Conversation ID: {conv_id}')
        print(f'   📊 Number of results: {len(search_results)}')
        
        for i, result in enumerate(search_results):
            print(f'\n   🔍 Query {i+1}: {result["query"]}')
            
            if "No relevant" in result["context"]:
                print(f'   ❌ No content found')
            else:
                print(f'   ✅ Content found!')
                context_preview = result["context"][:300]
                print(f'   📄 Context preview: {context_preview}...')

if __name__ == "__main__":
    asyncio.run(test_user_2_fresh())