#!/usr/bin/env python3
"""
Direct test with correct database IDs for user 3.
"""

import asyncio
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
from nemori.retrieval import RetrievalStrategy

async def test_with_correct_ids():
    """Test search functionality with correct database user IDs."""
    
    print(f"🗣️ Testing user 3 with correct database IDs")
    print(f"🆔 Using: joanna_3 & nate_3")
    
    # Get unified client with fixed database path
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await get_nemori_unified_client(
        user_id="test_user_3",
        version="episode_semantic",  # This will now use the correct database
        retrievalstrategy=RetrievalStrategy.EMBEDDING,
        emb_api_key="EMPTY",
        emb_base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb"
    )
    
    # Test questions for Joanna & Nate
    test_questions = [
        "Is it likely that Nate has friends besides Joanna?",
        "What kind of interests do Joanna and Nate share?", 
        "When did Nate win his first tournament?"
    ]
    
    for i, query in enumerate(test_questions):
        print(f"\n💬 Question {i+1}: {query}")
        
        try:
            context, duration = await nemori_unified_search(
                unified_retrieval_service=unified_retrieval,
                retrieval_service=retrieval_service,
                query=query,
                speaker_a_user_id="joanna_3",  # Use correct database ID
                speaker_b_user_id="nate_3",    # Use correct database ID
                top_k=3
            )
            
            print(f"✅ Search completed in {duration:.2f}ms")
            
            # Check if we actually found content
            if "No relevant" in context or len(context.strip()) < 100:
                print(f"⚠️ Limited content found")
                print(f"📄 Context: {context[:200]}...")
            else:
                print(f"✅ Rich content found!")
                print(f"📄 Context preview: {context[:400]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # Cleanup
    if retrieval_service:
        await retrieval_service.close()
    if episode_repo:
        await episode_repo.close()
    if semantic_repo:
        await semantic_repo.close()


if __name__ == "__main__":
    asyncio.run(test_with_correct_ids())