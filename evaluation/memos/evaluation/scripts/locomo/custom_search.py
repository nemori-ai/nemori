#!/usr/bin/env python3
"""
Custom search script for locomo evaluation using specific database path.
Searches both episodic and semantic memories using embedding retrieval.
"""

import asyncio
import json
from pathlib import Path
from time import time

import pandas as pd

from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.storage.storage_types import StorageConfig


# Template for unified memory search results
TEMPLATE_UNIFIED = """Unified memories for conversation between {speaker_1} and {speaker_2}:

Episodic Memories:
{episodic_memories}

Semantic Knowledge:
{semantic_knowledge}
"""


async def create_unified_client(db_path: str):
    """Create unified client for episodic and semantic search."""
    # Storage config for both episodic and semantic repositories
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=db_path,
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=True,
    )
    
    # Semantic storage config with embedding support
    semantic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=db_path,
        batch_size=100,
        cache_size=1000,
    )
    # Configure embedding settings for local server
    semantic_config.openai_api_key = "EMPTY"
    semantic_config.openai_base_url = "http://localhost:6007/v1"
    semantic_config.openai_embed_model = "qwen3-emb"

    # Initialize repositories
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    semantic_repo = DuckDBSemanticMemoryRepository(semantic_config)
    
    await episode_repo.initialize()
    await semantic_repo.initialize()

    # Setup unified retrieval service
    unified_retrieval = UnifiedRetrievalService(episode_repo, semantic_repo)
    
    # Traditional retrieval service for fallback
    retrieval_service = RetrievalService(episode_repo)
    storage_dir = Path(db_path).parent
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key="EMPTY",
        base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb",
    )
    retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
    await retrieval_service.initialize()

    return unified_retrieval, retrieval_service, episode_repo, semantic_repo


async def unified_search(
    unified_retrieval_service,
    retrieval_service,
    query: str,
    speaker_a_user_id: str,
    speaker_b_user_id: str,
    top_k: int = 10,
) -> tuple[str, float]:
    """Search using unified memory (episodic + semantic)."""
    start = time()

    print(f"\n🔍 [UNIFIED SEARCH] Query: '{query}'")
    print(f"   👤 Speaker A: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")

    try:
        # Enhanced query combines episodic and semantic retrieval
        unified_results = await unified_retrieval_service.enhanced_query(
            owner_id=speaker_a_user_id,
            query=query,
            include_semantic=True,
            episode_limit=top_k,
            semantic_limit=top_k*2
        )
        
        # Fallback: If unified retrieval returns no episodic memories, use direct embedding search
        if len(unified_results.get('episodes', [])) == 0 and retrieval_service:
            try:
                query_obj = RetrievalQuery(
                    text=query, 
                    owner_id=speaker_a_user_id, 
                    limit=top_k, 
                    strategy=RetrievalStrategy.EMBEDDING
                )
                episodic_result = await retrieval_service.search(query_obj)
                
                if hasattr(episodic_result, 'episodes') and len(episodic_result.episodes) > 0:
                    unified_results['episodes'] = episodic_result.episodes
                    print(f"   🔄 Used fallback embedding search for episodes")
            except Exception as e:
                print(f"   ❌ Fallback embedding search failed: {e}")
        
        print(f"   ✅ Search completed")
        print(f"   📖 Found {len(unified_results.get('episodes', []))} episodic memories")
        print(f"   🧠 Found {len(unified_results.get('semantic_knowledge', []))} semantic concepts")
        
        # Format episodic memories
        episodic_memories = []
        for episode in unified_results.get('episodes', []):
            if hasattr(episode, 'title') and hasattr(episode, 'content'):
                memory_text = f"• {episode.title}: {episode.content}"
                episodic_memories.append(memory_text)
            else:
                # Handle dictionary format
                title = episode.get("title", "")
                content = episode.get("content", "")
                memory_text = f"• {title}: {content}"
                episodic_memories.append(memory_text)
        
        # Format semantic knowledge
        semantic_knowledge = []
        for semantic_node in unified_results.get('semantic_knowledge', []):
            if hasattr(semantic_node, 'key') and hasattr(semantic_node, 'value'):
                confidence = getattr(semantic_node, 'confidence', 0.0)
                knowledge_text = f"• {semantic_node.key}: {semantic_node.value} (confidence: {confidence:.2f})"
                semantic_knowledge.append(knowledge_text)
        
        # Generate unified context
        context = TEMPLATE_UNIFIED.format(
            speaker_1=speaker_a_user_id,
            speaker_2=speaker_b_user_id,
            episodic_memories="\n".join(episodic_memories) if episodic_memories else "No relevant episodic memories found",
            semantic_knowledge="\n".join(semantic_knowledge) if semantic_knowledge else "No relevant semantic knowledge found",
        )
        
    except Exception as e:
        print(f"   ❌ Unified search failed: {e}")
        context = f"Search failed: {e}"

    duration_ms = (time() - start) * 1000
    print(f"   ⏱️ Duration: {duration_ms:.2f}ms")

    return context, duration_ms


async def main():
    """Main function to test search with locomo questions."""
    # Database path
    db_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/results/locomo/nemori-episode_semantic/storages/nemori_memory.duckdb"
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"✅ Database found: {db_path}")
    
    # Load locomo questions
    data_path = "/data/jcxy/haolu/workspace/frameworks/nemori-semantic/nemori/evaluation/memos/evaluation/data/locomo/locomo10.json"
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    print(f"✅ Loading questions from: {data_path}")
    locomo_df = pd.read_json(data_path)
    print(f"📊 Loaded {len(locomo_df)} conversations")
    
    # Create unified client
    print("\n🚀 Initializing unified search client...")
    unified_retrieval, retrieval_service, episode_repo, semantic_repo = await create_unified_client(db_path)
    
    try:
        # Test with first few conversations
        num_test_conversations = 3
        for conv_idx in range(min(num_test_conversations, len(locomo_df))):
            conversation = locomo_df["conversation"].iloc[conv_idx]
            qa_set = locomo_df["qa"].iloc[conv_idx]
            
            speaker_a = conversation.get("speaker_a", "")
            speaker_b = conversation.get("speaker_b", "")
            
            print(f"\n{'='*60}")
            print(f"🗣️ Conversation {conv_idx}: {speaker_a} & {speaker_b}")
            print(f"❓ Questions: {len(qa_set)}")
            
            # Test with first 2 questions from this conversation
            for qa_idx, qa in enumerate(qa_set[:2]):
                if qa.get("category") == 5:  # Skip category 5 questions
                    continue
                    
                query = qa.get("question", "")
                if not query:
                    continue
                
                print(f"\n💬 Question {qa_idx + 1}: {query}")
                
                # Perform search
                context, duration = await unified_search(
                    unified_retrieval_service=unified_retrieval,
                    retrieval_service=retrieval_service,
                    query=query,
                    speaker_a_user_id=speaker_a,
                    speaker_b_user_id=speaker_b,
                    top_k=5
                )
                
                print(f"\n📄 Context Preview:")
                print(context[:500] + "..." if len(context) > 500 else context)
                
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if retrieval_service:
            await retrieval_service.close()
        if episode_repo:
            await episode_repo.close()
        if semantic_repo:
            await semantic_repo.close()
        
        print("✅ Search completed!")


if __name__ == "__main__":
    asyncio.run(main())