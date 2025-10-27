#!/usr/bin/env python3
"""
简单的索引构建测试脚本
"""
import asyncio
from pathlib import Path
import os

# Nemori imports
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
from nemori.retrieval.providers.semantic_embedding_provider import SemanticEmbeddingProvider


async def build_indices_for_version(version: str):
    """为特定版本构建索引"""
    print(f"\n🚀 Building indices for version: {version}")
    print("=" * 60)
    
    # Setup storage
    storage_dir = Path(f"results/locomo/nemori-{version}/storages")
    db_path = storage_dir / "nemori_full_semantic.duckdb"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
        
    print(f"📁 Database path: {db_path}")
    
    # Storage config for both episodic and semantic repositories
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=True,
    )
    
    # Semantic storage config with embedding support
    semantic_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
    )
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
    
    # Traditional retrieval service for backwards compatibility
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
        api_key="EMPTY",
        base_url="http://localhost:6007/v1",
        embed_model="qwen3-emb",
    )
    retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)
    await retrieval_service.initialize()
    
    print("✅ All components initialized")
    
    # Get all owner IDs from the database
    try:
        # Get episodic owners
        episodic_owners = await episode_repo._execute_query(
            "SELECT DISTINCT owner_id FROM episodes"
        )
        episodic_owner_ids = [row[0] for row in episodic_owners]
        
        # Get semantic owners  
        semantic_owners = await semantic_repo._execute_query(
            "SELECT DISTINCT owner_id FROM semantic_nodes"
        )
        semantic_owner_ids = [row[0] for row in semantic_owners]
        
        all_owner_ids = list(set(episodic_owner_ids + semantic_owner_ids))
        print(f"👥 Found owners: {all_owner_ids}")
        
    except Exception as e:
        print(f"⚠️ Error getting owner IDs: {e}")
        all_owner_ids = ["Caroline", "Melanie"]  # fallback
    
    # Build episodic indices for each owner
    print(f"\n📖 Building Episodic Memory Indices")
    all_episodes_count = 0
    
    for owner_id in all_owner_ids:
        try:
            # Trigger index building with a simple search
            query = RetrievalQuery(
                text="test", 
                owner_id=owner_id, 
                limit=1, 
                strategy=RetrievalStrategy.BM25
            )
            result = await retrieval_service.search(query)
            episode_count = len(result.episodes)
            all_episodes_count += episode_count
            print(f"   👤 {owner_id}: {episode_count} episodes indexed")
        except Exception as e:
            print(f"   ⚠️ Error indexing episodes for {owner_id}: {e}")
    
    # Build semantic memory indices
    print(f"\n🧠 Building Semantic Memory Indices") 
    all_semantic_nodes_count = 0
    semantic_owners_with_nodes = []
    
    for owner_id in all_owner_ids:
        try:
            # Initialize semantic embedding provider for efficient indexing
            semantic_embedding_provider = SemanticEmbeddingProvider(
                semantic_storage=semantic_repo,
                api_key="EMPTY",
                base_url="http://localhost:6007/v1", 
                embed_model="qwen3-emb",
                persistence_dir=storage_dir,
                enable_persistence=True
            )
            
            # Build semantic index
            await semantic_embedding_provider._rebuild_semantic_index(owner_id)
            
            # Count semantic nodes
            nodes = await semantic_repo.search_semantic_nodes_by_owner(owner_id, limit=1000)
            node_count = len(nodes.semantic_nodes)
            
            if node_count > 0:
                all_semantic_nodes_count += node_count
                semantic_owners_with_nodes.append(owner_id)
                print(f"   👤 {owner_id}: {node_count} semantic nodes indexed")
            else:
                print(f"   👤 {owner_id}: No semantic nodes found")
                
        except Exception as e:
            print(f"   ⚠️ Semantic index building failed for {owner_id}: {e}")
    
    print(f"\n📊 Index Building Summary:")
    print(f"   📖 Episodic Memory: {all_episodes_count} episodes indexed")
    print(f"   🧠 Semantic Memory: {all_semantic_nodes_count} nodes indexed")
    print(f"   👥 Owners with semantic knowledge: {len(semantic_owners_with_nodes)}/{len(all_owner_ids)}")
    print(f"   🎯 Both episodic and semantic indices ready for unified retrieval")
    
    # Check for index files
    print(f"\n📁 Checking for index files in {storage_dir}")
    for file in storage_dir.glob("*.json"):
        print(f"   📄 {file.name}: {file.stat().st_size} bytes")
    
    # Cleanup
    await episode_repo.close()
    await semantic_repo.close()
    if retrieval_service:
        await retrieval_service.close()
    print("✅ Cleanup completed")


async def main():
    """主函数"""
    versions = ["test_sample_1"]  # 可以添加更多版本
    
    for version in versions:
        await build_indices_for_version(version)


if __name__ == "__main__":
    asyncio.run(main())