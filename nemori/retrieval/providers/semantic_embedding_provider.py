"""
Semantic Memory Embedding Provider for efficient vector similarity search.

This provider creates and manages embedding indices for semantic nodes,
similar to the episodic memory embedding provider but optimized for semantic knowledge.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken

from ...core.episode import Episode

from ...storage.repository import SemanticMemoryRepository
from ..retrieval_types import (
    IndexStats,
    RetrievalConfig,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStorageType,
    RetrievalStrategy,
)
from .base import RetrievalProvider
from openai import AsyncOpenAI

from ...core.data_types import SemanticNode
from ...storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository


class SemanticEmbeddingProvider: #(RetrievalProvider)
    """
    Embedding-based retrieval provider for semantic memory with persistent indexing.
    
    Features:
    - In-memory embedding index for fast similarity search
    - Persistent index storage to disk (JSON format)
    - Automatic index rebuilding from database
    - Batch embedding generation
    - Cosine similarity ranking
    """
    
    def __init__(self, 
                 semantic_storage: DuckDBSemanticMemoryRepository,
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 embed_model: str = "text-embedding-ada-002",
                 persistence_dir: Optional[Path] = None,
                 enable_persistence: bool = True):
        """
        Initialize semantic embedding provider.
        
        Args:
            semantic_storage: DuckDB semantic memory repository
            api_key: OpenAI API key for embedding generation
            base_url: OpenAI API base URL
            embed_model: Embedding model to use
            persistence_dir: Directory to store index files
            enable_persistence: Whether to enable index persistence to disk
        """
        self.semantic_storage = semantic_storage
        self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.embed_model = embed_model
        
        # Persistence settings
        self.persistence_enabled = enable_persistence
        self.persistence_dir = persistence_dir
        if self.persistence_enabled and self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory indices: owner_id -> index structure
        self.user_indices: Dict[str, Dict[str, Any]] = {}
        
    def _get_user_index(self, owner_id: str) -> Dict[str, Any]:
        """Get or create user index structure."""
        if owner_id not in self.user_indices:
            self.user_indices[owner_id] = {
                "semantic_nodes": [],           # List of semantic nodes
                "embeddings": [],               # List of embedding vectors
                "node_id_to_index": {},         # node_id -> list index mapping
                "last_updated": datetime.now(),
            }
        return self.user_indices[owner_id]
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def _build_searchable_text(self, node: SemanticNode) -> str:
        """Build searchable text from semantic node."""
        # Combine key, value, and context for comprehensive search
        parts = []
        if node.key:
            parts.append(node.key)
        if node.value:
            parts.append(node.value)
        if node.context:
            parts.append(node.context)
        
        return " ".join(parts)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    async def _rebuild_semantic_index(self, owner_id: str) -> None:
        """
        Completely rebuild semantic embedding index for a user.
        This function clears existing index and rebuilds from database.
        """
        index = self._get_user_index(owner_id)
        
        print(f"   - Clearing existing semantic index for {owner_id}...")
        index["semantic_nodes"].clear()
        index["embeddings"].clear()
        index["node_id_to_index"].clear()
        
        # Fetch all semantic nodes from storage
        print(f"   - Fetching semantic nodes from storage for {owner_id}...")
        semantic_nodes = await self.semantic_storage.get_all_semantic_nodes_for_owner(owner_id)
        
        if not semantic_nodes:
            print(f"   - No semantic nodes found for owner {owner_id}. Index will be empty.")
            index["last_updated"] = datetime.now()
            if self.persistence_enabled:
                self._save_index_to_disk(owner_id)
            return
        
        print(f"   - Found {len(semantic_nodes)} semantic nodes. Generating embeddings...")
        
        # Process each semantic node
        for i, node in enumerate(semantic_nodes):
            # Print progress
            if (i + 1) % 10 == 0 or i == len(semantic_nodes) - 1:
                print(f"     - Processing semantic node {i + 1}/{len(semantic_nodes)}...")
            
            # Generate embedding for the node
            searchable_text = self._build_searchable_text(node)
            embedding = await self._generate_embedding(searchable_text)
            
            if embedding:  # Only add if embedding generation succeeded
                # Get current index position
                new_index_position = len(index["semantic_nodes"])
                
                # Add to all data structures
                index["semantic_nodes"].append(node)
                index["embeddings"].append(embedding)
                index["node_id_to_index"][node.node_id] = new_index_position
        
        # Update timestamp and save to disk
        index["last_updated"] = datetime.now()
        if self.persistence_enabled:
            print(f"   - Saving semantic index to disk for {owner_id}...")
            self._save_index_to_disk(owner_id)
        
        print(f"✅ Finished rebuilding semantic index for {owner_id}. Total nodes: {len(index['semantic_nodes'])}")
    
    def _serialize_semantic_node(self, node: SemanticNode) -> Dict[str, Any]:
        """Serialize semantic node to dictionary for storage."""
        if hasattr(node, 'to_dict'):
            return node.to_dict()
        else:
            # Manual serialization if to_dict method is not available
            return {
                'node_id': node.node_id,
                'owner_id': node.owner_id,
                'key': node.key,
                'value': node.value,
                'context': node.context,
                'confidence': node.confidence,
                'version': getattr(node, 'version', 1),
                'evolution_history': getattr(node, 'evolution_history', []),
                'linked_episode_ids': getattr(node, 'linked_episode_ids', []),
                'discovery_episode_id': getattr(node, 'discovery_episode_id', ''),
                'discovery_method': getattr(node, 'discovery_method', 'differential_analysis'),
                'search_keywords': getattr(node, 'search_keywords', []),
                'embedding_vector': getattr(node, 'embedding_vector', None),
                'access_count': getattr(node, 'access_count', 0),
                'relevance_score': getattr(node, 'relevance_score', 0.0),
                'importance_score': getattr(node, 'importance_score', 0.0),
                'created_at': getattr(node, 'created_at', datetime.now()).isoformat(),
                'updated_at': getattr(node, 'updated_at', datetime.now()).isoformat(),
            }
    
    def _get_index_file_path(self, owner_id: str) -> Optional[Path]:
        """Get the file path for a user's semantic index."""
        if not self.persistence_enabled or not self.persistence_dir:
            return None
        return self.persistence_dir / f"semantic_embedding_index_{owner_id}.json"
    
    def _save_index_to_disk(self, owner_id: str) -> None:
        """Save user semantic index to disk."""
        if not self.persistence_enabled:
            return
        
        try:
            index = self.user_indices.get(owner_id)
            if not index:
                return
            
            index_file = self._get_index_file_path(owner_id)
            if not index_file:
                return
            
            # Prepare data for serialization
            serializable_data = {
                "semantic_nodes": [],
                "embeddings": index["embeddings"],
                "node_id_to_index": index["node_id_to_index"],
                "last_updated": index["last_updated"].isoformat(),
                "metadata": {
                    "total_nodes": len(index["semantic_nodes"]),
                    "embedding_dimension": len(index["embeddings"][0]) if index["embeddings"] else 0,
                },
            }
            
            # Serialize semantic nodes
            for node in index["semantic_nodes"]:
                node_data = self._serialize_semantic_node(node)
                serializable_data["semantic_nodes"].append(node_data)
            
            with open(index_file, "w", encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"Warning: Failed to save semantic index for {owner_id}: {e}")
    
    def _load_index_from_disk(self, owner_id: str) -> bool:
        """Load user semantic index from disk. Returns True if successful."""
        if not self.persistence_enabled:
            return False
        
        try:
            index_file = self._get_index_file_path(owner_id)
            if not index_file or not index_file.exists():
                return False
            
            with open(index_file, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            # Recreate the index structure
            index = self._get_user_index(owner_id)
            
            # Load semantic nodes (will be loaded from storage when needed)
            index["semantic_nodes"] = data["semantic_nodes"]
            index["embeddings"] = data["embeddings"]
            index["node_id_to_index"] = data["node_id_to_index"]
            index["last_updated"] = datetime.fromisoformat(data["last_updated"])
            
            return True
        
        except Exception as e:
            print(f"Warning: Failed to load semantic index for {owner_id}: {e}")
            return False
    
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Perform embedding-based similarity search on semantic nodes.
        
        Args:
            query: Retrieval query with owner_id, text, and limit
            
        Returns:
            RetrievalResult containing ranked semantic nodes
        """
        owner_id = query.owner_id
        query_text = query.text
        limit = min(query.limit, 100)  # Cap at 100 results
        
        # Ensure index exists and is loaded
        await self._ensure_index_ready(owner_id)
        
        index = self.user_indices.get(owner_id)
        if not index or not index["embeddings"]:
            return RetrievalResult(episodes=[], semantic_nodes=[], total_count=0)
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query_text)
            if not query_embedding:
                return RetrievalResult(episodes=[], semantic_nodes=[], total_count=0)
            
            # Calculate similarities
            similarities = []
            for i, node_embedding in enumerate(index["embeddings"]):
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and take top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:limit]
            
            # Get corresponding semantic nodes
            result_nodes = []
            for node_index, similarity in top_similarities:
                if node_index < len(index["semantic_nodes"]):
                    node = index["semantic_nodes"][node_index]
                    # Add similarity score as metadata
                    if hasattr(node, 'relevance_score'):
                        node.relevance_score = similarity
                    result_nodes.append(node)
            
            return RetrievalResult(
                episodes=[],
                semantic_nodes=result_nodes,
                total_count=len(result_nodes)
            )
        
        except Exception as e:
            print(f"Error during semantic embedding search: {e}")
            return RetrievalResult(episodes=[], semantic_nodes=[], total_count=0)
    
    async def _ensure_index_ready(self, owner_id: str) -> None:
        """Ensure the index for owner_id is ready for search."""
        # Try to load from disk first
        if owner_id not in self.user_indices:
            loaded = self._load_index_from_disk(owner_id)
            if not loaded:
                # If no saved index, rebuild from database
                await self._rebuild_semantic_index(owner_id)
        
        # Check if index needs updating (simple time-based check)
        index = self.user_indices.get(owner_id)
        if index:
            # If index is older than 1 hour, consider rebuilding
            # In production, you might want more sophisticated update detection
            time_diff = datetime.now() - index["last_updated"]
            if time_diff.total_seconds() > 3600:  # 1 hour
                print(f"Semantic index for {owner_id} is stale, rebuilding...")
                await self._rebuild_semantic_index(owner_id)
    
    async def add_semantic_node(self, node: SemanticNode) -> None:
        """Add a new semantic node to the index."""
        await self.add_semantic_nodes_batch([node])
    
    async def add_semantic_nodes_batch(self, nodes: List[SemanticNode]) -> None:
        """Add multiple semantic nodes to the index in batch."""
        if not nodes:
            return
        
        # Group nodes by owner
        nodes_by_owner: Dict[str, List[SemanticNode]] = {}
        for node in nodes:
            if node.owner_id not in nodes_by_owner:
                nodes_by_owner[node.owner_id] = []
            nodes_by_owner[node.owner_id].append(node)
        
        # Add to each user's index
        for owner_id, user_nodes in nodes_by_owner.items():
            await self._ensure_index_ready(owner_id)
            index = self._get_user_index(owner_id)
            
            for node in user_nodes:
                # Skip if node already exists in index
                if node.node_id in index["node_id_to_index"]:
                    continue
                
                # Build searchable text and generate embedding
                searchable_text = self._build_searchable_text(node)
                embedding = await self._generate_embedding(searchable_text)
                
                if embedding:
                    # Add to index
                    node_index = len(index["semantic_nodes"])
                    index["semantic_nodes"].append(node)
                    index["embeddings"].append(embedding)
                    index["node_id_to_index"][node.node_id] = node_index
            
            # Update timestamp and save
            index["last_updated"] = datetime.now()
            if self.persistence_enabled:
                self._save_index_to_disk(owner_id)
    
    async def remove_semantic_node(self, owner_id: str, node_id: str) -> None:
        """Remove a semantic node from the index."""
        await self._ensure_index_ready(owner_id)
        index = self._get_user_index(owner_id)
        
        if node_id not in index["node_id_to_index"]:
            return
        
        # This is complex because we need to maintain index consistency
        # For now, just rebuild the entire index (could be optimized)
        print(f"Removing semantic node {node_id} for {owner_id}, rebuilding index...")
        await self._rebuild_semantic_index(owner_id)
    
    async def initialize(self) -> None:
        """Initialize the semantic embedding provider."""
        print("✅ Semantic embedding provider initialized")
        
        # Pre-load indices for existing owners if needed
        if self.persistence_enabled and self.persistence_dir:
            # Find existing index files
            index_files = list(self.persistence_dir.glob("semantic_embedding_index_*.json"))
            print(f"Found {len(index_files)} existing semantic index files")
            
            for index_file in index_files:
                # Extract owner_id from filename
                filename = index_file.stem
                if filename.startswith("semantic_embedding_index_"):
                    owner_id = filename[26:]  # Remove "semantic_embedding_index_" prefix
                    self._load_index_from_disk(owner_id)
    
    async def close(self) -> None:
        """Close the semantic embedding provider and save all indices."""
        print("🧹 Closing semantic embedding provider...")
        
        if self.persistence_enabled:
            # Save all indices to disk
            for owner_id in self.user_indices:
                self._save_index_to_disk(owner_id)
        
        # Close OpenAI client
        if self.openai_client:
            await self.openai_client.close()
        
        print("✅ Semantic embedding provider closed")