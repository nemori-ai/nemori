"""
FAISS-based vector search engine
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from ..models import Episode, SemanticMemory
from ..utils import EmbeddingClient

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, falling back to numpy-based similarity search")

logger = logging.getLogger(__name__)


class VectorSearch:
    """FAISS-based vector search engine with incremental updates"""
    
    def __init__(self, embedding_client: EmbeddingClient, storage_path: str = "./memories", dimension: int = 1536):
        """
        Initialize vector search engine
        
        Args:
            embedding_client: Client for generating embeddings
            storage_path: Base storage path for persistent indices
            dimension: Embedding dimension
        """
        self.embedding_client = embedding_client
        self.dimension = dimension
        self.storage_path = storage_path
        
        # Episode indices and data
        self.episode_indices: Dict[str, Any] = {}  # user_id -> FAISS index
        self.episode_data: Dict[str, List[Episode]] = {}  # user_id -> [episodes]
        self.episode_embeddings: Dict[str, np.ndarray] = {}  # user_id -> embeddings
        
        # Semantic memory indices and data
        self.semantic_indices: Dict[str, Any] = {}  # user_id -> FAISS index
        self.semantic_data: Dict[str, List[SemanticMemory]] = {}  # user_id -> [memories]
        self.semantic_embeddings: Dict[str, np.ndarray] = {}  # user_id -> embeddings
        
        # Vector database directories
        self.episode_vector_dir = os.path.join(storage_path, "episodes", "vector_db")
        self.semantic_vector_dir = os.path.join(storage_path, "semantic", "vector_db")
        
        # Create directories
        os.makedirs(self.episode_vector_dir, exist_ok=True)
        os.makedirs(self.semantic_vector_dir, exist_ok=True)
        
        logger.info(f"Vector search engine initialized (FAISS available: {FAISS_AVAILABLE})")
        logger.info(f"Episode vector DB path: {self.episode_vector_dir}")
        logger.info(f"Semantic vector DB path: {self.semantic_vector_dir}")
    
    def load_user_indices(self, user_id: str, episodes: List[Episode], memories: List[SemanticMemory]):
        """
        Load existing indices for a user or create new ones if not found
        
        Args:
            user_id: User ID
            episodes: List of episodes (for fallback creation)
            memories: List of memories (for fallback creation)
        """
        try:
            # Try to load episode index
            episode_index = self._load_faiss_index(user_id, "episode")
            episode_embeddings = self._load_embeddings(user_id, "episode")
            
            if episode_index is not None and episode_embeddings is not None and episodes:
                if len(episodes) == len(episode_embeddings):
                    self.episode_indices[user_id] = episode_index
                    self.episode_data[user_id] = episodes
                    self.episode_embeddings[user_id] = episode_embeddings
                    logger.info(f"Loaded episode index for user {user_id} from disk")
                else:
                    logger.warning(f"Episode data length mismatch for user {user_id}, rebuilding index")
                    self.index_episodes(user_id, episodes)
            elif episodes:
                # Create new index if episodes exist
                self.index_episodes(user_id, episodes)
            
            # Try to load semantic index
            semantic_index = self._load_faiss_index(user_id, "semantic")
            semantic_embeddings = self._load_embeddings(user_id, "semantic")
            
            if semantic_index is not None and semantic_embeddings is not None and memories:
                if len(memories) == len(semantic_embeddings):
                    self.semantic_indices[user_id] = semantic_index
                    self.semantic_data[user_id] = memories
                    self.semantic_embeddings[user_id] = semantic_embeddings
                    logger.info(f"Loaded semantic index for user {user_id} from disk")
                else:
                    logger.warning(f"Semantic data length mismatch for user {user_id}, rebuilding index")
                    self.index_semantic_memories(user_id, memories)
            elif memories:
                # Create new index if memories exist
                self.index_semantic_memories(user_id, memories)
                
        except Exception as e:
            logger.error(f"Error loading indices for user {user_id}: {e}")
            # Fallback to creating new indices
            if episodes:
                self.index_episodes(user_id, episodes)
            if memories:
                self.index_semantic_memories(user_id, memories)

    def _create_faiss_index(self, embeddings: np.ndarray) -> Any:
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Embeddings array
            
        Returns:
            FAISS index or numpy array for fallback
        """
        try:
            if FAISS_AVAILABLE:
                # Always create FAISS index when FAISS is available
                # Even for empty embeddings, we create an empty index
                index = faiss.IndexFlatIP(self.dimension)
                
                if len(embeddings) > 0:
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings)
                    # Add embeddings to index
                    index.add(embeddings)
                
                return index
            else:
                # Fallback to numpy array
                return embeddings
                
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return embeddings
    
    def _search_faiss_index(
        self, 
        index: Any, 
        query_embedding: np.ndarray, 
        k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index with query embedding
        
        Args:
            index: FAISS index or numpy array
            query_embedding: Query embedding
            k: Number of results
            
        Returns:
            Tuple of (scores, indices)
        """
        try:
            # Validate input
            if index is None:
                logger.debug("Index is None, returning empty results")
                return np.array([]), np.array([])
            
            if query_embedding is None or query_embedding.size == 0:
                logger.debug("Query embedding is empty, returning empty results")
                return np.array([]), np.array([])
            
            if k <= 0:
                logger.debug(f"Invalid k value: {k}, returning empty results")
                return np.array([]), np.array([])
            
            if FAISS_AVAILABLE and hasattr(index, 'search'):
                # Check if index is empty
                if hasattr(index, 'ntotal') and index.ntotal == 0:
                    logger.debug("FAISS index is empty, returning empty results")
                    return np.array([]), np.array([])
                
                # Reshape query embedding
                query_array = query_embedding.reshape(1, -1).astype(np.float32)
                
                # Validate dimensions
                if hasattr(index, 'd') and query_array.shape[1] != index.d:
                    logger.error(f"Dimension mismatch: query has {query_array.shape[1]} dims, index expects {index.d}")
                    return np.array([]), np.array([])
                
                # Normalize query for cosine similarity
                faiss.normalize_L2(query_array)
                
                # Adjust k value to avoid exceeding index size
                actual_k = min(k, index.ntotal) if hasattr(index, 'ntotal') else k
                if actual_k == 0:
                    return np.array([]), np.array([])
                
                # Search index
                scores, indices = index.search(query_array, actual_k)
                
                return scores[0], indices[0]
            else:
                # Fallback to numpy similarity search
                return self._numpy_similarity_search(index, query_embedding, k)
                
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}", exc_info=True)
            # Return empty results instead of crashing
            return np.array([]), np.array([])
    
    def _numpy_similarity_search(
        self, 
        embeddings: np.ndarray, 
        query_embedding: np.ndarray, 
        k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback numpy-based similarity search
        
        Args:
            embeddings: Document embeddings
            query_embedding: Query embedding
            k: Number of results
            
        Returns:
            Tuple of (scores, indices)
        """
        try:
            # Normalize embeddings
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            # Calculate cosine similarities
            similarities = np.dot(embeddings_norm, query_norm)
            
            # Get top k results
            k = min(k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:k]
            top_scores = similarities[top_indices]
            
            return top_scores, top_indices
            
        except Exception as e:
            logger.error(f"Error in numpy similarity search: {e}")
            return np.array([]), np.array([])
    
    def _save_faiss_index(self, index: Any, user_id: str, index_type: str) -> bool:
        """
        Save FAISS index to disk
        
        Args:
            index: FAISS index to save
            user_id: User ID
            index_type: "episode" or "semantic"
            
        Returns:
            True if saved successfully
        """
        try:
            if not FAISS_AVAILABLE or isinstance(index, np.ndarray):
                return False
            
            # Determine save path
            if index_type == "episode":
                save_path = os.path.join(self.episode_vector_dir, f"{user_id}.faiss")
            else:
                save_path = os.path.join(self.semantic_vector_dir, f"{user_id}.faiss")
            
            # Save index
            faiss.write_index(index, save_path)
            logger.debug(f"Saved FAISS index for user {user_id} ({index_type}) to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index for user {user_id} ({index_type}): {e}")
            return False
    
    def _load_faiss_index(self, user_id: str, index_type: str) -> Optional[Any]:
        """
        Load FAISS index from disk
        
        Args:
            user_id: User ID
            index_type: "episode" or "semantic"
            
        Returns:
            FAISS index or None if not found
        """
        try:
            if not FAISS_AVAILABLE:
                return None
            
            # Determine load path
            if index_type == "episode":
                load_path = os.path.join(self.episode_vector_dir, f"{user_id}.faiss")
            else:
                load_path = os.path.join(self.semantic_vector_dir, f"{user_id}.faiss")
            
            if not os.path.exists(load_path):
                return None
            
            # Load index
            index = faiss.read_index(load_path)
            logger.debug(f"Loaded FAISS index for user {user_id} ({index_type}) from {load_path}")
            return index
            
        except Exception as e:
            logger.error(f"Error loading FAISS index for user {user_id} ({index_type}): {e}")
            return None
    
    def _save_embeddings(self, embeddings: np.ndarray, user_id: str, index_type: str) -> bool:
        """
        Save embeddings to disk
        
        Args:
            embeddings: Embeddings array
            user_id: User ID
            index_type: "episode" or "semantic"
            
        Returns:
            True if saved successfully
        """
        try:
            # Determine save path
            if index_type == "episode":
                save_path = os.path.join(self.episode_vector_dir, f"{user_id}_embeddings.npy")
            else:
                save_path = os.path.join(self.semantic_vector_dir, f"{user_id}_embeddings.npy")
            
            # Save embeddings
            np.save(save_path, embeddings)
            logger.debug(f"Saved embeddings for user {user_id} ({index_type}) to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings for user {user_id} ({index_type}): {e}")
            return False
    
    def _load_embeddings(self, user_id: str, index_type: str) -> Optional[np.ndarray]:
        """
        Load embeddings from disk
        
        Args:
            user_id: User ID
            index_type: "episode" or "semantic"
            
        Returns:
            Embeddings array or None if not found
        """
        try:
            # Determine load path
            if index_type == "episode":
                load_path = os.path.join(self.episode_vector_dir, f"{user_id}_embeddings.npy")
            else:
                load_path = os.path.join(self.semantic_vector_dir, f"{user_id}_embeddings.npy")
            
            if not os.path.exists(load_path):
                return None
            
            # Load embeddings
            embeddings = np.load(load_path)
            logger.debug(f"Loaded embeddings for user {user_id} ({index_type}) from {load_path}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings for user {user_id} ({index_type}): {e}")
            return None
    
    def index_episodes(self, user_id: str, episodes: List[Episode]):
        """
        Index episodes for a user (FULL REBUILD - only for initial indexing)
        
        Args:
            user_id: User ID
            episodes: List of episodes to index
        """
        try:
            if not episodes:
                logger.debug(f"No episodes to index for user {user_id}")
                return
            
            # Generate embeddings for episodes
            episode_texts = []
            for episode in episodes:
                # Combine title and content for embedding
                combined_text = f"{episode.title}. {episode.content}"
                episode_texts.append(combined_text)
            
            # Get embeddings
            embeddings_response = self.embedding_client.embed_texts(episode_texts)
            embeddings = embeddings_response.embeddings
            
            # Validate embeddings
            if not embeddings:
                logger.warning(f"No embeddings returned for user {user_id}")
                return
            
            # Convert to numpy array with proper validation
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Validate array shape
            if embeddings_array.ndim != 2:
                logger.error(f"Invalid embedding shape for user {user_id}: {embeddings_array.shape}")
                return
            
            if embeddings_array.shape[1] != self.dimension:
                logger.error(f"Embedding dimension mismatch for user {user_id}: expected {self.dimension}, got {embeddings_array.shape[1]}")
                return
            
            # Create FAISS index
            index = self._create_faiss_index(embeddings_array)
            
            # Store index, data, and embeddings
            self.episode_indices[user_id] = index
            self.episode_data[user_id] = episodes
            self.episode_embeddings[user_id] = embeddings_array
            
            # Save to disk
            self._save_faiss_index(index, user_id, "episode")
            self._save_embeddings(embeddings_array, user_id, "episode")
            
            logger.info(f"Indexed {len(episodes)} episodes for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error indexing episodes for user {user_id}: {e}")
    
    def index_semantic_memories(self, user_id: str, memories: List[SemanticMemory]):
        """
        Index semantic memories for a user (FULL REBUILD - only for initial indexing)
        
        Args:
            user_id: User ID
            memories: List of semantic memories to index
        """
        try:
            if not memories:
                logger.debug(f"No semantic memories to index for user {user_id}")
                return
            
            # Generate embeddings for memories
            memory_texts = [memory.content for memory in memories]
            
            # Get embeddings
            embeddings_response = self.embedding_client.embed_texts(memory_texts)
            embeddings = embeddings_response.embeddings
            
            # Validate embeddings
            if not embeddings:
                logger.warning(f"No embeddings returned for user {user_id}")
                return
            
            # Convert to numpy array with proper validation
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Validate array shape
            if embeddings_array.ndim != 2:
                logger.error(f"Invalid embedding shape for user {user_id}: {embeddings_array.shape}")
                return
            
            if embeddings_array.shape[1] != self.dimension:
                logger.error(f"Embedding dimension mismatch for user {user_id}: expected {self.dimension}, got {embeddings_array.shape[1]}")
                return
            
            # Create FAISS index
            index = self._create_faiss_index(embeddings_array)
            
            # Store index, data, and embeddings
            self.semantic_indices[user_id] = index
            self.semantic_data[user_id] = memories
            self.semantic_embeddings[user_id] = embeddings_array
            
            # Save to disk
            self._save_faiss_index(index, user_id, "semantic")
            self._save_embeddings(embeddings_array, user_id, "semantic")
            
            logger.info(f"Indexed {len(memories)} semantic memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error indexing semantic memories for user {user_id}: {e}")
    
    def search_episodes(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search episodes using vector similarity
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if user_id not in self.episode_indices:
                logger.debug(f"No episode index found for user {user_id}")
                return []
            
            index = self.episode_indices[user_id]
            episodes = self.episode_data[user_id]
            
            # Get query embedding
            query_embedding = self.embedding_client.embed_text(query)
            query_array = np.array(query_embedding)
            
            # Search index
            scores, indices = self._search_faiss_index(index, query_array, min(top_k, len(episodes)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if score > 0.1:  # Minimum similarity threshold
                    episode = episodes[idx]
                    results.append({
                        "type": "episodic",
                        "score": float(score),
                        "episode_id": episode.episode_id,
                        "title": episode.title,
                        "content": episode.content,
                        "original_messages": episode.original_messages,
                        "boundary_reason": episode.boundary_reason,
                        "timestamp": episode.timestamp.isoformat(),
                        "created_at": episode.created_at.isoformat(),
                        "message_count": episode.message_count,
                        "search_method": "vector"
                    })
            
            logger.debug(f"Vector episode search returned {len(results)} results for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching episodes for user {user_id}: {e}")
            return []
    
    def search_semantic_memories(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memories using vector similarity
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if user_id not in self.semantic_indices:
                logger.debug(f"No semantic index found for user {user_id}")
                return []
            
            index = self.semantic_indices[user_id]
            memories = self.semantic_data[user_id]
            
            # Get query embedding
            query_embedding = self.embedding_client.embed_text(query)
            query_array = np.array(query_embedding)
            
            # Search index
            scores, indices = self._search_faiss_index(index, query_array, min(top_k, len(memories)))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if score > 0.1:  # Minimum similarity threshold
                    memory = memories[idx]
                    results.append({
                        "type": "semantic",
                        "score": float(score),
                        "memory_id": memory.memory_id,
                        "knowledge_type": memory.knowledge_type,
                        "content": memory.content,
                        "confidence": memory.confidence,
                        "related_episodes": memory.source_episodes,
                        "created_at": memory.created_at.isoformat(),
                        "search_method": "vector"
                    })
            
            logger.debug(f"Vector semantic search returned {len(results)} results for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching semantic memories for user {user_id}: {e}")
            return []
    
    def search_all(
        self, 
        user_id: str, 
        query: str, 
        top_k_episodes: int = 5, 
        top_k_semantic: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search both episodes and semantic memories
        
        Args:
            user_id: User ID
            query: Search query
            top_k_episodes: Number of episode results
            top_k_semantic: Number of semantic results
            
        Returns:
            Combined search results
        """
        episode_results = self.search_episodes(user_id, query, top_k_episodes)
        semantic_results = self.search_semantic_memories(user_id, query, top_k_semantic)
        
        # Combine and sort by score
        all_results = episode_results + semantic_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results
    
    def add_episode(self, user_id: str, episode: Episode):
        """
        Add a single episode to the index using INCREMENTAL UPDATE
        
        Args:
            user_id: User ID
            episode: Episode to add
        """
        try:
            # Initialize data structures if user doesn't exist
            if user_id not in self.episode_data:
                self.episode_data[user_id] = []
                self.episode_embeddings[user_id] = np.empty((0, self.dimension), dtype=np.float32)
                self.episode_indices[user_id] = self._create_faiss_index(np.empty((0, self.dimension), dtype=np.float32))
            
            # Generate embedding for the new episode
            combined_text = f"{episode.title}. {episode.content}"
            episode_embedding = self.embedding_client.embed_text(combined_text)
            
            if not episode_embedding:
                logger.warning(f"No embedding returned for new episode")
                return
            
            # Convert to numpy array
            new_embedding = np.array([episode_embedding], dtype=np.float32)
            
            # Validate dimension
            if new_embedding.shape[1] != self.dimension:
                logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {new_embedding.shape[1]}")
                return
            
            # Add episode to data
            self.episode_data[user_id].append(episode)
            
            # Add embedding to embeddings array
            self.episode_embeddings[user_id] = np.vstack([self.episode_embeddings[user_id], new_embedding])
            
            # Incrementally update FAISS index
            index = self.episode_indices[user_id]
            if FAISS_AVAILABLE and hasattr(index, 'add'):
                # Normalize for cosine similarity
                faiss.normalize_L2(new_embedding)
                # Add to existing FAISS index
                index.add(new_embedding)
                logger.debug(f"Incrementally added episode to FAISS index for user {user_id}")
            else:
                # For numpy fallback, update the stored embeddings
                self.episode_indices[user_id] = self.episode_embeddings[user_id]
                logger.debug(f"Updated numpy embeddings for user {user_id}")
            
            # Save updated index and embeddings to disk
            self._save_faiss_index(index, user_id, "episode")
            self._save_embeddings(self.episode_embeddings[user_id], user_id, "episode")
            
            logger.info(f"🚀 Incrementally added episode to vector index for user {user_id} (total: {len(self.episode_data[user_id])})")
            
        except Exception as e:
            logger.error(f"Error incrementally adding episode to vector index: {e}")
            # Fallback to full rebuild if incremental update fails
            logger.warning(f"Falling back to full rebuild for user {user_id}")
            self.index_episodes(user_id, self.episode_data[user_id])
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory):
        """
        Add a single semantic memory to the index using INCREMENTAL UPDATE
        
        Args:
            user_id: User ID
            memory: Semantic memory to add
        """
        try:
            # Initialize data structures if user doesn't exist
            if user_id not in self.semantic_data:
                self.semantic_data[user_id] = []
                self.semantic_embeddings[user_id] = np.empty((0, self.dimension), dtype=np.float32)
                self.semantic_indices[user_id] = self._create_faiss_index(np.empty((0, self.dimension), dtype=np.float32))
            
            # Generate embedding for the new memory
            memory_embedding = self.embedding_client.embed_text(memory.content)
            
            if not memory_embedding:
                logger.warning(f"No embedding returned for new semantic memory")
                return
            
            # Convert to numpy array
            new_embedding = np.array([memory_embedding], dtype=np.float32)
            
            # Validate dimension
            if new_embedding.shape[1] != self.dimension:
                logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {new_embedding.shape[1]}")
                return
            
            # Add memory to data
            self.semantic_data[user_id].append(memory)
            
            # Add embedding to embeddings array
            self.semantic_embeddings[user_id] = np.vstack([self.semantic_embeddings[user_id], new_embedding])
            
            # Incrementally update FAISS index
            index = self.semantic_indices[user_id]
            if FAISS_AVAILABLE and hasattr(index, 'add'):
                # Normalize for cosine similarity
                faiss.normalize_L2(new_embedding)
                # Add to existing FAISS index
                index.add(new_embedding)
                logger.debug(f"Incrementally added semantic memory to FAISS index for user {user_id}")
            else:
                # For numpy fallback, update the stored embeddings
                self.semantic_indices[user_id] = self.semantic_embeddings[user_id]
                logger.debug(f"Updated numpy embeddings for user {user_id}")
            
            # Save updated index and embeddings to disk
            self._save_faiss_index(index, user_id, "semantic")
            self._save_embeddings(self.semantic_embeddings[user_id], user_id, "semantic")
            
            logger.info(f"🚀 Incrementally added semantic memory to vector index for user {user_id} (total: {len(self.semantic_data[user_id])})")
            
        except Exception as e:
            logger.error(f"Error incrementally adding semantic memory to vector index: {e}")
            # Fallback to full rebuild if incremental update fails
            logger.warning(f"Falling back to full rebuild for user {user_id}")
            self.index_semantic_memories(user_id, self.semantic_data[user_id])
    
    def clear_user_index(self, user_id: str) -> bool:
        """
        Clear all indices for a user
        
        Args:
            user_id: User ID
            
        Returns:
            True if cleared successfully
        """
        try:
            # Clear episode index
            if user_id in self.episode_indices:
                del self.episode_indices[user_id]
            if user_id in self.episode_data:
                del self.episode_data[user_id]
            if user_id in self.episode_embeddings:
                del self.episode_embeddings[user_id]
            
            # Clear semantic index
            if user_id in self.semantic_indices:
                del self.semantic_indices[user_id]
            if user_id in self.semantic_data:
                del self.semantic_data[user_id]
            if user_id in self.semantic_embeddings:
                del self.semantic_embeddings[user_id]
            
            logger.info(f"Cleared vector indices for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector indices for user {user_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get vector index statistics
        
        Returns:
            Index statistics
        """
        episode_users = len(self.episode_indices)
        semantic_users = len(self.semantic_indices)
        
        total_episodes = sum(len(episodes) for episodes in self.episode_data.values())
        total_memories = sum(len(memories) for memories in self.semantic_data.values())
        
        return {
            "episode_users": episode_users,
            "semantic_users": semantic_users,
            "total_episodes": total_episodes,
            "total_semantic_memories": total_memories,
            "embedding_dimension": self.dimension,
            "search_engine": "FAISS" if FAISS_AVAILABLE else "Numpy",
            "faiss_available": FAISS_AVAILABLE,
            "incremental_updates": True  # Indicates support for incremental updates
        } 