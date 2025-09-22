"""
BM25 Search Engine
"""

import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ..models import Episode, SemanticMemory

try:
    import spacy
    from spacy.lang.en import English
    from spacy.lang.zh import Chinese
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, falling back to simple tokenization")

logger = logging.getLogger(__name__)


class BM25Search:
    """BM25-based text search engine with incremental updates and spaCy tokenization"""
    
    def __init__(self, language: str = "en"):
        """
        Initialize BM25 search engine
        
        Args:
            language: Language for tokenization ("en" for English, "zh" for Chinese)
        """
        self.language = language
        self.episode_indices: Dict[str, BM25Okapi] = {}  # user_id -> BM25 index
        self.episode_data: Dict[str, List[Episode]] = {}  # user_id -> [episodes]
        self.episode_tokenized_texts: Dict[str, List[List[str]]] = {}  # user_id -> [tokenized_texts] - cached tokenized results
        
        self.semantic_indices: Dict[str, BM25Okapi] = {}  # user_id -> BM25 index
        self.semantic_data: Dict[str, List[SemanticMemory]] = {}  # user_id -> [memories]
        self.semantic_tokenized_texts: Dict[str, List[List[str]]] = {}  # user_id -> [tokenized_texts] - cached tokenized results
        
        # Initialize spaCy tokenizer
        self.nlp = self._initialize_spacy_tokenizer()
        
        logger.info(f"BM25 search engine initialized with {self.language} tokenization (spaCy available: {SPACY_AVAILABLE})")
    
    def _initialize_spacy_tokenizer(self):
        """
        Initialize spaCy tokenizer based on language
        
        Returns:
            spaCy language model or None if not available
        """
        if not SPACY_AVAILABLE:
            return None
        
        try:
            if self.language == "zh":
                # Try to load Chinese model, fallback to simple Chinese tokenizer
                try:
                    nlp = spacy.load("zh_core_web_sm")
                    logger.info("Loaded Chinese spaCy model")
                except OSError:
                    logger.warning("Chinese spaCy model not found, using simple Chinese tokenizer")
                    nlp = Chinese()
                    nlp.add_pipe('sentencizer')
            else:
                # Try to load English model, fallback to simple English tokenizer
                try:
                    nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded English spaCy model")
                except OSError:
                    logger.warning("English spaCy model not found, using simple English tokenizer")
                    nlp = English()
                    nlp.add_pipe('sentencizer')
            
            return nlp
        except Exception as e:
            logger.error(f"Error initializing spaCy tokenizer: {e}")
            return None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy or fallback to simple tokenization
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        try:
            if self.nlp:
                # Use spaCy for tokenization
                doc = self.nlp(text.lower())
                
                # Extract tokens, filter out punctuation, spaces, and stop words
                tokens = []
                for token in doc:
                    if (not token.is_punct and 
                        not token.is_space and 
                        not token.is_stop and 
                        len(token.text.strip()) > 1):
                        tokens.append(token.lemma_)  # Use lemmatized form
                
                return tokens
                
            else:
                # Fallback to simple tokenization
                import re
                tokens = re.findall(r'\b\w+\b', text.lower())
                return [token for token in tokens if len(token) > 1]  # Filter out single characters
                
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Ultra-simple fallback
            return text.lower().split()
    
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
            
            # Prepare texts for indexing
            episode_texts = []
            for episode in episodes:
                # Combine title and content for better search
                combined_text = f"{episode.title} {episode.content}"
                episode_texts.append(combined_text)
            
            # Tokenize texts using spaCy
            tokenized_texts = [self._tokenize(text) for text in episode_texts]
            
            # Filter out empty tokenizations
            valid_tokenized_texts = []
            valid_episodes = []
            for i, tokens in enumerate(tokenized_texts):
                if tokens:  # Only include episodes with valid tokens
                    valid_tokenized_texts.append(tokens)
                    valid_episodes.append(episodes[i])
            
            if not valid_tokenized_texts:
                logger.warning(f"No valid tokenized texts for user {user_id}")
                return
            
            # Create BM25 index
            bm25_index = BM25Okapi(valid_tokenized_texts)
            
            # Store index, data, and tokenized texts
            self.episode_indices[user_id] = bm25_index
            self.episode_data[user_id] = valid_episodes
            self.episode_tokenized_texts[user_id] = valid_tokenized_texts
            
            logger.info(f"Indexed {len(valid_episodes)} episodes for user {user_id} using spaCy tokenization")
            
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
            
            # Prepare texts for indexing
            memory_texts = [memory.content for memory in memories]
            
            # Tokenize texts using spaCy
            tokenized_texts = [self._tokenize(text) for text in memory_texts]
            
            # Filter out empty tokenizations
            valid_tokenized_texts = []
            valid_memories = []
            for i, tokens in enumerate(tokenized_texts):
                if tokens:  # Only include memories with valid tokens
                    valid_tokenized_texts.append(tokens)
                    valid_memories.append(memories[i])
            
            if not valid_tokenized_texts:
                logger.warning(f"No valid tokenized texts for user {user_id}")
                return
            
            # Create BM25 index
            bm25_index = BM25Okapi(valid_tokenized_texts)
            
            # Store index, data, and tokenized texts
            self.semantic_indices[user_id] = bm25_index
            self.semantic_data[user_id] = valid_memories
            self.semantic_tokenized_texts[user_id] = valid_tokenized_texts
            
            logger.info(f"Indexed {len(valid_memories)} semantic memories for user {user_id} using spaCy tokenization")
            
        except Exception as e:
            logger.error(f"Error indexing semantic memories for user {user_id}: {e}")
    
    def search_episodes(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search episodes using BM25 with spaCy tokenization
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if user_id not in self.episode_indices or self.episode_indices[user_id] is None:
                logger.debug(f"No episode index found for user {user_id}")
                return []
            
            bm25_index = self.episode_indices[user_id]
            episodes = self.episode_data[user_id]
            
            # Tokenize query using spaCy
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.debug("Query tokenization resulted in empty tokens")
                return []
            
            # Get BM25 scores
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    episode = episodes[idx]
                    results.append({
                        "type": "episodic",
                        "score": float(scores[idx]),
                        "episode_id": episode.episode_id,
                        "title": episode.title,
                        "content": episode.content,
                        "original_messages": episode.original_messages,
                        "boundary_reason": episode.boundary_reason,
                        "timestamp": episode.timestamp.isoformat(),
                        "created_at": episode.created_at.isoformat(),
                        "message_count": episode.message_count,
                        "search_method": "bm25"
                    })
            
            logger.debug(f"BM25 episode search returned {len(results)} results for user {user_id} (query tokens: {query_tokens})")
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
        Search semantic memories using BM25 with spaCy tokenization
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if user_id not in self.semantic_indices or self.semantic_indices[user_id] is None:
                logger.debug(f"No semantic index found for user {user_id}")
                return []
            
            bm25_index = self.semantic_indices[user_id]
            memories = self.semantic_data[user_id]
            
            # Tokenize query using spaCy
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.debug("Query tokenization resulted in empty tokens")
                return []
            
            # Get BM25 scores
            scores = bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    memory = memories[idx]
                    results.append({
                        "type": "semantic",
                        "score": float(scores[idx]),
                        "memory_id": memory.memory_id,
                        "knowledge_type": memory.knowledge_type,
                        "content": memory.content,
                        "confidence": memory.confidence,
                        "related_episodes": memory.source_episodes,
                        "created_at": memory.created_at.isoformat(),
                        "search_method": "bm25"
                    })
            
            logger.debug(f"BM25 semantic search returned {len(results)} results for user {user_id} (query tokens: {query_tokens})")
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
        Add a single episode to the index using OPTIMIZED UPDATE
        
        Args:
            user_id: User ID
            episode: Episode to add
        """
        try:
            # Initialize data structures if user doesn't exist
            if user_id not in self.episode_data:
                self.episode_data[user_id] = []
                self.episode_tokenized_texts[user_id] = []
                self.episode_indices[user_id] = None  # Will be created when we have data
            
            # Prepare text for the new episode
            combined_text = f"{episode.title} {episode.content}"
            tokenized_text = self._tokenize(combined_text)
            
            if not tokenized_text:
                logger.warning(f"No tokens generated for new episode")
                return
            
            # Add episode to data
            self.episode_data[user_id].append(episode)
            self.episode_tokenized_texts[user_id].append(tokenized_text)
            
            # Rebuild BM25 index with all tokenized texts (optimized: reuse cached tokenizations)
            # BM25Okapi doesn't support incremental updates, but we cache tokenizations to avoid re-tokenizing
            all_tokenized_texts = self.episode_tokenized_texts[user_id]
            
            # Only create index if we have valid tokenized texts
            if all_tokenized_texts and all(tokens for tokens in all_tokenized_texts):
                # Create new BM25 index with all texts
                bm25_index = BM25Okapi(all_tokenized_texts)
                self.episode_indices[user_id] = bm25_index
                
                logger.info(f"🚀 Optimized BM25 update: added episode for user {user_id} (total: {len(self.episode_data[user_id])}, reused {len(all_tokenized_texts)-1} cached tokenizations)")
            else:
                logger.warning(f"Cannot create BM25 index with empty tokenized texts for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding episode to BM25 index: {e}")
            # Fallback to full rebuild if incremental update fails
            logger.warning(f"Falling back to full rebuild for user {user_id}")
            if self.episode_data.get(user_id):
                self.index_episodes(user_id, self.episode_data[user_id])
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory):
        """
        Add a single semantic memory to the index using OPTIMIZED UPDATE
        
        Args:
            user_id: User ID
            memory: Semantic memory to add
        """
        try:
            # Initialize data structures if user doesn't exist
            if user_id not in self.semantic_data:
                self.semantic_data[user_id] = []
                self.semantic_tokenized_texts[user_id] = []
                self.semantic_indices[user_id] = None  # Will be created when we have data
            
            # Prepare text for the new memory
            tokenized_text = self._tokenize(memory.content)
            
            if not tokenized_text:
                logger.warning(f"No tokens generated for new semantic memory")
                return
            
            # Add memory to data
            self.semantic_data[user_id].append(memory)
            self.semantic_tokenized_texts[user_id].append(tokenized_text)
            
            # Rebuild BM25 index with all tokenized texts (optimized: reuse cached tokenizations)
            # BM25Okapi doesn't support incremental updates, but we cache tokenizations to avoid re-tokenizing
            all_tokenized_texts = self.semantic_tokenized_texts[user_id]
            
            # Only create index if we have valid tokenized texts
            if all_tokenized_texts and all(tokens for tokens in all_tokenized_texts):
                # Create new BM25 index with all texts
                bm25_index = BM25Okapi(all_tokenized_texts)
                self.semantic_indices[user_id] = bm25_index
                
                logger.info(f"🚀 Optimized BM25 update: added semantic memory for user {user_id} (total: {len(self.semantic_data[user_id])}, reused {len(all_tokenized_texts)-1} cached tokenizations)")
            else:
                logger.warning(f"Cannot create BM25 index with empty tokenized texts for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding semantic memory to BM25 index: {e}")
            # Fallback to full rebuild if incremental update fails
            logger.warning(f"Falling back to full rebuild for user {user_id}")
            if self.semantic_data.get(user_id):
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
            if user_id in self.episode_tokenized_texts:
                del self.episode_tokenized_texts[user_id]
            
            # Clear semantic index
            if user_id in self.semantic_indices:
                del self.semantic_indices[user_id]
            if user_id in self.semantic_data:
                del self.semantic_data[user_id]
            if user_id in self.semantic_tokenized_texts:
                del self.semantic_tokenized_texts[user_id]
            
            logger.info(f"Cleared BM25 indices for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing indices for user {user_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics
        
        Returns:
            Index statistics
        """
        episode_users = len(self.episode_indices)
        semantic_users = len(self.semantic_indices)
        
        total_episodes = sum(len(episodes) for episodes in self.episode_data.values())
        total_memories = sum(len(memories) for memories in self.semantic_data.values())
        
        # Count cached tokenizations
        total_episode_tokens = sum(len(tokens) for tokens in self.episode_tokenized_texts.values())
        total_semantic_tokens = sum(len(tokens) for tokens in self.semantic_tokenized_texts.values())
        
        return {
            "episode_users": episode_users,
            "semantic_users": semantic_users,
            "total_episodes": total_episodes,
            "total_semantic_memories": total_memories,
            "cached_episode_tokenizations": total_episode_tokens,
            "cached_semantic_tokenizations": total_semantic_tokens,
            "search_engine": "BM25Okapi",
            "tokenization": "spaCy" if SPACY_AVAILABLE else "simple",
            "language": self.language,
            "spacy_available": SPACY_AVAILABLE,
            "spacy_model_loaded": self.nlp is not None,
            "optimized_updates": True  # Indicate support for optimized updates
        } 