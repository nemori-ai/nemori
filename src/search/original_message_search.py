"""
Original message search engine - for comparison experiments
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


class OriginalMessageSearch:
    """
    Original message BM25 search engine - for comparison experiments
    
    Directly builds BM25 index on original messages without using episodic memory summary content
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize original message search engine
        
        Args:
            language: Language for tokenization ("en" for English, "zh" for Chinese)
        """
        self.language = language
        
        # Original message indexes
        self.message_indices: Dict[str, BM25Okapi] = {}  # user_id -> BM25 index
        self.message_data: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> [message_data]
        self.message_tokenized_texts: Dict[str, List[List[str]]] = {}  # user_id -> [tokenized_texts]
        
        # Initialize spaCy tokenizer
        self.nlp = self._initialize_spacy_tokenizer()
        
        logger.info(f"Original message search engine initialized with {self.language} tokenization (spaCy available: {SPACY_AVAILABLE})")
    
    def _initialize_spacy_tokenizer(self):
        """Initialize spaCy tokenizer based on language"""
        if not SPACY_AVAILABLE:
            return None
        
        try:
            if self.language == "zh":
                # Chinese tokenizer
                nlp = Chinese()
                nlp.add_pipe("sentencizer")
            else:
                # English tokenizer (default)
                nlp = English()
                nlp.add_pipe("sentencizer")
            
            logger.info(f"spaCy tokenizer initialized for {self.language}")
            return nlp
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy tokenizer: {e}, falling back to simple tokenization")
            return None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy or simple tokenization
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        try:
            if self.nlp:
                # Use spaCy tokenization
                doc = self.nlp(text.lower())
                tokens = []
                for token in doc:
                    if not token.is_space and not token.is_punct and len(token.text) > 1:
                        tokens.append(token.text)
                return tokens
            else:
                # Fallback to simple tokenization
                tokens = text.lower().split()
                # Remove punctuation and short tokens
                cleaned_tokens = []
                for token in tokens:
                    cleaned = ''.join(c for c in token if c.isalnum())
                    if len(cleaned) > 1:
                        cleaned_tokens.append(cleaned)
                return cleaned_tokens
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using simple split")
            return text.lower().split()
    
    def index_episodes_original_messages(self, user_id: str, episodes: List[Episode]):
        """
        Build BM25 index for original messages in all user's episodic memories
        
        Args:
            user_id: User ID
            episodes: List of episodic memories
        """
        try:
            if not episodes:
                logger.debug(f"No episodes to index for user {user_id}")
                return
            
            # Extract all original messages
            message_data = []
            message_texts = []
            
            for episode in episodes:
                if hasattr(episode, 'original_messages') and episode.original_messages:
                    for msg in episode.original_messages:
                        # Build message data
                        msg_data = {
                            "message_id": msg.get("message_id", ""),
                            "role": msg.get("role", ""),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("timestamp", ""),
                            "metadata": msg.get("metadata", {}),
                            "episode_id": episode.episode_id,
                            "episode_title": episode.title,
                            "episode_timestamp": episode.timestamp.isoformat() if hasattr(episode, 'timestamp') else "",
                            "source_type": "original_message"
                        }
                        
                        message_data.append(msg_data)
                        
                        # Build index using message content
                        message_texts.append(msg.get("content", ""))
            
            if not message_texts:
                logger.warning(f"No original messages found for user {user_id}")
                return
            
            # Tokenize
            tokenized_texts = [self._tokenize(text) for text in message_texts]
            
            # Filter empty tokenization results
            valid_tokenized_texts = []
            valid_message_data = []
            for i, tokens in enumerate(tokenized_texts):
                if tokens:  # only include messages with valid tokens
                    valid_tokenized_texts.append(tokens)
                    valid_message_data.append(message_data[i])
            
            if not valid_tokenized_texts:
                logger.warning(f"No valid tokenized texts for user {user_id}")
                return
            
            # Create BM25 index
            bm25_index = BM25Okapi(valid_tokenized_texts)
            
            # Store index, data and tokenized texts
            self.message_indices[user_id] = bm25_index
            self.message_data[user_id] = valid_message_data
            self.message_tokenized_texts[user_id] = valid_tokenized_texts
            
            logger.info(f"Indexed {len(valid_message_data)} original messages for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error indexing original messages for user {user_id}: {e}")
    
    def search_original_messages(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search original messages
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if user_id not in self.message_indices or self.message_indices[user_id] is None:
                logger.debug(f"No original message index found for user {user_id}")
                return []
            
            bm25_index = self.message_indices[user_id]
            messages = self.message_data[user_id]
            
            # Tokenize query
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
                if scores[idx] > 0:  # only include results with positive scores
                    message = messages[idx]
                    results.append({
                        "type": "original_message",
                        "score": float(scores[idx]),
                        "message_id": message.get("message_id", ""),
                        "content": message.get("content", ""),
                        "role": message.get("role", ""),
                        "timestamp": message.get("timestamp", ""),
                        "metadata": message.get("metadata", {}),
                        "episode_id": message.get("episode_id", ""),
                        "episode_title": message.get("episode_title", ""),
                        "episode_timestamp": message.get("episode_timestamp", ""),
                        "source_type": "original_message",
                        "search_method": "bm25_original"
                    })
            
            logger.debug(f"Original message search returned {len(results)} results for user {user_id} (query tokens: {query_tokens})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching original messages for user {user_id}: {e}")
            return []
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's original message index statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "user_id": user_id,
            "has_index": user_id in self.message_indices,
            "total_messages": 0,
            "total_episodes": 0
        }
        
        if user_id in self.message_data:
            messages = self.message_data[user_id]
            stats["total_messages"] = len(messages)
            
            # Count unique episode numbers
            episode_ids = set()
            for msg in messages:
                if msg.get("episode_id"):
                    episode_ids.add(msg["episode_id"])
            stats["total_episodes"] = len(episode_ids)
        
        return stats
    
    def clear_user_index(self, user_id: str) -> bool:
        """
        Clear user's index data
        
        Args:
            user_id: User ID
            
        Returns:
            Whether successfully cleared
        """
        try:
            removed_something = False
            
            if user_id in self.message_indices:
                del self.message_indices[user_id]
                removed_something = True
            
            if user_id in self.message_data:
                del self.message_data[user_id]
                removed_something = True
            
            if user_id in self.message_tokenized_texts:
                del self.message_tokenized_texts[user_id]
                removed_something = True
            
            if removed_something:
                logger.info(f"Cleared original message index for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing original message index for user {user_id}: {e}")
            return False 