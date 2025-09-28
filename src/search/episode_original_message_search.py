"""
Episode Original Message Search Engine
Search engine that builds indexes on the original_messages field in episodes
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

try:
    import spacy
    from spacy.lang.en import English
    from spacy.lang.zh import Chinese
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, falling back to simple tokenization")

logger = logging.getLogger(__name__)


class EpisodeOriginalMessageSearch:
    """
    Episode original message BM25 search engine
    
    Builds indexes on the original_messages field in episodes, returns the most relevant episode original message blocks during search
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize episode original message search engine
        
        Args:
            language: Language for tokenization ("en" for English, "zh" for Chinese)
        """
        self.language = language
        
        # Episode original message indexes
        self.episode_indices: Dict[str, BM25Okapi] = {}  # user_id -> BM25 index
        self.episode_data: Dict[str, List[Dict[str, Any]]] = {}  # user_id -> [episode_data]
        self.episode_tokenized_texts: Dict[str, List[List[str]]] = {}  # user_id -> [tokenized_texts]
        
        # Initialize spaCy tokenizer
        self.nlp = self._initialize_spacy_tokenizer()
        
        logger.info(f"Episode original message search engine initialized with {self.language} tokenization (spaCy available: {SPACY_AVAILABLE})")
    
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
    
    def load_user_episodes(self, user_id: str, storage_path: str):
        """
        Load user episodes and build indexes for original_messages
        
        Args:
            user_id: User ID
            storage_path: Storage path
        """
        try:
            # Build episode file path
            episode_file = os.path.join(storage_path, "episodes", f"{user_id}_episodes.jsonl")
            
            if not os.path.exists(episode_file):
                logger.warning(f"Episode file not found: {episode_file}")
                return
            
            # Read episodes
            episodes = []
            with open(episode_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode = json.loads(line.strip())
                        episodes.append(episode)
            
            if not episodes:
                logger.debug(f"No episodes found for user {user_id}")
                return
            
            # Build indexes for original_messages of each episode
            episode_data = []
            episode_texts = []
            
            for episode in episodes:
                original_messages = episode.get('original_messages', [])
                
                if not original_messages:
                    logger.debug(f"No original_messages in episode {episode.get('episode_id', 'unknown')}")
                    continue
                
                # Combine all original_messages of an episode into one text for indexing
                combined_text = ""
                message_contents = []
                
                for msg in original_messages:
                    content = msg.get('content', '')
                    if content:
                        message_contents.append(content)
                        combined_text += content + " "
                
                if not combined_text.strip():
                    continue
                
                # Build episode data
                episode_data_item = {
                    "episode_id": episode.get("episode_id", ""),
                    "user_id": user_id,
                    "title": episode.get("title", ""),
                    "episode_content": episode.get("content", ""),  # episode summary content
                    "original_messages": original_messages,  # complete original messages
                    "combined_message_text": combined_text.strip(),  # combined text for search
                    "message_contents": message_contents,  # list of individual message contents
                    "timestamp": episode.get("timestamp", ""),
                    "created_at": episode.get("created_at", ""),
                    "message_count": len(original_messages),
                    "source_type": "episode_original_messages"
                }
                
                episode_data.append(episode_data_item)
                episode_texts.append(combined_text.strip())
            
            if not episode_texts:
                logger.warning(f"No valid episode original messages found for user {user_id}")
                return
            
            # Tokenize
            tokenized_texts = [self._tokenize(text) for text in episode_texts]
            
            # Filter empty tokenization results
            valid_tokenized_texts = []
            valid_episode_data = []
            for i, tokens in enumerate(tokenized_texts):
                if tokens:  # only include episodes with valid tokens
                    valid_tokenized_texts.append(tokens)
                    valid_episode_data.append(episode_data[i])
            
            if not valid_tokenized_texts:
                logger.warning(f"No valid tokenized texts for user {user_id}")
                return
            
            # Create BM25 index
            bm25_index = BM25Okapi(valid_tokenized_texts)
            
            # Store index, data and tokenized texts
            self.episode_indices[user_id] = bm25_index
            self.episode_data[user_id] = valid_episode_data
            self.episode_tokenized_texts[user_id] = valid_tokenized_texts
            
            logger.info(f"Indexed {len(valid_episode_data)} episode original message blocks for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error loading episodes for user {user_id}: {e}")
    
    def search_episode_original_messages(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search episode original_messages
        
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
                    episode = episodes[idx]
                    results.append({
                        "type": "episode_original_messages",
                        "score": float(scores[idx]),
                        "episode_id": episode.get("episode_id", ""),
                        "title": episode.get("title", ""),
                        "episode_content": episode.get("episode_content", ""),
                        "original_messages": episode.get("original_messages", []),
                        "combined_message_text": episode.get("combined_message_text", ""),
                        "message_contents": episode.get("message_contents", []),
                        "timestamp": episode.get("timestamp", ""),
                        "created_at": episode.get("created_at", ""),
                        "message_count": episode.get("message_count", 0),
                        "source_type": "episode_original_messages",
                        "search_method": "bm25_episode_original"
                    })
            
            logger.debug(f"Episode original message search returned {len(results)} results for user {user_id} (query tokens: {query_tokens})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching episode original messages for user {user_id}: {e}")
            return []
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's episode original message index statistics
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "user_id": user_id,
            "has_index": user_id in self.episode_indices,
            "total_episodes": 0,
            "total_messages": 0
        }
        
        if user_id in self.episode_data:
            episodes = self.episode_data[user_id]
            stats["total_episodes"] = len(episodes)
            
            # Count total number of messages
            total_messages = sum(episode.get("message_count", 0) for episode in episodes)
            stats["total_messages"] = total_messages
        
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
            
            if user_id in self.episode_indices:
                del self.episode_indices[user_id]
                removed_something = True
            
            if user_id in self.episode_data:
                del self.episode_data[user_id]
                removed_something = True
            
            if user_id in self.episode_tokenized_texts:
                del self.episode_tokenized_texts[user_id]
                removed_something = True
            
            if removed_something:
                logger.info(f"Cleared episode original message index for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing episode original message index for user {user_id}: {e}")
            return False 