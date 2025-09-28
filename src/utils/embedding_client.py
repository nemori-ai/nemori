"""
Embedding Client
"""

import openai
import numpy as np
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """Embedding response data model"""
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    response_time: float


class EmbeddingClient:
    """Embedding vector client"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: Optional[str] = None):
        """
        Initialize embedding client
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            base_url: API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Configuration parameters
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30.0
        self.batch_size = 100  # Batch processing size
        
        # Embedding dimension (set based on model)
        self.embedding_dim = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-3-large" in self.model:
            return 3072
        elif "text-embedding-ada-002" in self.model:
            return 1536
        else:
            # Default dimension
            return 1536
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """
        Generate embedding vectors for text list
        
        Args:
            texts: Text list
            
        Returns:
            Embedding response
        """
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                usage={},
                model=self.model,
                response_time=0.0
            )
        
        start_time = time.time()
        all_embeddings = []
        total_usage = {"prompt_tokens": 0, "total_tokens": 0}
        
        # Batch processing
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        timeout=self.timeout
                    )
                    
                    # Extract embedding vectors
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Accumulate usage
                    if response.usage:
                        usage = response.usage.model_dump()
                        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        raise e
        
        response_time = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=total_usage,
            model=self.model,
            response_time=response_time
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for single text
        
        Args:
            text: Text
            
        Returns:
            Embedding vector
        """
        response = self.embed_texts([text])
        return response.embeddings[0] if response.embeddings else []
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity
        
        Args:
            vector1: Vector 1
            vector2: Vector 2
            
        Returns:
            Cosine similarity
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vector dimensions do not match")
        
        # Convert to numpy arrays
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def batch_cosine_similarity(self, query_vector: List[float], 
                              vectors: List[List[float]]) -> List[float]:
        """
        Batch calculate cosine similarity
        
        Args:
            query_vector: Query vector
            vectors: Vector list
            
        Returns:
            Similarity list
        """
        if not vectors:
            return []
        
        # Convert to numpy arrays
        query = np.array(query_vector)
        matrix = np.array(vectors)
        
        # Batch calculate cosine similarity
        dot_products = np.dot(matrix, query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        
        # Avoid division by zero
        similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms!=0)
        
        return similarities.tolist()
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize vector
        
        Args:
            vector: Vector
            
        Returns:
            Normalized vector
        """
        v = np.array(vector)
        norm = np.linalg.norm(v)
        
        if norm == 0:
            return vector
        
        return (v / norm).tolist()
    
    def vector_distance(self, vector1: List[float], vector2: List[float], 
                       metric: str = "cosine") -> float:
        """
        Calculate vector distance
        
        Args:
            vector1: Vector 1
            vector2: Vector 2
            metric: Distance metric ("cosine", "euclidean", "manhattan")
            
        Returns:
            Distance value
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vector dimensions do not match")
        
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == "cosine":
            return 1 - self.cosine_similarity(vector1, vector2)
        elif metric == "euclidean":
            return np.linalg.norm(v1 - v2)
        elif metric == "manhattan":
            return np.sum(np.abs(v1 - v2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count
        
        Args:
            text: Text
            
        Returns:
            Token count (rough estimate)
        """
        # Rough estimate: ~4 characters per token for English, ~1.5 characters per token for Chinese
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_chars = len(text) - chinese_chars
        
        return int(chinese_chars / 1.5 + english_chars / 4)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "api_key_prefix": self.api_key[:10] + "..." if self.api_key else None,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "batch_size": self.batch_size
        } 