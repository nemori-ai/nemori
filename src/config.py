"""
Memory System Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MemoryConfig:
    """Memory System Configuration"""
    
    # === Basic Configuration ===
    storage_path: str = "./memories"
    
    # === Model Configuration ===
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # === Language Configuration ===
    language: str = "en"  # "en" for English, "zh" for Chinese
    
    # === Buffer Configuration ===
    buffer_size_min: int = 2       # Minimum buffer size
    buffer_size_max: int = 25       # Maximum buffer size
    
    # === Batch Segmentation Configuration ===
    enable_batch_segmentation: bool = True       # Enable batch segmentation mode
    batch_threshold: int = 20                    # Number of messages to trigger batch processing
    
    # === Episode Generation Configuration ===
    episode_min_messages: int = 2    # Minimum number of messages for an episode
    episode_max_messages: int = 25   # Maximum number of messages for an episode
    
    # === Episode Merging Configuration ===
    enable_episode_merging: bool = True          # Enable episode merging
    merge_similarity_threshold: float = 0.85     # Similarity threshold for merge candidates
    merge_top_k: int = 5                         # Number of similar episodes to consider for merging
    
    # === Semantic Memory Configuration ===
    enable_semantic_memory: bool = True          # Enable semantic memory
    semantic_similarity_threshold: float = 1  # Semantic similarity threshold for duplication detection
    enable_prediction_correction: bool = True    # Enable prediction-correction mode (simplified two-step process only)
    extract_semantic_per_episode: bool = False   # Extract semantic memory immediately for each new episode (single episode mode)
    
    # === Search Configuration ===
    search_top_k_episodes: int = 10             # Number of episode memory search results
    search_top_k_semantic: int = 10             # Number of semantic memory search results
    enable_parallel_search: bool = True          # Enable parallel search
    
    # === Ranking Configuration (Vector Aggregation) ===
    enable_norlift_ranking: bool = False         # Enable NOR-LIFT aggregation for episode ranking
    norlift_pool_size_episodes: int = 100        # Candidate pool size from episode vectors
    norlift_pool_size_semantic: int = 200        # Candidate pool size from semantic vectors
    norlift_percentile_tau: float = 0.95         # Percentile threshold for z-score shift
    norlift_sigmoid_lambda: float = 2.5          # Sigmoid steepness
    norlift_epsilon: float = 1e-6                # Numerical stability epsilon
    
    # === Storage / Index Backends ===
    storage_backend: str = "filesystem"         # "filesystem" | "memory"
    vector_index_backend: str = "chroma"        # "chroma" | "memory"
    lexical_index_backend: str = "bm25"          # "bm25" | "memory"

    # === Vector Database Configuration ===
    vector_db_type: str = "chroma"              # Vector database type: "chroma"
    chroma_persist_directory: str = "./chroma_db"  # ChromaDB persistence directory
    chroma_collection_prefix: str = "nemori"    # ChromaDB collection name prefix
    
    # === Performance Configuration ===
    batch_size: int = 32                        # Batch size
    max_workers: int = 4                        # Maximum number of worker threads
    semantic_generation_workers: int = 8         # Number of semantic memory generation threads
    
    # === Cache Configuration ===
    enable_cache: bool = True                   # Enable cache
    cache_size: int = 1000                      # Cache size
    cache_ttl_seconds: int = 3600               # Cache expiration time (seconds)
    semantic_cache_ttl: int = 600               # Semantic cache TTL
    episode_cache_ttl: int = 600                # Episode cache TTL
    
    # === Environment Variable Configuration ===
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # === Prediction-Correction Configuration ===
    max_statements_for_prediction: int = 10      # Maximum number of statements for prediction
    statement_similarity_threshold: float = 0.7  # Statement similarity threshold
    prediction_temperature: float = 0.3  # Temperature parameter for prediction
    
    def __post_init__(self):
        """Configuration validation"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.buffer_size_min >= self.buffer_size_max:
            raise ValueError("Buffer min size must be less than max size")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConfig':
        """Create configuration from dictionary"""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False 
