"""
Semantic Memory Generator
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from ..models import SemanticMemory, Episode
from ..utils import LLMClient, EmbeddingClient
from ..config import MemoryConfig
from .prompts import PromptTemplates
from .prediction_correction_engine import PredictionCorrectionEngine

logger = logging.getLogger(__name__)


class SemanticGenerator:
    """Semantic memory generator for extracting semantic knowledge from episodes"""
    
    def __init__(self, llm_client: LLMClient, embedding_client: EmbeddingClient, config: MemoryConfig, vector_search=None):
        """
        Initialize semantic generator
        
        Args:
            llm_client: LLM client for generation
            embedding_client: Embedding client for similarity calculation
            config: Memory system configuration
            vector_search: Vector search engine instance (optional, for optimization)
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.vector_search = vector_search  # Add vector search engine reference
        self.prompts = PromptTemplates()
        
        # Initialize prediction-correction engine (if enabled)
        self.prediction_correction_engine = None
        if getattr(config, 'enable_prediction_correction', True):  # Default to new method
            self.prediction_correction_engine = PredictionCorrectionEngine(
                llm_client, embedding_client, config, vector_search  # Pass vector_search
            )
            logger.info("Semantic generator initialized with Prediction-Correction Engine")
        else:
            logger.info("Semantic generator initialized with traditional method")
    
    def check_and_generate_semantic_memories(
        self, 
        user_id: str,
        new_episode: Episode,
        existing_episodes: List[Episode],
        existing_semantic_memories: List[SemanticMemory] = None
    ) -> List[SemanticMemory]:
        """
        Check if new episode should trigger semantic memory generation
        
        Args:
            user_id: User ID
            new_episode: Newly created episode
            existing_episodes: All existing episodes for the user
            existing_semantic_memories: Existing semantic memories (for prediction-correction mode)
            
        Returns:
            List of generated semantic memories (empty if no generation triggered)
        """
        try:
            # If prediction-correction mode is enabled
            if self.prediction_correction_engine and existing_semantic_memories is not None:
                logger.info("Using Prediction-Correction Engine (simplified two-step) for semantic memory generation")
                return self.prediction_correction_engine.learn_from_episode_simplified(
                    user_id, new_episode, existing_semantic_memories
                )
            
            # Check if single episode direct extraction mode is enabled
            if getattr(self.config, 'extract_semantic_per_episode', False):
                logger.info("Using per-episode direct extraction mode")
                # Directly extract semantic memory from new episode
                semantic_memories = self.generate_semantic_memories(user_id, [new_episode])
                
                if semantic_memories:
                    logger.info(f"Generated {len(semantic_memories)} semantic memories from single episode: {new_episode.title}")
                
                return semantic_memories
            
            # If neither mode is enabled, return empty list
            logger.warning("No semantic memory generation method is enabled")
            return []
            
        except Exception as e:
            logger.error(f"Error in semantic memory check and generation: {e}")
            return []
    

    

    

    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def generate_semantic_memories(
        self, 
        user_id: str, 
        episodes: List[Episode]
    ) -> List[SemanticMemory]:
        """
        Generate semantic memories from episodes
        
        Args:
            user_id: User ID
            episodes: List of episodes to analyze
            
        Returns:
            List of generated semantic memories
        """
        try:
            if not episodes:
                logger.debug("No episodes provided for semantic memory generation")
                return []
            
            # For single episode extraction mode, use original_messages instead of content
            if len(episodes) == 1:
                episodes_text = self._format_episodes_from_original_messages(episodes)
                logger.debug("Using original messages for single episode extraction")
            else:
                episodes_text = self.prompts.format_episodes_for_semantic([ep.to_dict() for ep in episodes])
            
            # Generate semantic memories using LLM
            statements = self._generate_semantic_with_llm(episodes_text)
            
            # Convert to SemanticMemory objects
            semantic_memories = self._convert_to_semantic_memories(user_id, statements, episodes)
            
            logger.info(f"Generated {len(semantic_memories)} semantic memories from {len(episodes)} episodes")
            return semantic_memories
            
        except Exception as e:
            logger.error(f"Error generating semantic memories: {e}")
            return []
    
    def _generate_semantic_with_llm(self, episodes_text: str) -> List[str]:
        """Generate semantic memories using LLM"""
        try:
            prompt = self.prompts.get_semantic_generation_prompt(episodes=episodes_text)
            
            # Check if using single episode extraction mode
            if getattr(self.config, 'extract_semantic_per_episode', False):
                # Single episode mode returns memories of different types
                response = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=1500,
                    default_response={},  # Default response for single episode mode
                    max_retries=5
                )
                
                # Extract all memories from different types
                statements = []
                memory_types = ["user_profile", "experience", "knowledge", "other"]
                
                for memory_type in memory_types:
                    if memory_type in response:
                        # Handle single or multiple memories of the same type
                        if isinstance(response[memory_type], list):
                            statements.extend(response[memory_type])
                        elif isinstance(response[memory_type], str):
                            statements.append(response[memory_type])
                
                # If response has other unexpected keys, also try to extract
                for key, value in response.items():
                    if key not in memory_types:
                        if isinstance(value, str):
                            statements.append(value)
                        elif isinstance(value, list):
                            statements.extend(value)
                
                logger.debug(f"Extracted {len(statements)} statements from single episode response")
                return statements
            
            else:
                # Default mode (for prediction-correction simplified calls)
                response = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=1024*8,
                    default_response={"statements": []},
                    max_retries=5
                )
                
                # Extract statements list
                statements = response.get("statements", [])
                
                return statements
            
        except Exception as e:
            logger.error(f"Error in LLM semantic generation: {e}")
            return []
    
    def _format_episodes_from_original_messages(self, episodes: List[Episode]) -> str:
        """
        Format episodes using original messages for single episode extraction mode
        
        Args:
            episodes: List of episodes
            
        Returns:
            Formatted text for LLM prompt
        """
        formatted = []
        for i, episode in enumerate(episodes, 1):
            formatted.append(f"Episode {i}:")
            formatted.append(f"Title: {episode.title}")
            
            # Use original messages instead of content
            if episode.original_messages:
                formatted.append("Original Conversation:")
                for msg in episode.original_messages:
                    role = msg.get('role', 'Unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('metadata', {}).get('timestamp', '')
                    if timestamp:
                        formatted.append(f"  [{timestamp}] {role}: {content}")
                    else:
                        formatted.append(f"  {role}: {content}")
            else:
                # If no original_messages, fallback to content
                logger.warning(f"No original messages found for episode {episode.episode_id}, using content instead")
                formatted.append(f"Content: {episode.content}")
            
            formatted.append(f"Created at: {episode.created_at.isoformat()}")
            formatted.append("")  # Empty line separator
            
        return "\n".join(formatted)
    
    def _convert_to_semantic_memories(
        self, 
        user_id: str, 
        statements: List[str], 
        source_episodes: List[Episode]
    ) -> List[SemanticMemory]:
        """Convert semantic data to SemanticMemory objects"""
        memories = []
        source_episode_ids = [ep.episode_id for ep in source_episodes]
        
        # Use the earliest episode timestamp for semantic memories
        earliest_timestamp = None
        if source_episodes:
            earliest_episode = min(source_episodes, key=lambda ep: ep.timestamp)
            earliest_timestamp = earliest_episode.timestamp
            logger.debug(f"Using earliest episode timestamp for semantic memories: {earliest_timestamp}")
        
        for content in statements:
            try:
                content = content.strip()
                if not content:
                    continue
                
                # Create semantic memory with preserved timestamp
                memory = SemanticMemory(
                    user_id=user_id,
                    content=content,
                    knowledge_type="knowledge",  # Use generic "knowledge" type
                    source_episodes=source_episode_ids.copy(),
                    confidence=0.8
                )
                
                # Override created_at timestamp if we have episode timestamp
                if earliest_timestamp:
                    memory.created_at = earliest_timestamp
                
                memories.append(memory)
                
            except Exception as e:
                logger.error(f"Error creating semantic memory: {e}")
                continue
        
        return memories
    
    def batch_check_and_generate(
        self,
        user_episodes_map: Dict[str, Tuple[Episode, List[Episode]]]
    ) -> Dict[str, List[SemanticMemory]]:
        """
        Batch process semantic memory generation for multiple users
        
        Args:
            user_episodes_map: Dict mapping user_id to (new_episode, existing_episodes)
            
        Returns:
            Dict mapping user_id to generated semantic memories
        """
        all_memories = {}
        
        for user_id, (new_episode, existing_episodes) in user_episodes_map.items():
            try:
                memories = self.check_and_generate_semantic_memories(
                    user_id, new_episode, existing_episodes
                )
                all_memories[user_id] = memories
                
            except Exception as e:
                logger.error(f"Error in batch semantic generation for user {user_id}: {e}")
                all_memories[user_id] = []
        
        total_memories = sum(len(memories) for memories in all_memories.values())
        logger.info(f"Batch generated {total_memories} semantic memories for {len(user_episodes_map)} users")
        
        return all_memories
    

    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get semantic generator statistics"""
        return {
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "enable_semantic_memory": self.config.enable_semantic_memory,
                "enable_prediction_correction": self.config.enable_prediction_correction,
                "extract_semantic_per_episode": getattr(self.config, 'extract_semantic_per_episode', False)
            },
            "methods": ["prediction_correction_simplified", "single_episode_extraction"]
        } 