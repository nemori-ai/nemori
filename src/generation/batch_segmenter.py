"""
Batch-based message segmentation for intelligent episode grouping.

This module provides a batch segmentation strategy that accumulates messages
and uses LLM to intelligently group them into multiple episodes at once.
"""

import json
import logging
from typing import List, Dict, Any
from ..models import Message
from ..utils import LLMClient
from ..config import MemoryConfig
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class BatchSegmenter:
    """Batch-based segmenter using LLM for intelligent episode grouping.
    
    This segmenter accumulates messages up to a threshold, then uses LLM
    to analyze the entire batch and intelligently group messages into episodes.
    Episodes can have non-continuous indices based on topic coherence.
    
    Attributes:
        llm_client: LLM client for segmentation.
        config: Memory system configuration.
        threshold: Number of messages to trigger batch segmentation.
        prompts: Prompt templates.
    """
    
    def __init__(self, llm_client: LLMClient, config: MemoryConfig):
        """Initialize batch segmenter.
        
        Args:
            llm_client: LLM client for segmentation.
            config: Memory system configuration.
        """
        self.llm_client = llm_client
        self.config = config
        self.threshold = config.batch_threshold
        self.prompts = PromptTemplates()
        
        logger.info(f"BatchSegmenter initialized (threshold={self.threshold}, model={llm_client.model})")
    
    def should_create_episode(self, buffer_size: int) -> tuple[bool, str]:
        """Check if batch threshold is reached.
        
        Args:
            buffer_size: Number of messages in buffer.
            
        Returns:
            (True, reason) if threshold reached, (False, "") otherwise.
        """
        if buffer_size >= self.threshold:
            return True, f"Batch threshold reached: {buffer_size}/{self.threshold}"
        return False, ""
    
    def segment_batch(self, messages: List[Message]) -> List[List[int]]:
        """Segment a batch of messages into episode groups using LLM.
        
        This method analyzes all messages together and returns intelligent groupings
        that can have non-continuous indices based on topic coherence.
        
        Args:
            messages: List of messages to segment.
            
        Returns:
            List of episode groups, where each group is a list of message indices (1-based).
            Example: [[1,2,3], [4,5,6,7], [8,10,11], [9,12]]
            
        Raises:
            Exception: If LLM returns invalid response.
        """
        if not messages:
            return []
        
        # Format messages with indices for the prompt
        formatted_messages = self._format_messages_with_indices(messages)
        
        # Generate segmentation prompt
        prompt = self.prompts.get_batch_segmentation_prompt(
            count=len(messages),
            messages=formatted_messages
        )
        
        try:
            # Call LLM for intelligent segmentation
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.2,  # Low temperature for consistent segmentation
                max_tokens=8192,
                category="batch_segmentation"
            )
            
            if not isinstance(response, dict):
                raise ValueError("LLM returned invalid response format")
            
            # Extract episode groups
            episode_groups: List[List[int]] = []
            for ep in response.get("episodes", []):
                indices = ep.get("indices", [])
                topic = ep.get("topic", "")
                
                if indices:
                    episode_groups.append(indices)
                    logger.debug(f"Episode group: {indices} - {topic}")
            
            if not episode_groups:
                logger.warning("LLM returned no episode groups, creating single episode")
                # Fallback: all messages in one episode
                episode_groups = [[i+1 for i in range(len(messages))]]
            
            logger.info(
                f"Batch segmentation complete: {len(messages)} messages → "
                f"{len(episode_groups)} episodes"
            )
            
            return episode_groups
            
        except Exception as e:
            logger.error(f"Error in batch segmentation: {e}")
            # Fallback: all messages in one episode
            return [[i+1 for i in range(len(messages))]]
    
    def _format_messages_with_indices(self, messages: List[Message]) -> str:
        """Format messages with 1-based indices for the segmentation prompt.
        
        Args:
            messages: List of messages to format.
            
        Returns:
            Formatted string with numbered messages.
        """
        lines: List[str] = []
        
        for i, message in enumerate(messages, 1):
            timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            role = message.role
            content = message.content[:200]  # Limit to 200 chars for clarity
            
            lines.append(
                f"{i}. [{timestamp}] {role}: {content}"
            )
        
        return "\n".join(lines)


__all__ = [
    "BatchSegmenter",
]

