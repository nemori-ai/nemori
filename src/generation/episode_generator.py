"""
Episode Generator
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from ..models import Episode, Message
from ..utils import LLMClient
from ..config import MemoryConfig
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class EpisodeGenerator:
    """Episode generator for creating episodic memories from conversations"""
    
    def __init__(self, llm_client: LLMClient, config: MemoryConfig):
        """
        Initialize episode generator
        
        Args:
            llm_client: LLM client for generation
            config: Memory system configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.prompts = PromptTemplates()
        
        logger.info("Episode generator initialized")
    
    def generate_episode(
        self, 
        user_id: str, 
        messages: List[Message], 
        boundary_reason: str = ""
    ) -> Episode:
        """
        Generate an episode from a list of messages
        
        Args:
            user_id: User ID
            messages: List of messages to convert to episode
            boundary_reason: Reason for episode boundary
            
        Returns:
            Generated episode
        """
        try:
            if not messages:
                raise ValueError("Cannot generate episode from empty message list")
            
            # Format conversation for prompt
            conversation = self.prompts.format_conversation([msg.to_dict(clean_metadata=False) for msg in messages])
            
            # Generate episode using LLM
            episode_data = self._generate_episode_with_llm(conversation, boundary_reason)
            
            # Determine timestamp for the episode
            episode_timestamp = self._determine_episode_timestamp(episode_data, messages)
            
            # Create episode object with determined timestamp
            episode = Episode(
                user_id=user_id,
                title=episode_data.get("title", "Untitled Episode"),
                content=episode_data.get("content", "No content generated"),
                original_messages=[msg.to_dict(clean_metadata=True) for msg in messages],
                message_count=len(messages),
                boundary_reason=boundary_reason,
                timestamp=episode_timestamp
            )
            
            logger.info(f"Generated episode: {episode.title} ({len(messages)} messages) at {episode_timestamp}")
            return episode
            
        except Exception as e:
            logger.error(f"Error generating episode: {e}")
            # Create fallback episode
            return self._create_fallback_episode(user_id, messages, boundary_reason)
    
    def _generate_episode_with_llm(self, conversation: str, boundary_reason: str) -> Dict[str, Any]:
        """
        Generate episode using LLM
        
        Args:
            conversation: Formatted conversation text
            boundary_reason: Reason for episode boundary
            
        Returns:
            Episode data dictionary
        """
        try:
            # Generate prompt
            prompt = self.prompts.get_episode_generation_prompt(
                conversation=conversation,
                boundary_reason=boundary_reason
            )
            
            # Call LLM
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024*8,
                default_response={
                    "title": "Episode generation failed",
                    "content": "Unable to generate episode content due to an error.",
                    "timestamp": datetime.now().isoformat()
                }  # 提供适合的默认响应
            )
            
            # Validate response
            if not isinstance(response, dict):
                raise ValueError("Invalid response format")
            
            required_fields = ["title", "content", "timestamp"]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                logger.warning(f"Missing fields in LLM response: {missing_fields}")
                # Only require title and content as essential
                if "title" not in response or "content" not in response:
                    raise ValueError("Missing required fields in response")
            
            # Clean and validate title
            title = str(response["title"]).strip()
            if not title:
                title = "Generated Episode"
            
            # Clean and validate content
            content = str(response["content"]).strip()
            if not content:
                content = "Episode content not generated"
            
            # Parse and validate timestamp
            timestamp = None
            if "timestamp" in response:
                timestamp_str = str(response["timestamp"]).strip()
                if timestamp_str:
                    try:
                        # Parse ISO format timestamp
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        logger.debug(f"Parsed timestamp from LLM: {timestamp}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid timestamp format from LLM: {timestamp_str}, error: {e}")
                        timestamp = None
            
            logger.debug(f"Generated episode: {title}")
            
            result = {
                "title": title,
                "content": content
            }
            
            # Include timestamp if valid
            if timestamp:
                result["timestamp"] = timestamp
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM episode generation: {e}")
            raise
    
    def _determine_episode_timestamp(self, episode_data: Dict[str, Any], messages: List[Message]) -> datetime:
        """
        Determine the timestamp for the episode
        
        Args:
            episode_data: Episode data from LLM
            messages: Original messages
            
        Returns:
            Timestamp for the episode
        """
        # Priority 1: Use timestamp from LLM if available and valid
        if "timestamp" in episode_data and episode_data["timestamp"]:
            llm_timestamp = episode_data["timestamp"]
            if isinstance(llm_timestamp, datetime):
                logger.debug(f"Using LLM-generated timestamp: {llm_timestamp}")
                return llm_timestamp
        
        # Priority 2: Use the earliest message timestamp
        if messages:
            try:
                earliest_message = min(messages, key=lambda msg: msg.timestamp)
                logger.debug(f"Using earliest message timestamp: {earliest_message.timestamp}")
                return earliest_message.timestamp
            except (AttributeError, ValueError) as e:
                logger.warning(f"Error getting message timestamps: {e}")
        
        # Priority 3: Use current time as fallback
        current_time = datetime.now()
        logger.warning(f"Using current time as fallback timestamp: {current_time}")
        return current_time
    
    def _create_fallback_episode(
        self, 
        user_id: str, 
        messages: List[Message], 
        boundary_reason: str
    ) -> Episode:
        """
        Create a fallback episode when LLM generation fails
        
        Args:
            user_id: User ID
            messages: List of messages
            boundary_reason: Reason for episode boundary
            
        Returns:
            Fallback episode
        """
        try:
            # Determine episode timestamp using the same logic
            episode_timestamp = self._determine_episode_timestamp({}, messages)
            
            # Generate simple title based on timestamp and message count
            if messages:
                first_message_time = messages[0].timestamp
                title = f"Conversation {first_message_time.strftime('%Y-%m-%d %H:%M')} ({len(messages)} messages)"
            else:
                title = f"Episode {episode_timestamp.strftime('%Y-%m-%d %H:%M')} (0 messages)"
            
            # Generate simple content summary
            content_parts = [
                f"Episode created at {episode_timestamp.strftime('%Y-%m-%d %H:%M:%S')} with {len(messages)} messages.",
                f"Participants exchanged messages about various topics.",
                f"Episode created due to: {boundary_reason}" if boundary_reason else "Episode created automatically."
            ]
            
            # Add sample of messages if available
            if len(messages) > 0:
                first_msg = messages[0]
                content_parts.append(f"First message from {first_msg.role}: {first_msg.content[:100]}...")
            
            if len(messages) > 1:
                last_msg = messages[-1]
                content_parts.append(f"Last message from {last_msg.role}: {last_msg.content[:100]}...")
            
            content = " ".join(content_parts)
            
            # Create episode with determined timestamp
            episode = Episode(
                user_id=user_id,
                title=title,
                content=content,
                original_messages=[msg.to_dict(clean_metadata=True) for msg in messages],
                message_count=len(messages),
                boundary_reason=boundary_reason or "Fallback episode creation",
                timestamp=episode_timestamp
            )
            
            logger.warning(f"Created fallback episode: {title} at {episode_timestamp}")
            return episode
            
        except Exception as e:
            logger.error(f"Error creating fallback episode: {e}")
            # Last resort - minimal episode with current time
            current_time = datetime.now()
            return Episode(
                user_id=user_id,
                title="Error Episode",
                content="Episode could not be generated due to errors",
                original_messages=[msg.to_dict(clean_metadata=True) for msg in messages] if messages else [],
                message_count=len(messages) if messages else 0,
                boundary_reason="Error in episode generation",
                timestamp=current_time
            )
    
    def batch_generate_episodes(
        self, 
        user_id: str, 
        message_batches: List[List[Message]], 
        boundary_reasons: List[str] = None
    ) -> List[Episode]:
        """
        Generate multiple episodes in batch
        
        Args:
            user_id: User ID
            message_batches: List of message lists
            boundary_reasons: Optional list of boundary reasons
            
        Returns:
            List of generated episodes
        """
        episodes = []
        boundary_reasons = boundary_reasons or [""] * len(message_batches)
        
        for i, messages in enumerate(message_batches):
            try:
                reason = boundary_reasons[i] if i < len(boundary_reasons) else ""
                episode = self.generate_episode(user_id, messages, reason)
                episodes.append(episode)
                
            except Exception as e:
                logger.error(f"Error generating episode {i} in batch: {e}")
                # Create fallback episode for this batch
                fallback = self._create_fallback_episode(user_id, messages, f"Batch generation error: {str(e)}")
                episodes.append(fallback)
        
        logger.info(f"Generated {len(episodes)} episodes in batch for user {user_id}")
        return episodes
    
    def validate_episode(self, episode: Episode) -> bool:
        """
        Validate an episode
        
        Args:
            episode: Episode to validate
            
        Returns:
            True if episode is valid
        """
        try:
            # Check required fields
            if not episode.title or not episode.content:
                return False
            
            if not episode.user_id or not episode.episode_id:
                return False
            
            if episode.message_count <= 0:
                return False
            
            if not episode.original_messages:
                return False
            
            # Check data consistency
            if len(episode.original_messages) != episode.message_count:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating episode: {e}")
            return False
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """
        Get episode generator statistics
        
        Returns:
            Generator statistics
        """
        return {
            "config": {
                "llm_model": self.config.llm_model,
                "episode_min_messages": self.config.episode_min_messages,
                "episode_max_messages": self.config.episode_max_messages
            },
            "llm_client": self.llm_client.get_model_info() if hasattr(self.llm_client, 'get_model_info') else {}
        } 