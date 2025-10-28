"""
Episode merger for consolidating similar episodes.

This module provides functionality to detect and merge similar episodes
based on semantic similarity and temporal proximity.
"""

import logging
import uuid
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from ..models import Episode
from ..utils import LLMClient, EmbeddingClient
from ..config import MemoryConfig
from ..storage import EpisodeStorage
from ..search import ChromaSearchEngine
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


def _normalize_datetime(dt: datetime) -> datetime:
    """Convert timezone-aware datetime to naive datetime for comparison."""
    if dt is None:
        return datetime.now()
    if dt.tzinfo is not None:
        # Convert to UTC and remove timezone info
        return dt.replace(tzinfo=None)
    return dt


class EpisodeMerger:
    """Merge similar episodes.
    
    This merger uses vector similarity search to find related episodes
    and LLM-based decision making to determine if merging is appropriate.
    
    Attributes:
        llm_client: LLM client for merge decisions.
        embedding_client: Embedding client for generating embeddings.
        config: Memory system configuration.
        episode_storage: Episode storage for reading episodes.
        vector_search: Vector search engine for similarity search.
        prompts: Prompt templates.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        config: MemoryConfig,
        episode_storage: EpisodeStorage,
        vector_search: ChromaSearchEngine
    ):
        """Initialize episode merger.
        
        Args:
            llm_client: LLM client for merge decisions.
            embedding_client: Embedding client for generating embeddings.
            config: Memory system configuration.
            episode_storage: Episode storage for reading episodes.
            vector_search: Vector search engine for similarity search.
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.episode_storage = episode_storage
        self.vector_search = vector_search
        self.prompts = PromptTemplates()
        
        logger.info("EpisodeMerger initialized")
    
    def check_and_merge(
        self,
        new_episode: Episode,
        top_k: int = 5,
        similarity_threshold: float = 0.85
    ) -> Tuple[bool, Optional[Episode], Optional[str]]:
        """Check if new episode should merge with existing ones.
        
        This method:
        1. Searches for similar existing episodes using vector similarity
        2. Uses LLM to decide if merging is appropriate
        3. If yes, generates merged content
        
        Args:
            new_episode: Newly created episode.
            top_k: Number of similar episodes to consider.
            similarity_threshold: Minimum similarity score to consider merging.
            
        Returns:
            Tuple of (merged, final_episode, old_episode_id):
            - If merged: (True, merged_episode, old_episode_id_to_delete)
            - If not merged: (False, None, None)
        """
        try:
            # 1. Search for similar episodes
            candidates = self._search_similar_episodes(
                new_episode,
                top_k=top_k
            )
            
            if not candidates:
                return False, None, None
            
            # 2. Decide if merging is appropriate
            should_merge, target_id, reason = self._decide_merge(
                new_episode,
                candidates
            )
            
            if not should_merge or not target_id:
                return False, None, None
            
            # 3. Find the target episode
            target_episode = next(
                (ep for ep in candidates if ep.episode_id == target_id),
                None
            )
            
            if not target_episode:
                return False, None, None
            
            # 4. Generate merged content
            merged_episode = self._merge_contents(
                target_episode,
                new_episode
            )
            
            # Return merged episode and the old episode ID to delete
            logger.info(f"Episode merge successful: {merged_episode.episode_id} (deleted old: {target_id[:8]}...)")
            return True, merged_episode, target_id
            
        except Exception as e:
            logger.error(f"Error in check_and_merge: {e}")
            # On error, don't merge
            return False, None, None
    
    def _search_similar_episodes(
        self,
        episode: Episode,
        top_k: int = 5
    ) -> List[Episode]:
        """Search for similar episodes using vector similarity.
        
        Args:
            episode: Episode to find similar ones for.
            top_k: Number of results to return.
            
        Returns:
            List of similar episodes sorted by relevance.
        """
        try:
            # Use vector search with episode content
            search_results = self.vector_search.search_episodes(
                user_id=episode.user_id,
                query=episode.content[:500],  # Use first 500 chars as query
                top_k=top_k + 1  # +1 to account for potential self-match
            )
            
            # Extract episodes (excluding self)
            similar_episodes: List[Episode] = []
            for result in search_results:
                ep_id = result.get("episode_id")
                if ep_id and ep_id != episode.episode_id:
                    # Load full episode from storage
                    ep = self.episode_storage.get_episode(ep_id, episode.user_id)
                    if ep:
                        similar_episodes.append(ep)
            
            logger.info(f"Found {len(similar_episodes)} similar episodes for merge consideration")
            return similar_episodes[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search similar episodes: {e}")
            return []
    
    def _decide_merge(
        self,
        new_episode: Episode,
        candidates: List[Episode]
    ) -> Tuple[bool, Optional[str], str]:
        """Use LLM to decide if merging is appropriate.
        
        Args:
            new_episode: Newly created episode.
            candidates: List of candidate episodes to merge with.
            
        Returns:
            Tuple of (should_merge, target_id, reason).
        """
        try:
            # Format candidates
            candidates_text = self._format_candidates(candidates)
            
            # Format new episode time range
            new_time_range = (
                f"{new_episode.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({new_episode.message_count} messages)"
            )
            
            prompt = self.prompts.get_merge_decision_prompt(
                new_time_range=new_time_range,
                new_content=new_episode.content,
                candidates=candidates_text
            )
            
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=4096,
                category="merge_decision"
            )
            
            if not isinstance(response, dict):
                raise ValueError("LLM returned invalid response format")
            
            decision = response.get("decision", "new")
            target_id = response.get("merge_target_id")
            reason = response.get("reason", "")
            
            should_merge = decision == "merge" and target_id is not None
            
            logger.debug(f"Merge decision: {decision}, target: {target_id}, reason: {reason}")
            return should_merge, target_id, reason
            
        except Exception as e:
            logger.error(f"Error in merge decision: {e}")
            return False, None, f"Error in merge decision: {str(e)}"
    
    def _merge_contents(
        self,
        target: Episode,
        new_episode: Episode
    ) -> Episode:
        """Generate merged episode content.
        
        Args:
            target: Original episode to merge into.
            new_episode: New episode to merge.
            
        Returns:
            New merged Episode instance.
        """
        try:
            # Format time ranges
            original_time_range = (
                f"{target.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({target.message_count} messages)"
            )
            new_time_range = (
                f"{new_episode.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({new_episode.message_count} messages)"
            )
            
            # Combined events description
            combined_events = f"Original episode: {target.content}\n\nNew episode: {new_episode.content}"
            
            prompt = self.prompts.get_merge_content_prompt(
                original_time_range=original_time_range,
                original_title=target.title,
                original_content=target.content,
                new_time_range=new_time_range,
                new_title=new_episode.title,
                new_content=new_episode.content,
                combined_events=combined_events
            )
            
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.7,
                max_tokens=8192,
                category="merge_content"
            )
            
            if not isinstance(response, dict):
                raise ValueError("LLM returned invalid response format")
            
            # Merge original messages from both episodes
            merged_messages = target.original_messages + new_episode.original_messages
            
            # Use earliest timestamp (normalize for comparison)
            target_ts = _normalize_datetime(target.timestamp)
            new_ts = _normalize_datetime(new_episode.timestamp)
            merged_timestamp = min(target_ts, new_ts)
            
            # Parse timestamp from response
            timestamp_str = response.get("timestamp", merged_timestamp.isoformat())
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                parsed_timestamp = merged_timestamp
            
            # Create new merged episode
            merged_episode = Episode(
                user_id=new_episode.user_id,
                title=response.get("title", f"Merged: {target.title}"),
                content=response.get("content", f"{target.content}\n\n{new_episode.content}"),
                original_messages=merged_messages,
                message_count=len(merged_messages),
                boundary_reason=f"Merged episode (original: {target.episode_id}, new: {new_episode.episode_id})",
                timestamp=parsed_timestamp,
                episode_id=str(uuid.uuid4()),  # New ID for merged episode
                metadata={
                    "merged_from": [target.episode_id, new_episode.episode_id],
                    "merge_timestamp": datetime.now().isoformat()
                }
            )
            
            return merged_episode
            
        except Exception as e:
            logger.error(f"Failed to generate merged content: {e}")
            # Fallback: return new episode without merging
            raise ValueError(f"Failed to generate merged content: {str(e)}")
    
    def _format_candidates(self, candidates: List[Episode]) -> str:
        """Format candidate episodes for the prompt.
        
        Args:
            candidates: List of candidate episodes.
            
        Returns:
            Formatted string representation.
        """
        lines: List[str] = []
        for i, episode in enumerate(candidates, 1):
            time_str = episode.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            lines.append(
                f"{i}. Candidate ID: {episode.episode_id}\n"
                f"   Time: {time_str} ({episode.message_count} messages)\n"
                f"   Title: {episode.title}\n"
                f"   Content: {episode.content[:200]}..."  # First 200 chars
            )
        
        return "\n\n".join(lines)


__all__ = [
    "EpisodeMerger",
]

