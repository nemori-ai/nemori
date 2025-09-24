"""
Prediction-Correction Engine
Knowledge learning system based on Free Energy Principle
"""

import logging
import json
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from ..models import Episode, SemanticMemory
from ..utils import LLMClient, EmbeddingClient
from ..config import MemoryConfig
from .prompts import PromptTemplates

logger = logging.getLogger(__name__)


class PredictionCorrectionEngine:
    """
    Prediction-correction engine (Simplified two-step process)
    
    Implements simplified learning cycle:
    1. Prediction: Predict episode content based on existing knowledge
    2. Direct extraction: Compare prediction with actual content and extract new knowledge directly
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        embedding_client: EmbeddingClient,
        config: MemoryConfig,
        vector_search=None
    ):
        """
        Initialize prediction-correction engine
        
        Args:
            llm_client: LLM client
            embedding_client: Embedding client
            config: System configuration
            vector_search: Vector search engine instance (optional, for optimization)
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.vector_search = vector_search  # Add vector search engine reference
        self.prompts = PromptTemplates()
        
        # 注意：使用ChromaDB向量搜索后，不再需要缓存embeddings
        # ChromaDB会自动管理和索引所有embeddings
        
        logger.info("Prediction-Correction Engine initialized")
    

    
    def learn_from_episode_simplified(
        self,
        user_id: str,
        new_episode: Episode,
        existing_statements: List[SemanticMemory]
    ) -> List[SemanticMemory]:
        """
        Simplified version: Learn knowledge from new episode (two-step process)
        
        Args:
            user_id: User ID
            new_episode: Newly created episode
            existing_statements: Existing statement knowledge base
            
        Returns:
            List of newly generated semantic memories
        """
        try:
            logger.debug(f"Starting simplified extraction for episode: {new_episode.title}")
            logger.debug(f"Existing knowledge statements: {len(existing_statements)}")
            
            # If no historical knowledge exists, use cold start mode
            if not existing_statements:
                logger.info("No existing knowledge, using cold start mode")
                return self._cold_start_extraction(user_id, new_episode)
            
            # Step 1: Prediction
            relevant_statements = self._retrieve_relevant_statements(
                new_episode, existing_statements
            )
            
            logger.info(f"Step 1 - Prediction: Retrieved {len(relevant_statements)} relevant statements")
            
            predicted_episode = self._predict_episode(
                new_episode.title, relevant_statements
            )
            
            # Step 2: Direct comparison and knowledge extraction
            logger.info("Step 2 - Direct extraction from comparison")
            new_statements = self._extract_knowledge_from_comparison(
                original_messages=new_episode.original_messages,
                predicted_episode=predicted_episode
            )
            
            logger.info(f"Extracted {len(new_statements)} knowledge statements")
            if new_statements:
                logger.info("Sample extracted statements:")
                for stmt in new_statements[:3]:
                    logger.info(f"  - {stmt}")
            
            # Step 3: Convert to semantic memory objects
            semantic_memories = self._convert_to_semantic_memories(
                user_id, new_statements, new_episode
            )
            
            logger.info(f"Generated {len(semantic_memories)} new semantic memories from simplified extraction")
            
            return semantic_memories
            
        except Exception as e:
            logger.error(f"Error in simplified extraction: {e}")
            return []
    
    def _cold_start_extraction(self, user_id: str, episode: Episode) -> List[SemanticMemory]:
        """
        Cold start mode: Extract initial knowledge directly from single episode
        
        Args:
            user_id: User ID
            episode: Episode
            
        Returns:
            List of extracted semantic memories
        """
        try:
            logger.info("Cold start: Extracting initial knowledge from first episode")
            
            # Build cold start prompt
            prompt = f"""
You are a knowledge extraction specialist. Extract HIGH-VALUE, PERSISTENT knowledge statements from the following episode.

Episode Title: {episode.title}
Episode Content: {episode.content}
Episode Time: {episode.timestamp.strftime("%Y-%m-%d")}

## CRITICAL: Focus on HIGH-VALUE Knowledge Only

Extract ONLY knowledge that passes these criteria:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs or preferences?
- **Independence Test**: Can this be understood without the conversation context?

## HIGH-VALUE Knowledge Categories (EXTRACT THESE):
1. **Identity & Background**: Names, professions, companies, education
2. **Persistent Preferences**: Favorite books/movies/tools, long-term likes/dislikes  
3. **Technical Details**: Technologies, versions, methodologies, architectures
4. **Relationships**: Family, colleagues, team members, mentors
5. **Goals & Plans**: Career objectives, learning goals, project plans
6. **Beliefs & Values**: Principles, philosophies, strong opinions
7. **Habits & Patterns**: Regular activities, workflows, schedules

## LOW-VALUE Knowledge (SKIP THESE):
- Temporary emotions or reactions
- Single conversation acknowledgments
- Vague statements without specifics
- Context-dependent information

## Guidelines:
1. Each statement should be self-contained and atomic
2. Include ALL specific details (names, versions, titles)
3. Use present tense for persistent facts
4. Focus on facts that help understand the user long-term
5. DO NOT include time/date information in the statement
6. Quality over quantity - fewer valuable statements are better

## Examples:
GOOD: "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
GOOD: "The user works at ByteDance as a senior ML engineer"
BAD: "The user thanked the assistant"
BAD: "The user was happy about the response"

Return in JSON format:
{{
    "statements": [
        "First high-value knowledge statement...",
        "Second high-value knowledge statement...",
        "Third high-value knowledge statement..."
    ]
}}

Extract ONLY high-value, persistent knowledge. Return empty list if none found.
"""
            
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1024*8,
                default_response={"statements": []},  # Provide appropriate default response
                max_retries=4  # Increase retry count
            )
            
            # Extract statement list
            statements = response.get("statements", [])
            
            # Convert to semantic memory
            semantic_memories = self._convert_to_semantic_memories(
                user_id, statements, episode
            )
            
            logger.info(f"Cold start extracted {len(semantic_memories)} initial knowledge statements")
            return semantic_memories
            
        except Exception as e:
            logger.error(f"Error in cold start extraction: {e}")
            return []
    
    def _retrieve_relevant_statements(
        self,
        episode: Episode,
        existing_statements: List[SemanticMemory]
    ) -> List[str]:
        """
        Retrieve statement knowledge related to new episode
        Optimized version: Use ChromaDB's vector search directly
        
        Args:
            episode: New episode
            existing_statements: Existing statements
            
        Returns:
            List of relevant statement contents
        """
        try:
            if not existing_statements:
                logger.debug("No existing statements for retrieval")
                return []
            
            user_id = episode.user_id
            
            # 优化：使用ChromaDB的向量搜索而不是手动计算
            if self.vector_search and hasattr(self.vector_search, 'search_semantic_memories'):
                logger.debug(f"Using ChromaDB vector search for user {user_id}")
                
                # 构建查询文本
                query_text = f"{episode.title}. {episode.content}"
                
                # 获取配置的参数
                max_statements = getattr(self.config, 'max_statements_for_prediction', 10)
                
                # 使用ChromaDB搜索相关语义记忆
                search_results = self.vector_search.search_semantic_memories(
                    user_id=user_id,
                    query=query_text,
                    top_k=max_statements * 2  # 搜索更多结果以便筛选
                )
                
                # 提取内容并按相似度筛选
                relevant_statements = []
                similarity_threshold = getattr(self.config, 'statement_similarity_threshold', 0.7)
                
                for result in search_results[:max_statements]:
                    if result.get('score', 0) >= similarity_threshold:
                        # 从结果中提取内容
                        content = result.get('content') or result.get('document', '')
                        if content:
                            relevant_statements.append(content)
                
                logger.debug(f"Retrieved {len(relevant_statements)} relevant statements using ChromaDB search")
                return relevant_statements
            
            else:
                # 降级方案：如果没有vector_search，使用简化的随机采样
                logger.warning("No vector search available, using random sampling fallback")
                
                # 随机采样一部分语句，避免处理全部
                sample_size = min(50, len(existing_statements))
                if len(existing_statements) > sample_size:
                    import random
                    sampled_statements = random.sample(existing_statements, sample_size)
                else:
                    sampled_statements = existing_statements
                
                # 简单返回最近的N个语句
                max_statements = getattr(self.config, 'max_statements_for_prediction', 10)
                relevant_statements = [stmt.content for stmt in sampled_statements[-max_statements:]]
                
                logger.debug(f"Retrieved {len(relevant_statements)} recent statements using fallback method")
                return relevant_statements
            
        except Exception as e:
            logger.error(f"Error retrieving relevant statements: {e}")
            # 错误时返回最近的几个语句
            try:
                max_statements = getattr(self.config, 'max_statements_for_prediction', 10)
                return [stmt.content for stmt in existing_statements[-max_statements:]]
            except:
                return []
    
    def _predict_episode(
        self,
        episode_title: str,
        relevant_statements: List[str]
    ) -> str:
        """
        Predict episode content based on title and relevant knowledge
        
        Args:
            episode_title: Episode title
            relevant_statements: Relevant statement knowledge
            
        Returns:
            Predicted episode content
        """
        try:
            # If no relevant knowledge, return basic prediction
            if not relevant_statements:
                return f"Based on the title '{episode_title}', a conversation likely occurred but no specific knowledge is available to predict its content."
            
            # Use LLM to generate prediction
            prompt = self.prompts.get_prediction_prompt(
                episode_title, relevant_statements
            )
            
            prediction = self.llm_client.generate_text_response(
                prompt=prompt,
                temperature=getattr(self.config, 'prediction_temperature', 0.3),
                max_tokens=1024*8
            )
            
            logger.debug(f"Generated prediction for episode: {episode_title[:50]}...")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting episode: {e}")
            return "Unable to generate prediction due to an error."
    

    
    def _extract_knowledge_from_comparison(
        self,
        original_messages: List[Dict[str, Any]],
        predicted_episode: str
    ) -> List[str]:
        """
        Extract knowledge statements directly from comparison of original messages and prediction
        
        Args:
            original_messages: Original conversation message list
            predicted_episode: Predicted episode content
            
        Returns:
            List of knowledge statements
        """
        try:
            # Format original messages
            formatted_messages = []
            for i, msg in enumerate(original_messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                
                if timestamp:
                    formatted_messages.append(f"Message {i} [{timestamp}] - {role}: {content}")
                else:
                    formatted_messages.append(f"Message {i} - {role}: {content}")
            
            messages_text = "\n".join(formatted_messages)
            
            # Build prompt
            prompt = self.prompts.EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT.format(
                original_messages=messages_text,
                predicted_episode=predicted_episode
            )
            
            # Call LLM
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1024*8,
                default_response={"statements": []},
                max_retries=5
            )
            
            statements = response.get("statements", [])
            
            # Filter empty statements
            filtered_statements = [stmt for stmt in statements if stmt and stmt.strip()]
            
            logger.info(f"Extracted {len(filtered_statements)} knowledge statements from comparison")
            
            return filtered_statements
            
        except Exception as e:
            logger.error(f"Error extracting knowledge from comparison: {e}")
            return []
    

    
    def _convert_to_semantic_memories(
        self,
        user_id: str,
        statements: List[str],
        source_episode: Episode
    ) -> List[SemanticMemory]:
        """
        Convert refined statements to semantic memory objects
        
        Args:
            user_id: User ID
            statements: List of statements
            source_episode: Source episode
            
        Returns:
            List of semantic memory objects
        """
        semantic_memories = []
        
        for statement in statements:
            try:
                # Create semantic memory
                memory = SemanticMemory(
                    user_id=user_id,
                    content=statement,
                    knowledge_type="knowledge",  # Use generic type
                    source_episodes=[source_episode.episode_id],
                    confidence=0.9,  # High confidence, learned from actual differences
                    created_at=source_episode.timestamp  # Use actual occurrence time of source episode
                )
                
                # Add metadata
                memory.metadata = {
                    "generation_method": "prediction_correction",
                    "episode_title": source_episode.title
                }
                
                semantic_memories.append(memory)
                
            except Exception as e:
                logger.error(f"Error creating semantic memory for statement: {e}")
                continue
        
        return semantic_memories
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        try:
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            return float(np.dot(vec1_norm, vec2_norm))
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "engine": "prediction_correction_simplified",
            "config": {
                "max_statements_for_prediction": getattr(self.config, 'max_statements_for_prediction', 10),
                "statement_similarity_threshold": getattr(self.config, 'statement_similarity_threshold', 0.7),
                "prediction_temperature": getattr(self.config, 'prediction_temperature', 0.3)
            },
            "phases": [
                "prediction",
                "direct_extraction"
            ]
        } 