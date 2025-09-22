"""
Boundary Detector
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from ..models import MessageBuffer, Message
from ..utils import LLMClient
from ..config import MemoryConfig
from ..generation.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Intelligent boundary detector for conversation episodes"""
    
    def __init__(self, llm_client: LLMClient, config: MemoryConfig):
        """
        Initialize boundary detector
        
        Args:
            llm_client: LLM client for boundary detection
            config: Memory system configuration
        """
        self.llm_client = llm_client
        self.config = config
        self.prompts = PromptTemplates()
        
        logger.info("Boundary detector initialized")
    
    def detect_boundary(
        self, 
        buffer: MessageBuffer, 
        new_messages: List[Message]
    ) -> Dict[str, Any]:
        """
        Detect if new messages should trigger episode boundary using only intelligent LLM-based detection
        
        Args:
            buffer: Current message buffer
            new_messages: New messages to analyze
            
        Returns:
            Boundary detection result
        """
        try:
            # Check if buffer is empty (first message) - this is the only essential quick check
            if buffer.is_empty():
                return {
                    "should_end": False,
                    "reason": "First message in buffer",
                    "confidence": 1.0,
                    "topic_summary": "",
                    "detection_method": "first_message"
                }
            
            # Use only intelligent boundary detection with LLM
            return self._intelligent_boundary_detection(buffer, new_messages)
                
        except Exception as e:
            logger.error(f"Error in boundary detection: {e}")
            return {
                "should_end": False,
                "reason": "Error in boundary detection",
                "confidence": 0.0,
                "topic_summary": "",
                "detection_method": "error"
            }
    
    
    
    
    def _intelligent_boundary_detection(
        self, 
        buffer: MessageBuffer, 
        new_messages: List[Message]
    ) -> Dict[str, Any]:
        """
        Use LLM for intelligent boundary detection
        
        Args:
            buffer: Current message buffer
            new_messages: New messages to analyze
            
        Returns:
            Boundary detection result
        """
        try:
            # Format conversation history
            conversation_history = buffer.get_conversation_text()
            new_messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in new_messages])
            
            # Generate prompt
            prompt = self.prompts.get_boundary_detection_prompt(
                conversation_history=conversation_history,
                new_messages=new_messages_text
            )
            
            # Call LLM
            response = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024*8
            )
            
            # Validate response
            if not isinstance(response, dict):
                raise ValueError("Invalid response format")
            
            # Ensure required fields
            result = {
                "should_end": response.get("should_end", False),
                "reason": response.get("reason", "LLM boundary detection"),
                "confidence": min(max(response.get("confidence", 0.5), 0.0), 1.0),
                "topic_summary": response.get("topic_summary", ""),
                "detection_method": "llm_intelligent"
            }
            
            # ⚠️ Note: Confidence threshold check removed - only use should_end from LLM
            # This matches the behavior of the old system which ignores confidence
            
            logger.debug(f"LLM boundary detection result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in intelligent boundary detection: {e}")
            # Fallback to no boundary detection on error
            return {
                "should_end": False,
                "reason": f"Error in intelligent boundary detection: {e}",
                "confidence": 0.0,
                "topic_summary": "",
                "detection_method": "error_fallback"
            }
    
    
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """
        Get boundary detector statistics
        
        Returns:
            Detector statistics
        """
        return {
            "config": {
                "intelligent_detection_only": True,
                "boundary_confidence_threshold": getattr(self.config, 'boundary_confidence_threshold', 0.5)
            },
            "llm_model": self.llm_client.model if hasattr(self.llm_client, 'model') else "unknown",
            "detection_method": "intelligent_only"
        } 