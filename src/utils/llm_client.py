"""
LLM Client
"""

import openai
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response data model"""
    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str
    response_time: float


class LLMClient:
    """Language model client"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key
            model: Model name
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
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            **kwargs: Other parameters
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs
                )
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    usage=response.usage.model_dump() if response.usage else {},
                    model=response.model,
                    finish_reason=response.choices[0].finish_reason,
                    response_time=response_time
                )
                
            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise e
    
    def generate_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        default_response: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate JSON response
        
        Args:
            prompt: Prompt text
            system_prompt: System prompt text
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            default_response: Default response to return when parsing fails (optional)
            max_retries: Maximum retry count when JSON parsing fails
            
        Returns:
            Parsed JSON object
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
                # Retry loop
        for attempt in range(max_retries):
            try:
                response = self.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        
                # Try to parse JSON
                result = self._extract_json_from_response(response.content, default_response=None)
    
                # If parsing succeeds and is not default response, return result
                if result is not None and not (
                    result.get("reason") == "JSON parsing failed, using default response" and 
                    result.get("topic_summary") == "Parsing error"
                ):
                    return result
                    
                # If this is the last attempt, log warning
                if attempt == max_retries - 1:
                    logger.warning(f"JSON parsing failed, retried {max_retries} times")
                else:
                    logger.info(f"JSON parsing failed, retrying ({attempt + 1}/{max_retries})")
                    
                    # On retry, add additional instruction to messages requesting pure JSON
                    if attempt == 0:
                        messages.append({
                            "role": "assistant", 
                            "content": response.content
                        })
                        messages.append({
                            "role": "user", 
                            "content": "Please provide your response in valid JSON format only, without any additional text or explanation. The JSON should be properly formatted and parseable."
                        })
                    
                    # Slightly increase temperature to get different response
                    temperature = min(temperature + 0.1, 0.5)
                    
            except Exception as e:
                logger.error(f"Error generating JSON response (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed, return default response
                    if default_response is not None:
                        logger.warning(f"All retries failed, returning specified default response")
                        return default_response
                    else:
                        raise e
        
        # All retries failed, return default response
        if default_response is not None:
            logger.warning(f"JSON parsing failed and all retries exhausted, returning default response")
            return default_response
        else:
            # Return generic default response
            return {
                "error": "JSON parsing failed after all retries",
                "attempts": max_retries
            }
    
    def _extract_json_from_response(self, content: str, default_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract JSON from response, supports multiple formats
        
        Args:
            content: Response content
            default_response: Default response to return when parsing fails (optional)
            
        Returns:
            Parsed JSON object, returns None if parsing fails and no default response
        """
        import re
        
        try:
            # Method 1: Direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        try:
            # Method 2: Clean and parse
            cleaned_content = content.strip()
            
            # Remove markdown code block markers
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            
            # Clean whitespace again
            cleaned_content = cleaned_content.strip()
            
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            pass
        
        try:
            # Method 3: Use regex to find JSON
            # Find JSON objects starting with { and ending with }
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        try:
            # Method 4: More complex regex matching, supports nesting
            # Find content between first { and last }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If all methods fail, log detailed error information
        logger.error(f"JSON parsing failed, tried multiple methods but unable to parse")
        logger.error(f"Original response content: {repr(content)}")
        logger.error(f"Response length: {len(content)}")
        logger.error(f"First 100 characters of response: {content[:100]}")
        logger.error(f"Last 100 characters of response: {content[-100:]}")
        
        # If default response provided, return it, otherwise return marker response to let caller know parsing failed
        if default_response is not None:
            logger.warning(f"Returning specified default response: {default_response}")
            return default_response
        else:
            # Return special response indicating parsing failure
            logger.warning("JSON parsing failed, returning marker response")
        return {
            "should_end": False,
            "reason": "JSON parsing failed, using default response",
            "confidence": 0.5,
            "topic_summary": "Parsing error"
        }
    
    def generate_text_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text response
        
        Args:
            prompt: Prompt text
            system_prompt: System prompt text
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.content
    
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
    
    def validate_response(self, response: str, expected_format: str = "json") -> bool:
        """
        Validate response format
        
        Args:
            response: Response content
            expected_format: Expected format
            
        Returns:
            Whether valid
        """
        if expected_format == "json":
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model": self.model,
            "api_key_prefix": self.api_key[:10] + "..." if self.api_key else None,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout
        } 