"""
Nemori - Nature-inspired Episodic Memory for AI

A comprehensive memory framework that transforms various forms of user experiences
into searchable, narrative episodic memories.
"""

__version__ = "0.1.0"
__author__ = "Nemori Team"
__description__ = "Nature-inspired Episodic Memory for AI"

from .core.builders import EpisodeBuilder
from .core.data_types import DataType, RawEventData
from .core.episode import Episode, EpisodeMetadata
from .llm import AnthropicProvider, GeminiProvider, LLMProvider, OpenAIProvider

__all__ = [
    "DataType",
    "RawEventData",
    "Episode",
    "EpisodeMetadata",
    "EpisodeBuilder",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
]
