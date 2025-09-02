"""
Nemori - Nature-inspired Episodic Memory for AI

A comprehensive memory framework that transforms various forms of user experiences
into searchable, narrative episodic memories with semantic knowledge discovery.
"""

__version__ = "0.1.0"
__author__ = "Nemori Team"
__description__ = "Nature-inspired Episodic Memory for AI with Semantic Memory"

from .core.builders import EpisodeBuilder
from .core.data_types import DataType, RawEventData, SemanticNode, SemanticRelationship
from .core.episode import Episode, EpisodeMetadata
from .episode_manager import EpisodeManager
from .llm import AnthropicProvider, GeminiProvider, LLMProvider, OpenAIProvider
from .semantic import (
    ContextAwareSemanticDiscoveryEngine,
    SemanticDiscoveryEngine,
    SemanticEvolutionManager,
    UnifiedRetrievalService,
)
from .builders import EnhancedConversationEpisodeBuilder

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
    "EpisodeManager",
    # Semantic memory components
    "SemanticNode",
    "SemanticRelationship", 
    "SemanticDiscoveryEngine",
    "ContextAwareSemanticDiscoveryEngine",
    "SemanticEvolutionManager",
    "UnifiedRetrievalService",
    "EnhancedConversationEpisodeBuilder",
]
