"""
Semantic memory package for Nemori.

This package provides semantic memory capabilities through differential analysis
of episodic memory compression, discovering private domain knowledge.
"""

from .discovery import SemanticDiscoveryEngine, ContextAwareSemanticDiscoveryEngine
from .evolution import SemanticEvolutionManager
from .unified_retrieval import UnifiedRetrievalService

__all__ = [
    "SemanticDiscoveryEngine",
    "ContextAwareSemanticDiscoveryEngine", 
    "SemanticEvolutionManager",
    "UnifiedRetrievalService",
]