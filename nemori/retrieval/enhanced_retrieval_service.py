from .service import RetrievalService
from .enhanced_retrieval_service import SemanticRetrievalService
from typing import Any


class EnhancedRetrievalService:
    """
    Enhanced retrieval combining episodic and semantic memory.
    结合情景和语义记忆的增强检索服务。
    """

    def __init__(self, episodic_retrieval: RetrievalService, semantic_retrieval: SemanticRetrievalService):
        self.episodic_retrieval = episodic_retrieval
        self.semantic_retrieval = semantic_retrieval

    async def enhanced_query(self, owner_id: str, query: str, include_semantic: bool = True) -> dict[str, Any]:
        """
        Enhanced query combining episodic and semantic results.
        结合情景和语义结果的增强查询。
        """
        results = {"episodes": await self.episodic_retrieval.search_episodes(owner_id, query), "semantic_knowledge": []}

        if include_semantic:
            results["semantic_knowledge"] = await self.semantic_retrieval.query_semantic_knowledge(owner_id, query)

        return results
