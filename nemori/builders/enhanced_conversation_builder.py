"""
Enhanced conversation episode builder with semantic memory integration.

This builder extends the base ConversationEpisodeBuilder to include semantic
knowledge discovery and evolution capabilities.
"""

from typing import Any

from ..core.data_types import ConversationData, SemanticNode, TypedEventData
from ..core.episode import Episode
from ..llm.protocol import LLMProvider
from ..semantic.evolution import SemanticEvolutionManager
from .conversation_builder import ConversationEpisodeBuilder


class EnhancedConversationEpisodeBuilder(ConversationEpisodeBuilder):
    """
    Enhanced conversation builder with semantic memory integration.
    集成语义记忆的增强对话构建器。
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider | None = None,
        semantic_manager: SemanticEvolutionManager | None = None,
        custom_instructions: str | None = None,
        **kwargs
    ):
        super().__init__(llm_provider, custom_instructions, **kwargs)
        self.semantic_manager = semantic_manager
        
    async def build_episode(self, data: TypedEventData, for_owner: str) -> Episode:
        """
        Build episode and process semantic knowledge.
        构建情景并处理语义知识。
        """
        # Build episode using parent method
        # 使用父方法构建情景
        episode = await super().build_episode(data, for_owner)
        
        # Process semantic knowledge if manager available
        # 如果管理器可用则处理语义知识
        if self.semantic_manager and isinstance(data, ConversationData):
            try:
                # Get the original conversation content for semantic analysis
                # 获取原始对话内容进行语义分析
                original_content = data.get_conversation_text(include_timestamps=True)
                
                # Discover and store semantic knowledge
                # 发现并存储语义知识
                semantic_nodes = await self.semantic_manager.process_episode_for_semantics(
                    episode, original_content
                )
                
                # Add semantic metadata to episode
                # 向情景添加语义元数据
                if semantic_nodes:
                    episode.metadata.custom_fields["discovered_semantics"] = len(semantic_nodes)
                    episode.metadata.custom_fields["semantic_node_ids"] = [
                        node.node_id for node in semantic_nodes
                    ]
                    
                    print(f"[EnhancedConversationBuilder] Discovered {len(semantic_nodes)} semantic nodes for episode {episode.episode_id}")
                    
            except Exception as e:
                print(f"[EnhancedConversationBuilder] Warning: Semantic processing failed: {e}")
                # Continue even if semantic processing fails
                
        return episode

    async def get_semantic_context(self, episode_id: str) -> dict[str, Any]:
        """
        Get semantic context for an episode.
        获取情景的语义上下文。
        """
        if not self.semantic_manager:
            return {}
            
        try:
            # Get semantic nodes discovered from this episode
            discovered_nodes = []
            if self.semantic_manager.storage:
                discovered_nodes = await self.semantic_manager.storage.find_semantic_nodes_by_episode(episode_id)
            
            # Get linked semantic nodes
            linked_nodes = []
            if self.semantic_manager.storage:
                linked_nodes = await self.semantic_manager.storage.find_semantic_nodes_by_linked_episode(episode_id)
            
            return {
                "discovered_semantic_nodes": [
                    {"key": node.key, "value": node.value, "confidence": node.confidence}
                    for node in discovered_nodes
                ],
                "linked_semantic_nodes": [
                    {"key": node.key, "value": node.value, "confidence": node.confidence}
                    for node in linked_nodes if node.node_id not in [n.node_id for n in discovered_nodes]
                ],
                "total_semantic_connections": len(discovered_nodes) + len(linked_nodes)
            }
            
        except Exception as e:
            print(f"Error getting semantic context: {e}")
            return {}

    def get_semantic_summary(self, semantic_nodes: list[SemanticNode]) -> str:
        """
        Generate a summary of discovered semantic knowledge.
        生成发现的语义知识摘要。
        """
        if not semantic_nodes:
            return "No semantic knowledge discovered."
        
        knowledge_summary = []
        for node in semantic_nodes[:5]:  # Top 5 nodes
            knowledge_summary.append(f"{node.key}: {node.value[:100]}...")
        
        return f"Discovered {len(semantic_nodes)} pieces of semantic knowledge: " + "; ".join(knowledge_summary)