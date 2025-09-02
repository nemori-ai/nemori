"""
Semantic discovery engine for identifying private domain knowledge.

This module implements the core semantic discovery functionality using differential
analysis between episodic compression and LLM reconstruction capabilities.
"""

import asyncio
import json
from typing import Any, Dict, List

from ..core.data_types import SemanticNode
from ..core.episode import Episode
from ..llm.protocol import LLMProvider


class SemanticDiscoveryEngine:
    """
    Context-aware engine for discovering semantic knowledge through differential analysis.
    通过差分分析发现语义知识的上下文感知引擎。
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        
    async def discover_semantic_knowledge(
        self, 
        episode: Episode, 
        original_content: str,
        context: dict[str, Any] | None = None
    ) -> list[SemanticNode]:
        """
        Discover semantic knowledge using episodic memory as a knowledge mask.
        利用情景记忆作为知识掩码发现语义知识。
        
        Process | 流程:
        1. Use episode as mask to reconstruct original | 使用情景作为掩码重建原始内容
        2. Compare reconstructed vs original | 比较重建与原始内容
        3. Extract knowledge gaps as semantic nodes | 提取知识差距作为语义节点
        """
        try:
            # Step 1: Context-aware reconstruction
            # 步骤1：上下文感知重建
            reconstructed_content = await self._reconstruct_with_context(episode, context or {})
            
            # Step 2: Perform differential analysis
            # 步骤2：执行差分分析
            knowledge_gaps = await self._analyze_knowledge_gaps(
                original=original_content,
                reconstructed=reconstructed_content,
                episode=episode,
                context=context or {}
            )
            
            # Step 3: Create semantic nodes with bidirectional links
            # 步骤3：创建带双向链接的语义节点
            semantic_nodes = []
            for gap in knowledge_gaps:
                node = SemanticNode(
                    owner_id=episode.owner_id,
                    key=gap.get("key", ""),
                    value=gap.get("value", ""),
                    context=gap.get("context", ""),
                    discovery_episode_id=episode.episode_id,
                    linked_episode_ids=[episode.episode_id],  # Initial linking
                    confidence=gap.get("confidence", 1.0),
                    search_keywords=self._extract_keywords(gap)
                )
                semantic_nodes.append(node)
                
            return semantic_nodes
            
        except Exception as e:
            print(f"Error in semantic discovery: {e}")
            return []
    
    async def _reconstruct_with_context(self, episode: Episode, context: dict[str, Any]) -> str:
        """
        Reconstruct original conversation from episodic summary using context.
        利用上下文从情景摘要重建原始对话。
        """
        reconstruction_prompt = self._build_reconstruction_prompt(episode, context)
        
        try:
            response = await self.llm_provider.generate_async(
                prompt=reconstruction_prompt,
                max_tokens=1500,
                temperature=0.3
            )
            return response.strip()
        except Exception as e:
            print(f"Error in reconstruction: {e}")
            return ""
    
    async def _analyze_knowledge_gaps(
        self, 
        original: str, 
        reconstructed: str, 
        episode: Episode,
        context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Analyze differences between original and reconstructed content to identify knowledge gaps.
        分析原始内容与重建内容的差异以识别知识差距。
        """
        analysis_prompt = self._build_knowledge_gap_analysis_prompt(
            original, reconstructed, episode, context
        )
        
        try:
            response = await self.llm_provider.generate_async(
                prompt=analysis_prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(response.strip())
                return analysis_result.get("knowledge_gaps", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, extract gaps manually
                return self._extract_gaps_from_text(response)
                
        except Exception as e:
            print(f"Error in knowledge gap analysis: {e}")
            return []
    
    def _build_reconstruction_prompt(self, episode: Episode, context: dict[str, Any]) -> str:
        """Build prompt for content reconstruction."""
        related_context = ""
        if context:
            related_context = f"\n\nRelated context for reference:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        
        return f"""You are an expert at reconstructing original conversations from episodic summaries.
您是从情景摘要重建原始对话的专家。

Given this episodic memory:
给定以下情景记忆：
Title: {episode.title}
Summary: {episode.summary}
Content: {episode.content}{related_context}

Please reconstruct what the original conversation might have looked like, using your general world knowledge.
请使用您的通用世界知识重建原始对话可能的样子。

Important guidelines | 重要准则:
1. Use only common knowledge that a typical LLM would know | 只使用典型大语言模型会知道的常识
2. Make reasonable assumptions for missing details | 对缺失细节做合理假设
3. Focus on factual reconstruction, not creative interpretation | 专注于事实重建，而非创意解释
4. Maintain the same conversation structure and flow | 保持相同的对话结构和流程

Return the reconstructed conversation:
返回重建的对话："""

    def _build_knowledge_gap_analysis_prompt(
        self, 
        original: str, 
        reconstructed: str, 
        episode: Episode,
        context: dict[str, Any]
    ) -> str:
        """Build prompt for knowledge gap analysis."""
        return f"""You are an expert at identifying private domain knowledge gaps.
您是识别私域知识差距的专家。

Original content | 原始内容:
{original}

Reconstructed content (using general LLM knowledge) | 重建内容（使用通用大语言模型知识）:
{reconstructed}

Please identify specific pieces of information that exist in the original but are missing or incorrectly assumed in the reconstruction. These represent private domain knowledge.
请识别原始内容中存在但在重建中缺失或错误假设的具体信息。这些代表私域知识。

Focus on | 关注:
1. Proper names, project names, specific terminology | 专有名词、项目名称、特定术语
2. Personal preferences, habits, and characteristics | 个人偏好、习惯和特征
3. Specific facts, dates, numbers that differ | 具体的事实、日期、数字差异
4. Context-specific meanings and interpretations | 上下文特定的含义和解释

Return your analysis in JSON format:
以 JSON 格式返回分析：
{{
    "knowledge_gaps": [
        {{
            "key": "specific identifier or topic",
            "value": "the correct private knowledge",
            "context": "surrounding context from original",
            "gap_type": "proper_noun|personal_fact|specific_detail|contextual_meaning",
            "confidence": 0.0-1.0
        }}
    ]
}}"""

    def _extract_gaps_from_text(self, response: str) -> list[dict[str, Any]]:
        """Extract knowledge gaps from text response when JSON parsing fails."""
        gaps = []
        lines = response.split('\n')
        
        current_gap = {}
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                # New gap item
                if current_gap:
                    gaps.append(current_gap)
                current_gap = {
                    "key": line[2:].split(':')[0] if ':' in line else line[2:],
                    "value": line[2:].split(':', 1)[1].strip() if ':' in line else "",
                    "context": "",
                    "gap_type": "specific_detail",
                    "confidence": 0.7
                }
        
        if current_gap:
            gaps.append(current_gap)
        
        return gaps
    
    def _extract_keywords(self, gap: dict[str, Any]) -> list[str]:
        """Extract search keywords from a knowledge gap."""
        keywords = []
        
        # Extract from key
        key_words = gap.get("key", "").split()
        keywords.extend([word.lower() for word in key_words if len(word) > 2])
        
        # Extract from value
        value_words = gap.get("value", "").split()[:5]  # First 5 words
        keywords.extend([word.lower() for word in value_words if len(word) > 2])
        
        # Remove duplicates and common stop words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [kw for kw in set(keywords) if kw not in stop_words]
        
        return keywords[:10]  # Limit to 10 keywords


class ContextAwareSemanticDiscoveryEngine(SemanticDiscoveryEngine):
    """
    Enhanced semantic discovery engine with context awareness.
    具有上下文感知能力的增强语义发现引擎。
    """
    
    def __init__(self, llm_provider: LLMProvider, retrieval_service=None):
        super().__init__(llm_provider)
        self.retrieval_service = retrieval_service
        
    async def discover_semantic_knowledge(
        self, 
        episode: Episode, 
        original_content: str,
        context: dict[str, Any] | None = None
    ) -> list[SemanticNode]:
        """
        Enhanced discovery with context from related memories.
        利用相关记忆的上下文进行增强发现。
        """
        # Step 1: Gather context from related memories
        # 步骤1：从相关记忆中收集上下文
        enhanced_context = await self._gather_discovery_context(episode, context or {})
        
        # Step 2: Use parent implementation with enhanced context
        # 步骤2：使用增强上下文调用父类实现
        return await super().discover_semantic_knowledge(episode, original_content, enhanced_context)
    
    async def _gather_discovery_context(
        self, 
        episode: Episode, 
        base_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Gather related semantic memories and historical episodes for context.
        收集相关语义记忆和历史情景作为上下文。
        """
        context = base_context.copy()
        
        if not self.retrieval_service:
            return context
        
        try:
            # Search for related semantic memories (if semantic retrieval is available)
            # 搜索相关语义记忆（如果语义检索可用）
            if hasattr(self.retrieval_service, 'search_semantic_memories'):
                related_semantics = await self.retrieval_service.search_semantic_memories(
                    owner_id=episode.owner_id,
                    query=f"{episode.title} {episode.summary}",
                    limit=5
                )
                context["related_semantic_memories"] = [
                    {"key": node.key, "value": node.value, "context": node.context}
                    for node in related_semantics
                ]
            
            # Search for related historical episodes
            # 搜索相关历史情景
            if hasattr(self.retrieval_service, 'search_episodes'):
                related_episodes = await self.retrieval_service.search_episodes(
                    owner_id=episode.owner_id,
                    query=episode.content,
                    limit=3
                )
                context["related_historical_episodes"] = [
                    {"title": ep.title, "summary": ep.summary, "timestamp": ep.temporal_info.timestamp.isoformat()}
                    for ep in related_episodes.episodes if ep.episode_id != episode.episode_id
                ]
                
        except Exception as e:
            print(f"Warning: Failed to gather context: {e}")
        
        context["current_episode"] = {
            "title": episode.title,
            "summary": episode.summary,
            "timestamp": episode.temporal_info.timestamp.isoformat()
        }
        
        return context