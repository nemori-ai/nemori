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
from ..retrieval import RetrievalStrategy
from .unified_retrieval import UnifiedRetrievalService


class ContextAwareSemanticDiscoveryEngine:
    """
    Context-aware engine for discovering semantic knowledge through differential analysis.
    通过差分分析发现语义知识的上下文感知引擎。
    """

    def __init__(self, llm_provider: LLMProvider, retrieval_service: UnifiedRetrievalService):
        self.llm_provider = llm_provider
        self.retrieval_service = retrieval_service
        self.topk = 10

    async def discover_semantic_knowledge(self, episode: Episode, original_content: str) -> list[SemanticNode]:
        """
        Discover semantic knowledge with context from related memories.
        利用相关记忆的上下文发现语义知识。

        Process | 流程:
        1. Gather related semantic memories and historical episodes | 收集相关语义记忆和历史情景
        2. Use episode as mask to reconstruct original | 使用情景作为掩码重建原始内容
        3. Compare reconstructed vs original with context | 结合上下文比较重建与原始内容
        4. Extract knowledge gaps as semantic nodes | 提取知识差距作为语义节点
        """
        # Step 1: Gather context from related memories
        # 步骤1：从相关记忆中收集上下文
        context = await self._gather_discovery_context(episode)
        #context = {}
        # Step 2: Context-aware reconstruction
        # 步骤2：上下文感知重建
        reconstructed_content = await self._reconstruct_with_context(episode, context)
        print("reconstructed_content:",reconstructed_content)
        # Step 3: Perform differential analysis
        # 步骤3：执行差分分析
        knowledge_gaps = await self._analyze_knowledge_gaps(
            original=original_content, reconstructed=reconstructed_content, episode=episode, context=context
        )

        # Step 4: Create semantic nodes with bidirectional links
        # 步骤4：创建带双向链接的语义节点
        
        # Clean owner_id to remove segment suffix for consistent indexing
        # 清理owner_id，去掉segment后缀以确保索引一致性
        clean_owner_id = self._clean_owner_id(episode.owner_id)
        print(f"   🔧 Cleaning owner_id: '{episode.owner_id}' -> '{clean_owner_id}'")
        
        semantic_nodes = []
        for gap in knowledge_gaps:
            node = SemanticNode(
                owner_id=clean_owner_id,  # Use cleaned owner_id
                key=gap.get("key", ""),
                value=gap.get("value", ""),
                context=gap.get("context", ""),
                discovery_episode_id=episode.episode_id,
                linked_episode_ids=[episode.episode_id],  # Initial linking
                confidence=gap.get("confidence", 1.0),
                search_keywords=self._extract_keywords(gap),
            )
            semantic_nodes.append(node)

        return semantic_nodes

    async def _gather_discovery_context(self, episode: Episode) -> dict[str, Any]:
        """
        Gather related semantic memories and historical episodes for context.
        收集相关语义记忆和历史情景作为上下文。
        """
        print(f"🔍 Gathering discovery context for episode {episode.episode_id} (owner: {episode.owner_id})")
        
        # Search for related semantic memories
        # 搜索相关语义记忆
        query_text = f"{episode.title} {episode.content}"
        print(f"   🧠 Searching semantic memories with query: '{query_text[:100]}...'")
        related_semantics = await self.retrieval_service.search_semantic_memories(
            owner_id=episode.owner_id, query=query_text, limit=5 #self.topk#self.topk * 2
        )
        print(f"   ✅ Found {len(related_semantics)} related semantic memories")

        # Search for related historical episodes (excluding current episode)
        # 搜索相关历史情景（排除当前情景）
        episode_query = f"{episode.content}"
        print(f"   📚 Searching episodic memories with query: '{episode_query[:100]}...'")
        
        related_episodes_result = await self.retrieval_service.search_episodic_memories(
            owner_id=episode.owner_id, query=episode_query, limit=3 #self.topk // 2# Increase limit to get more candidates
        )

        # Handle the search result properly - it might be EpisodeSearchResult or direct episodes list
        if hasattr(related_episodes_result, 'episodes'):
            related_episodes = related_episodes_result.episodes
        else:
            related_episodes = related_episodes_result if related_episodes_result else []

        # # Only filter out current episode if we actually found some episodes
        # # 只有在实际找到episodes的情况下才过滤当前episode
        # filtered_episodes = []
        # if related_episodes:
        #     for ep in related_episodes:
        #         if ep.episode_id != episode.episode_id:
        #             filtered_episodes.append(ep)
        #     print(f"   ✅ Found {len(related_episodes)} total episodes, {len(filtered_episodes)} historical episodes (excluding current)")
            
        #     # If we filtered out everything, it means we only found the current episode
        #     if len(related_episodes) > 0 and len(filtered_episodes) == 0:
        #         print(f"   ⚠️ Only found current episode in search results - no historical episodes available")
        # else:
        #     print(f"   ✅ Found {len(related_episodes)} episodes (none to filter)")
        
        # # Log some details about found episodes for debugging
        # if filtered_episodes:
        #     print(f"   📝 Sample historical episodes:")
        #     for i, ep in enumerate(filtered_episodes[:2]):  # Show first 2
        #         content_preview = ep.content[:50] + "..." if len(ep.content) > 50 else ep.content
        #         print(f"      {i+1}. {ep.episode_id}: {content_preview}")
        # else:
        #     print(f"   ⚠️ No historical episodes available for context")

        return {
            "related_semantic_memories": related_semantics,
            "related_historical_episodes": related_episodes,
            "current_episode": episode,
        }

    async def _reconstruct_with_context(self, episode: Episode, context: dict[str, Any]) -> str:
        """
        Reconstruct original conversation from episodic summary using context.
        利用上下文从情景摘要重建原始对话。
        """
        reconstruction_prompt = self._build_reconstruction_prompt(episode, context)
        print("reconstruction_prompt:",reconstruction_prompt)
        response = await self.llm_provider.generate(prompt=reconstruction_prompt, temperature=0.3)
        return response.strip()

    async def _analyze_knowledge_gaps(
        self, original: str, reconstructed: str, episode: Episode, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Analyze differences between original and reconstructed content to identify knowledge gaps.
        分析原始内容与重建内容的差异以识别知识差距。
        """
        analysis_prompt = self._build_knowledge_gap_analysis_prompt(original, reconstructed, episode, context)
        print("analysis_prompt:",analysis_prompt)
        response = await self.llm_provider.generate(prompt=analysis_prompt, temperature=0.1)
        print("analysis_gaps:",response)
        #print(self.llm_provider.model,response)
        # 提取 JSON 字符串部分
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("响应中未找到有效的 JSON 部分")
        judge_string = response[start_idx:end_idx]
        analysis_result = json.loads(judge_string.strip())
        return analysis_result.get("knowledge_gaps", [])

    def _build_reconstruction_prompt(self, episode: Episode, context: dict[str, Any]) -> str:
        """Build prompt for content reconstruction."""
        related_context = ""
        if context:
            # Convert context to JSON-serializable format
            serializable_context = {
                "related_semantic_memories": [
                    {"key": node.key, "value": node.value, "context": node.context} for node in context.get("related_semantic_memories", [])
                ],
                "related_episodes_memories": [
                    {"title": ep.title, "content": ep.content} for ep in context.get("related_historical_episodes", [])
                ],
                "current_episode": (
                    {"title": context["current_episode"].title, "content": context["current_episode"].content}
                    if "current_episode" in context
                    else None
                ),
            }
            related_context = (
                f"\n{json.dumps(serializable_context, indent=2, ensure_ascii=False)}\n"
            )
#         return f"""You are an expert at reconstructing original conversations from episodic summaries.
# 您是从情景摘要重建原始对话的专家。

# Given this related episodes memories and related semantic memories:
# 给定以下相关的情景记忆和语义记忆：
# {related_context}

# Please reconstruct what the original conversation might have looked like, using your general world knowledge.
# 请使用您的通用世界知识重建原始对话可能的样子。

# Important guidelines | 重要准则:
# 1. Use only common knowledge that a typical LLM would know | 只使用典型大语言模型会知道的常识
# 2. Make reasonable assumptions for missing details | 对缺失细节做合理假设
# 3. Focus on factual reconstruction, not creative interpretation | 专注于事实重建，而非创意解释
# 4. Maintain the same conversation structure and flow | 保持相同的对话结构和流程

# Return the reconstructed conversation:
# 返回重建的对话："""
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
        self, original: str, reconstructed: str, episode: Episode, context: dict[str, Any]
    ) -> str:
        """Build prompt for knowledge gap analysis."""
#         return f"""
# You are a meticulous, high-fidelity knowledge analyst. Your mission is to perform a detailed comparison between an 'Original Content' block, which represents private domain truth, and a 'Reconstructed Content' block, which is an attempt to summarize or infer that truth.

# 您是一名一丝不苟的高保真知识分析师。您的任务是详细比较“原始内容”（代表私域真理）和“重建内容”（尝试总结或推断该真理）之间的差异。

# Your goal is to precisely identify pieces of private, non-public knowledge that are present in the original but are either missing, incorrectly stated, over-generalized, or inferred without sufficient evidence in the reconstruction.

# 您的目标是精确识别出原始内容中存在的，但在重建内容中缺失、陈述错误、过度泛化或在证据不足的情况下被推断出来的私域、非公开知识。

# Original content | 原始内容:
# {original}

# Reconstructed content | 重建内容:
# {reconstructed}
# Please identify specific pieces of information that exist in the original but are missing or incorrectly assumed in the reconstruction. These represent private domain knowledge.
# 请识别原始内容中存在但在重建中缺失或错误假设的具体信息。这些代表私域知识。

# Focus on | 关注:
# 1. Proper names, project names, specific terminology | 专有名词、项目名称、特定术语
# 2. Personal preferences, habits, and characteristics | 个人偏好、习惯和特征
# 3. Specific facts, dates, numbers that differ | 具体的事实、日期、数字差异
# 4. Context-specific meanings and interpretations | 上下文特定的含义和解释
# 53.  Focus on the loss of fidelity and the introduction of assumptions. | 专注于保真度的损失和假设的引入。

# Return your analysis in JSON format:
# 以 JSON 格式返回分析：
# {{
#     "knowledge_gaps": [
#         {{  
#             "analysis": "A brief explanation of WHY this is a knowledge gap. Explain the nature of the mismatch (e.g., generalization, assumption, factual error).",
#             "key": "specific identifier or topic",
#             "value": "the correct private knowledge",
#             "context": "surrounding context from original",
#             "gap_type": "proper_noun|personal_fact|specific_detail|contextual_meaning",
#             "confidence": 0.0-1.0
#         }}
#     ]
# }}"""
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
        lines = response.split("\n")

        current_gap = {}
        for line in lines:
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                # New gap item
                if current_gap:
                    gaps.append(current_gap)
                current_gap = {
                    "key": line[2:].split(":")[0] if ":" in line else line[2:],
                    "value": line[2:].split(":", 1)[1].strip() if ":" in line else "",
                    "context": "",
                    "gap_type": "specific_detail",
                    "confidence": 0.7,
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

    def _clean_owner_id(self, owner_id: str) -> str:
        """
        Clean owner_id by removing segment suffix (_segment_X_Y) for consistent indexing.
        清理owner_id，去掉segment后缀以确保索引一致性。
        
        Examples:
        - "张三_0_segment_0_5" -> "张三_0"
        - "李四_1_segment_2_3" -> "李四_1"  
        - "王博士_0" -> "王博士_0" (no change)
        """
        import re
        
        # Remove _segment_X_Y pattern
        cleaned = re.sub(r'_segment_\d+_\d+$', '', owner_id)
        
        return cleaned
