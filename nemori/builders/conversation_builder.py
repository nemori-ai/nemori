"""
Conversation Episode Builder for Nemori.

This builder specializes in transforming conversation data into episodic memories,
implementing intelligent boundary detection and LLM-powered episode generation
to create coherent narrative memories from conversational exchanges.
"""

import json
import re
import tiktoken
from datetime import datetime
from typing import Any

from ..core.builders import EpisodeBuilder
from ..core.data_types import ConversationData, DataType, TypedEventData
from ..core.episode import EpisodeLevel, EpisodeMetadata, EpisodeType
from ..llm.protocol import LLMProvider

# Prompts for LLM-based conversation processing
DEFAULT_CUSTOM_INSTRUCTIONS = """
Follow these principles when generating episodic memories:
1. Each episode should be a complete, independent story or event
2. Preserve all important information including names, time, location, emotions, etc.
3. Use declarative language to describe episodes, not dialogue format
4. Highlight key information and emotional changes
5. Ensure episode content is easy to retrieve later
"""

BOUNDARY_DETECTION_PROMPT = """
You are an episodic memory boundary detection expert. You need to determine if the newly added dialogue should end the current episode and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Please carefully analyze the following aspects to determine if a new episode should begin:

1. **Substantive Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different substantive topic with meaningful content?
   - Is there a shift from one specific event/experience to another distinct event/experience?
   - Has the conversation moved from one meaningful question to an unrelated new question?

2. **Intent and Purpose Transition**:
   - Has the fundamental purpose of the conversation changed significantly?
   - Has the core question or issue of the current topic been fully resolved and a new substantial topic begun?

3. **Meaningful Content Assessment**:
   - **IMPORTANT**: Ignore pure greetings, small talk, transition phrases, and social pleasantries
   - Focus only on content that would be memorable and worth recalling later
   - Consider: Would a person remember this as part of the main conversation topic or as a separate discussion?

4. **Structural and Temporal Signals**:
   - Are there explicit topic transition phrases introducing substantial new content?
   - Are there clear concluding statements followed by genuinely new topics?
   - Is there a significant time gap between messages?

5. **Content Relevance and Independence**:
   - How related is the new substantive content to the previous meaningful discussion?
   - Does it involve completely different events, experiences, or substantial topics?

6. **Value Completion or Insight Achievement** (High Priority):  
   - Does the newly added message (or sequence) represent a **self-contained meaningful conclusion**, such as:  
   - A clear decision made ("I've decided to quit my job")  
   - A personal insight or realization ("Oh! So that’s why I’ve been feeling anxious…")  
   - Completion of a reflective process ("After thinking it through, I realize I need to set boundaries")  
   - Resolution of an emotional arc (from confusion → clarity, distress → calm)  
   - A concrete plan or commitment ("I’ll call my mom tomorrow and talk to her")  
   - Even if the topic hasn’t changed, **if the new message adds significant closure, insight, or actionable value that stands as a memorable mental event**, it should end the current episode.
   
**Special Rules for Common Patterns**:
- **Greetings + Topic**: "Hey!" followed by actual content should be ONE episode
- **Transition Phrases**: "By the way", "Oh, also", "Speaking of which" usually continue the same episode unless introducing major topic shifts
- **Social Closures and Farewells**: "Thanks!", "Take care!", "Talk to you soon!", "I'm off to go...", "See you later!" should continue the current episode as natural conversation endings
- **Supportive Responses**: Brief encouragement or acknowledgment should usually continue the current episode

Decision Principles:
- **Prioritize meaningful content**: Each episode should contain substantive, memorable content
- **Ignore social formalities**: Don't split on greetings, pleasantries, brief transitions, or conversation closures
- **Treat closures as episode endings**: Messages that announce departure ("I'm off to go...", "Talk to you soon!") or provide closure ("Thanks!", "Take care!") should stay with the current episode as natural endings
- **Consider time gaps**: Long time gaps (hours or days) strongly suggest new episodes, while short gaps (minutes) usually indicate continuing conversation
- **Episodic memory focus**: Think about what a person would naturally group together when recalling this conversation
- **Reasonable episode length**: Aim for episodes with 3-20 meaningful exchanges
- **When in doubt, consider context**: If unsure, keep related content together rather than over-splitting

- **Value closure matters**: An episode should end not only on topic shift, but also when a meaningful insight, decision, or emotional resolution is achieved—even if the conversation stops there.  
- **Last meaningful utterance deserves preservation**: If the conversation may end after this turn, and the new message contains high episodic value (e.g., insight, decision, resolution), treat it as an episode boundary to ensure it is captured.

Please return your judgment in JSON format:
{{
    "reasoning": "One sentence summary of your reasoning process",
    "should_end": true/false,
    "confidence": 0.0-1.0,
    "topic_summary": "If should_end = true, summarize the core meaningful topic of the current episode, otherwise leave it blank"
}}

Note:
- If conversation history is empty, this is the first message, return false
- Focus on episodic memory principles: what would people naturally remember as distinct experiences?
- Each episode should contain substantive content that stands alone as a meaningful memory unit
"""


BOUNDARY_DETECTION_PROMPT = """You are an episodic memory boundary detection expert. You need to determine if the newly added dialogue should end the current episode and start a new one.

Current conversation history:
{conversation_history}

Time gap information:
{time_gap_info}

Newly added messages:
{new_messages}

Please carefully analyze the following aspects to determine if a new episode should begin. Your primary goal is to identify self-contained, memorable units of interaction.

**Core Decision Criteria (Evaluate in this order of priority):**

1.  **Value Completion & Task Resolution (Highest Priority):**
    *   Has the user's initial question, command, or problem been **fully and comprehensively resolved** by the new messages?
    *   Does the new message provide a complete, actionable, and conclusive answer that fulfills the user's stated intent?
    *   **Crucially, if a user asks a question and the new message provides a detailed, definitive answer, this "Question-Answer" pair should be treated as a complete and memorable episode, even if no new topic is introduced.** This signifies the achievement of a specific goal.

2.  **Substantive Topic Change:**
    *   Do the new messages introduce a completely different substantive topic with meaningful content?
    *   Is there a clear shift from one specific event, experience, or informational query to a distinctly unrelated one?

3.  **Intent and Purpose Transition:**
    *   Has the fundamental purpose of the conversation shifted? For example, from information gathering to social chatting, or from planning to reflecting.

4.  **Meaningful Content Assessment:**
    *   **Focus on Memorable Units:** An episode should represent a coherent "chunk" of memory. Ask yourself: Would a person recall this interaction as a single, complete event (e.g., "the time I asked about gold prices and got the full breakdown")?
    *   **Ignore trivialities:** Do not segment based on greetings, social pleasantries, or simple confirmations. Focus on the core substance.

5.  **Structural and Temporal Signals:**
    *   Are there explicit topic transition phrases introducing substantial new content?
    *   Is there a significant time gap that naturally suggests a new interaction context?

**Special Rules for Common Patterns:**

*   **Query-Response Closure:** A detailed, comprehensive response that fully answers a user's query **is a strong signal to end the episode**. The value has been delivered, and the cognitive "task" is complete.
*   **Greetings + Topic:** "Hey!" followed by the start of a query is part of the SAME episode.
*   **Social Closures:** "Thanks!", "Talk to you soon!", "I'm heading out" are natural endings and should remain within the current episode.
*   **Supportive Responses:** Brief acknowledgments ("Got it," "Okay") typically continue the current episode unless followed by a major topic shift.

**Decision Principles:**

*   **Prioritize Value Delivery:** The completion of a user's goal or the delivery of a complete piece of information is a primary reason to end an episode. Each episode should feel "complete" in its purpose.
*   **Think in "Memory Chunks":** Structure episodes as a person would naturally recall them. A successfully answered question is a perfect example of a memorable chunk.
*   **Don't Wait for a Topic Change:** If a task is finished (like providing a detailed price list), the episode can end. The next message, even if it's a "thank you," might logically start a new, albeit short, social-closure episode.
*   **When in doubt, favor closure:** If a segment feels self-contained and valuable on its own, it's better to mark it as a complete episode rather than merging it with a potentially unrelated future topic.

Please return your judgment in JSON format:
{{
    "reasoning": "One sentence summary of your reasoning process",
    "should_end": true/false,
    "confidence": 0.0-1.0,
    "topic_summary": "If should_end = true, summarize the core meaningful topic of the current episode, otherwise leave it blank"
}}

Note:
- If conversation history is empty, this is the first message, return false."""
EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation start time: {conversation_start_time}
Conversation content:
{conversation}

Custom instructions:
{custom_instructions}

IMPORTANT TIME HANDLING:
- Use the provided "Conversation start time" as the exact time when this conversation/episode began
- When the conversation mentions relative times (e.g., "yesterday", "last week"), preserve both the original relative expression AND calculate the absolute date
- Format time references as: "original relative time (absolute date)" - e.g., "last week (May 7, 2023)"
- This dual format supports both absolute and relative time-based questions
- All absolute time calculations should be based on the provided start time

Please generate a structured episodic memory and return only a JSON object containing the following two fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A detailed factual record of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a chronological account of what actually happened, focusing on observable actions and direct statements rather than interpretive conclusions. Use the provided conversation start time as the base time for this episode."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search, especially concrete activities, places, and objects.
7. For time references, use the dual format: "relative time (absolute date)" to support different question types.
8. When describing decisions or actions, naturally include the reasoning or motivation behind them.
9. Use specific names consistently rather than pronouns to avoid ambiguity in retrieval.

Example:
If the conversation start time is "March 14, 2024 (Thursday) at 3:00 PM UTC" and the conversation is about Caroline planning to go hiking:
{{
    "title": "Caroline's Mount Rainier Hiking Plan March 14, 2024: Weekend Adventure Planning Session",
    "content": "On March 14, 2024 at 3:00 PM UTC, Caroline expressed interest in going hiking this weekend (March 16-17, 2024) and sought advice. Caroline particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear by Melanie, Caroline received suggestions including hiking boots, warm clothing because it's cold at the summit, a flashlight for the pre-dawn start, water, and high-energy food. Caroline decided to leave early on Saturday morning (March 16, 2024) to catch the sunrise because Caroline wanted to experience the full beauty of the mountain. Caroline planned to invite friends for the adventure, showing Caroline's preference for shared experiences. Caroline was very excited about the trip, hoping to connect with nature and take a break from work stress."
}}

Return only the JSON object, do not add any other text:
"""

EPISODE_GENERATION_PROMPT = """
你是一位情景记忆生成专家。请将以下对话转换为一段情景记忆。

对话开始时间: {conversation_start_time}
对话内容:
{conversation}

自定义指令:
{custom_instructions}

重要的时间处理规则:
- 将提供的“对话开始时间”作为本次对话/情景开始的精确时间。
- 当对话中提到相对时间（例如，“昨天”，“上周”）时，既要保留原始的相对表述，也要计算出绝对日期。
- 时间参照的格式应为：“原始相对时间 (绝对日期)” - 例如，“上周 (2023年5月7日)”。
- 这种双重格式可以同时支持基于绝对时间和相对时间的问题。
- 所有的绝对时间计算都应基于所提供的开始时间。

请生成一段结构化的情景记忆，并只返回一个包含以下两个字段的JSON对象：
{{
    "title": "一个简洁、描述性的标题，能准确概括主题（10-20个词）",
    "content": "一段详细的、以第三人称叙述的对话事实记录。它必须包含所有重要信息：谁在什么时间参与了对话，讨论了什么，做出了什么决定，表达了什么情感，以及形成了什么计划或结果。请将其写成一个按时间顺序记述实际发生事件的说明，侧重于可观察到的行为和直接陈述，而不是解释性的结论。请使用提供的对话开始时间作为此情景的基础时间。"
}}

要求:
1.  标题应具体且易于搜索（包括关键主题/活动）。
2.  内容必须包含对话中的所有重要信息。
3.  将对话格式转换为叙事性描述。
4.  保持时间顺序和因果关系。
5.  除非明确是第一人称，否则使用第三人称。
6.  包含有助于关键词搜索的具体细节，特别是具体的活动、地点和物品。
7.  对于时间参照，请使用双重格式：“相对时间 (绝对日期)”，以支持不同类型的问题。
8.  在描述决定或行动时，自然地融入其背后的推理或动机。
9.  为了避免检索时的歧义，请始终使用具体的名字而不是代词。

示例:
如果对话开始时间是“中国标准时间 2024年3月14日（星期四）下午3:00”，对话内容是关于 Caroline 计划去徒步旅行：
{{
    "title": "Caroline 于2024年3月14日的雷尼尔山徒步计划：周末探险规划会议",
    "content": "在中国标准时间2024年3月14日下午3:00，Caroline 表达了想在这个周末（2024年3月16-17日）去徒步旅行的兴趣并寻求建议。Caroline 特别想去看雷尼尔山的日出，因为听说那里的风景很美。当 Melanie 询问装备时，Caroline 收到了包括徒步靴、保暖衣物（因为山顶很冷）、用于黎明前出发的手电筒、水和高能量食物在内的建议。Caroline 决定在周六早上（2024年3月16日）提早出发去看日出，因为 Caroline 想要体验这座山的全部美景。Caroline 计划邀请朋友一起去探险，这显示了 Caroline 偏爱共享的体验。Caroline 对这次旅行感到非常兴奋，希望能亲近自然，并从工作压力中得到放松。"
}}

只返回JSON对象，不要添加任何其他文本："""
class ConversationEpisodeBuilder(EpisodeBuilder):
    """
    Specialized builder for conversation data.

    This builder implements intelligent conversation boundary detection and episode
    generation using LLM-powered analysis, converting dialogue into narrative episodic memories.
    """

    def __init__(self, llm_provider: LLMProvider | None = None, custom_instructions: str | None = None):
        print("⚠️⚠️⚠️ DEBUG: ConversationEpisodeBuilder is being initialized! ⚠️⚠️⚠️")
        super().__init__(llm_provider)
        self.custom_instructions = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        # try:
        #     self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # except Exception as e:
        #     print(f"Warning: Failed to initialize tiktoken, falling back to character-based summary. Error: {e}")
        #     self.tokenizer = None

    @property
    def supported_data_type(self) -> DataType:
        return DataType.CONVERSATION

    @property
    def default_episode_type(self) -> EpisodeType:
        return EpisodeType.CONVERSATIONAL

    async def _extract_content(self, data: TypedEventData) -> tuple[str, str, str]:
        """Extract title, content, and summary from conversation data."""
        if not isinstance(data, ConversationData):
            raise ValueError("Expected ConversationData")

        # Get conversation text with timestamps for better LLM understanding
        conversation_text = data.get_conversation_text(include_timestamps=True)

        # Get the start time for the conversation
        start_time = data.timestamp

        if self.llm_provider:
            # Use LLM to generate structured episode content
            return await self._generate_episode_with_llm(conversation_text, start_time)
        else:
            # Fallback to simple content extraction
            return self._generate_content_simple(data, conversation_text)

    async def _generate_episode_with_llm(self, conversation_text: str, start_time: datetime) -> tuple[str, str, str]:
        """Generate episode using LLM provider with structured prompting."""

        # Format start time for the prompt in natural language
        weekday = start_time.strftime("%A")  # Monday, Tuesday, etc.
        month_day = start_time.strftime("%B %d, %Y")  # March 14, 2024
        time_of_day = start_time.strftime("%I:%M %p")  # 3:00 PM

        start_time_str = f"{month_day} ({weekday}) at {time_of_day} UTC"

        prompt = EPISODE_GENERATION_PROMPT.format(
            conversation_start_time=start_time_str,
            conversation=conversation_text,
            custom_instructions=self.custom_instructions,
        )

        print("[ConversationEpisodeBuilder] Generating episode – sending prompt to LLM…")

        try:
            resp = await self.llm_provider.generate(prompt, temperature=0.3)
            print(f"[ConversationEpisodeBuilder] Episode generation response length: {len(resp)} chars")

            # Try to extract JSON from response using regex pattern matching
            json_match = re.search(r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Try to parse the entire response as JSON
                data = json.loads(resp)

        except Exception as e:
            print(f"Episode generation error: {e}")
            print(f"Response was: {resp[:200] if 'resp' in locals() else 'No response'}...")
            data = {"title": "Conversation Episode", "content": conversation_text}

        # Ensure we have required fields with fallback defaults
        if "title" not in data:
            data["title"] = "Conversation Episode"
        if "content" not in data:
            data["content"] = conversation_text

        title = data["title"]
        content = data["content"]

        summary = data.get("summary", content)

        return title, content, summary

    def _generate_content_simple(self, data: ConversationData, conversation_text: str) -> tuple[str, str, str]:
        """Simple content generation without LLM (fallback)."""

        messages = data.messages
        if not messages:
            return "Empty Conversation", "No messages in conversation", "Empty conversation"

        # Generate simple title based on first message content
        first_msg = messages[0] if messages else None
        if first_msg and first_msg.content:
            title = first_msg.content[:50] + ("..." if len(first_msg.content) > 50 else "")
        else:
            title = f"Conversation on {data.timestamp.strftime('%Y-%m-%d %H:%M')}"

        # Use the formatted conversation as content
        content = conversation_text

        summary = data.get("summary", content)

        return title, content, summary

    async def _detect_boundary(
        self, conversation_history: list[dict[str, str]], new_messages: list[dict[str, str]], smart_mask: bool = False
    ) -> tuple[bool, str, bool]:
        """
        Detect episode boundary using LLM-based analysis.

        Analyzes conversation flow to determine if new messages should trigger
        the creation of a new episode based on topic changes, intent transitions,
        temporal markers, and content relevance.

        About smart mask:
        When the dialogue length is too long, automatically mask the last sentence (because usually a transitional sentence will affect the granularity of segmentation).
        When masked and segmented, this sentence can be placed in both the preceding and following contexts for better granularity.

        Args:
            conversation_history: Previous messages in the conversation
            new_messages: Newly added messages to analyze
            smart_mask: When enabled, automatically mask the last sentence when dialogue > 5 sentences

        Returns:
            Tuple of (should_end_episode, reason_for_decision, masked_boundary_detected)
        """
        # if not conversation_history or len(conversation_history) <= len(new_messages):
        #     return False, "First messages in conversation", False
        if not conversation_history:
            return False, "First messages in conversation", False
        # Apply smart masking logic
        masked_boundary_detected = False
        analysis_history = conversation_history

        if smart_mask and len(conversation_history) > 5:
            # Mask the last sentence by removing it from analysis
            analysis_history = conversation_history[:-1]
            masked_boundary_detected = True

        history_text = self._format_conversation_dicts(analysis_history, include_timestamps=True)
        new_text = self._format_conversation_dicts(new_messages, include_timestamps=True)

        # Calculate time gap between last message in history and first new message
        time_gap_info = self._calculate_time_gap_info(conversation_history, new_messages)

        print(
            f"[ConversationEpisodeBuilder] Detect boundary – history tokens: {len(history_text)} new tokens: {len(new_text)} time gap: {time_gap_info}"
        )

        if not self.llm_provider:
            # Fallback without LLM
            return False, "No LLM provider available", masked_boundary_detected

        prompt = BOUNDARY_DETECTION_PROMPT.format(
            conversation_history=history_text, new_messages=new_text, time_gap_info=time_gap_info
        )
        print("BOUNDARY_DETECTION_PROMPT:",prompt)
        print("[ConversationEpisodeBuilder] Sending boundary prompt to LLM…")

        try:
            resp = await self.llm_provider.generate(prompt, temperature=0.3)
            print(f"[ConversationEpisodeBuilder] Boundary response length: {len(resp)} chars")

            # Parse JSON response from LLM boundary detection
            json_match = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    data.get("should_end", False),
                    data.get("reasoning", "No reason provided"),
                    masked_boundary_detected,
                )
            else:
                return False, "Failed to parse LLM response", masked_boundary_detected

        except Exception as e:
            print(f"Boundary detection error: {e}")
            return False, "Failed to parse LLM response", masked_boundary_detected

    def _calculate_time_gap_info(
        self, conversation_history: list[dict[str, str]], new_messages: list[dict[str, str]]
    ) -> str:
        """Calculate and format time gap information between last history message and first new message."""
        if not conversation_history or not new_messages:
            return "No time gap information available"

        try:
            # Get the last message from history and first new message
            last_history_msg = conversation_history[-1]
            first_new_msg = new_messages[0]

            last_timestamp_str = last_history_msg.get("timestamp", "")
            first_timestamp_str = first_new_msg.get("timestamp", "")

            if not last_timestamp_str or not first_timestamp_str:
                return "No timestamp information available"

            # Parse timestamps
            last_time = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00"))
            first_time = datetime.fromisoformat(first_timestamp_str.replace("Z", "+00:00"))

            # Calculate time difference
            time_diff = first_time - last_time
            total_seconds = time_diff.total_seconds()

            if total_seconds < 0:
                return "Time gap: Messages appear to be out of order"
            elif total_seconds < 60:  # Less than 1 minute
                return f"Time gap: {int(total_seconds)} seconds (immediate response)"
            elif total_seconds < 3600:  # Less than 1 hour
                minutes = int(total_seconds // 60)
                return f"Time gap: {minutes} minutes (recent conversation)"
            elif total_seconds < 86400:  # Less than 1 day
                hours = int(total_seconds // 3600)
                return f"Time gap: {hours} hours (same day, but significant pause)"
            else:  # More than 1 day
                days = int(total_seconds // 86400)
                return f"Time gap: {days} days (long gap, likely new conversation)"

        except (ValueError, KeyError, AttributeError) as e:
            return f"Time gap calculation error: {str(e)}"

    def _format_conversation_dicts(self, messages: list[dict[str, str]], include_timestamps: bool = False) -> str:
        """Format conversation from message dictionaries into plain text."""
        lines = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            speaker_id = msg.get("speaker_id", "")
            timestamp = msg.get("timestamp", "")

            if content:
                if include_timestamps and timestamp:
                    try:
                        # Parse and format timestamp if available
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        lines.append(f"[{time_str}] {speaker_id}: {content}")
                    except (ValueError, AttributeError):
                        # Fallback if timestamp parsing fails
                        lines.append(f"{speaker_id}: {content}")
                else:
                    lines.append(f"{speaker_id}: {content}")
            else:
                print(f"[ConversationEpisodeBuilder] Warning: message {i} has no content")
        return "\n".join(lines)

    def _generate_metadata(self, data: TypedEventData) -> EpisodeMetadata:
        """Generate metadata for conversation data."""
        base_metadata = super()._generate_metadata(data)

        if not isinstance(data, ConversationData):
            return base_metadata

        messages = data.messages

        # Calculate conversation duration
        duration = None
        if len(messages) >= 2:
            first_msg = next((msg for msg in messages if msg.timestamp), None)
            last_msg = next((msg for msg in reversed(messages) if msg.timestamp), None)
            if first_msg and last_msg and first_msg.timestamp and last_msg.timestamp:
                duration = (last_msg.timestamp - first_msg.timestamp).total_seconds()

        # Basic metadata tracking message count and participants
        custom_fields = {
            **base_metadata.custom_fields,
            "message_count": len(messages),
            "unique_participants": len({msg.speaker_id for msg in messages}),
        }

        return EpisodeMetadata(
            source_data_ids=base_metadata.source_data_ids,
            source_types=base_metadata.source_types,
            processing_timestamp=base_metadata.processing_timestamp,
            processing_version=base_metadata.processing_version,
            duration_seconds=duration,
            custom_fields=custom_fields,
        )

    def _extract_structured_data(self, data: TypedEventData) -> dict[str, Any]:
        """Extract conversation-specific structured data (minimal implementation)."""
        base_data = super()._extract_structured_data(data)

        if not isinstance(data, ConversationData):
            return base_data

        messages = data.messages

        return {
            **base_data,
            "conversation_data": {
                "message_count": len(messages),
                "participants": list({msg.speaker_id for msg in messages}),
                "start_time": messages[0].timestamp.isoformat() if messages and messages[0].timestamp else None,
                "end_time": messages[-1].timestamp.isoformat() if messages and messages[-1].timestamp else None,
            },
        }

    def _determine_episode_level(self, data: TypedEventData) -> EpisodeLevel:
        """Determine episode level based on conversation characteristics."""
        if not isinstance(data, ConversationData):
            return EpisodeLevel.ATOMIC

        message_count = len(data.messages)

        # Determine level based on conversation length
        if message_count <= 5:
            return EpisodeLevel.ATOMIC
        elif message_count <= 15:
            return EpisodeLevel.COMPOUND
        else:
            return EpisodeLevel.THEMATIC

    # Empty implementations for other methods as requested
    def _extract_entities(self, data: ConversationData) -> list[str]:
        """Extract entities from conversation (empty implementation)."""
        _ = data  # Suppress unused parameter warning
        return []

    def _extract_topics(self, data: ConversationData) -> list[str]:
        """Extract topics from conversation (empty implementation)."""
        _ = data  # Suppress unused parameter warning
        return []

    def _extract_emotions(self, data: ConversationData) -> list[str]:
        """Extract emotions from conversation (empty implementation)."""
        _ = data  # Suppress unused parameter warning
        return []

    def _extract_key_points(self, data: ConversationData) -> list[str]:
        """Extract key points from conversation (empty implementation)."""
        _ = data  # Suppress unused parameter warning
        return []
