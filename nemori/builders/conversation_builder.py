"""
Conversation Episode Builder for Nemori.

This builder specializes in transforming conversation data into episodic memories,
implementing intelligent boundary detection and LLM-powered episode generation
to create coherent narrative memories from conversational exchanges.
"""

import json
import re
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
You are a dialogue boundary detection expert. You need to determine if the newly added dialogue should end the current episode and start a new one.

Current conversation history:
{conversation_history}

Newly added messages:
{new_messages}

Please carefully analyze the following aspects to determine if a new episode should begin:

1. **Topic Change** (Highest Priority):
   - Do the new messages introduce a completely different topic?
   - Is there a shift from one specific event to another?
   - Has the conversation moved from one question to an unrelated new question?

2. **Intent Transition**:
   - Has the purpose of the conversation changed? (e.g., from casual chat to seeking help, from discussing work to discussing personal life)
   - Has the core question or issue of the current topic been answered or fully discussed?

3. **Temporal Markers**:
   - Are there temporal transition markers ("earlier", "before", "by the way", "oh right", "also", etc.)?
   - Is the time gap between messages more than 30 minutes?

4. **Structural Signals**:
   - Are there explicit topic transition phrases ("changing topics", "speaking of which", "quick question", etc.)?
   - Are there concluding statements indicating the current topic is finished?

5. **Content Relevance**:
   - How related is the new message to the previous discussion? (Consider splitting if relevance < 30%)
   - Does it involve completely different people, places, or events?

Custom instructions:
{custom_instructions}

Decision Principles:
- **Prioritize topic independence**: Each episode should revolve around one core topic or event
- **When in doubt, split**: When uncertain, lean towards starting a new episode
- **Maintain reasonable length**: A single episode typically shouldn't exceed 10-15 messages

Please return your judgment in JSON format:
{{
    "should_end": true/false,
    "reason": "Specific reason for the judgment",
    "confidence": 0.0-1.0,
    "topic_summary": "If ending, summarize the core topic of the current episode"
}}

Note:
- If conversation history is empty, this is the first message, return false
- When a clear topic change is detected, split even if the conversation flows naturally
- Each episode should be a self-contained conversational unit that can be understood independently
"""

EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation content:
{conversation}

Boundary detection reason:
{boundary_reason}

Custom instructions:
{custom_instructions}

Please generate a structured episodic memory and return only a JSON object containing the following two fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A detailed description of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a coherent story so that the reader can clearly understand what happened. Ensure that time information is precise to the hour, including year, month, day, and hour."
}}

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search.
8. Notice the time information, and write the time information in the content.
9. When relative times (e.g., last week, next month, etc.) are mentioned in the conversation, you need to convert them to absolute dates (year, month, day). The transformed context should no longer contain relative time references.

Example:
If the conversation is about someone planning to go hiking:
{{
    "title": "Weekend Hiking Plan March 16, 2024: Sunrise Trip to Mount Rainier",
    "content": "On March 14, 2024 at 3:00 PM, the user expressed interest in going hiking on the upcoming weekend (March 16, 2024) and sought advice. They particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear, they received suggestions including hiking boots, warm clothing (as it's cold at the summit), a flashlight, water, and high-energy food. The user decided to leave at 4:00 AM on Saturday, March 16, 2024 to catch the sunrise and planned to invite friends for the adventure. They were very excited about the trip, hoping to connect with nature."
}}

Return only the JSON object, do not add any other text:
"""


class ConversationEpisodeBuilder(EpisodeBuilder):
    """
    Specialized builder for conversation data.

    This builder implements intelligent conversation boundary detection and episode
    generation using LLM-powered analysis, converting dialogue into narrative episodic memories.
    """

    def __init__(self, llm_provider: LLMProvider | None = None, custom_instructions: str | None = None):
        super().__init__(llm_provider)
        self.custom_instructions = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS

    @property
    def supported_data_type(self) -> DataType:
        return DataType.CONVERSATION

    @property
    def default_episode_type(self) -> EpisodeType:
        return EpisodeType.CONVERSATIONAL

    def _extract_content(self, data: TypedEventData) -> tuple[str, str, str]:
        """Extract title, content, and summary from conversation data."""
        if not isinstance(data, ConversationData):
            raise ValueError("Expected ConversationData")

        conversation_text = data.get_conversation_text()

        if self.llm_provider:
            # Use LLM to generate structured episode content
            return self._generate_episode_with_llm(conversation_text, "User requested episode generation")
        else:
            # Fallback to simple content extraction
            return self._generate_content_simple(data, conversation_text)

    def _generate_episode_with_llm(
        self, conversation_text: str, boundary_reason: str = "Episode generation requested"
    ) -> tuple[str, str, str]:
        """Generate episode using LLM provider with structured prompting."""

        prompt = EPISODE_GENERATION_PROMPT.format(
            conversation=conversation_text,
            boundary_reason=boundary_reason,
            custom_instructions=self.custom_instructions,
        )

        print("[ConversationEpisodeBuilder] Generating episode – sending prompt to LLM…")

        try:
            resp = self.llm_provider.generate(prompt, temperature=0.3)
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

        # Use LLM-provided summary if available, otherwise generate from content
        summary = data.get("summary", content[:200] + "..." if len(content) > 200 else content)

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

        # Generate summary
        summary = content[:200] + "..." if len(content) > 200 else content

        return title, content, summary

    def _detect_boundary(
        self, conversation_history: list[dict[str, str]], new_messages: list[dict[str, str]]
    ) -> tuple[bool, str]:
        """
        Detect episode boundary using LLM-based analysis.

        Analyzes conversation flow to determine if new messages should trigger
        the creation of a new episode based on topic changes, intent transitions,
        temporal markers, and content relevance.

        Args:
            conversation_history: Previous messages in the conversation
            new_messages: Newly added messages to analyze

        Returns:
            Tuple of (should_end_episode, reason_for_decision)
        """
        if not conversation_history or len(conversation_history) <= len(new_messages):
            return False, "First messages in conversation"

        history_text = self._format_conversation_dicts(conversation_history[: -len(new_messages)])
        new_text = self._format_conversation_dicts(new_messages)

        print(
            f"[ConversationEpisodeBuilder] Detect boundary – history tokens: {len(history_text)} new tokens: {len(new_text)}"
        )

        if not self.llm_provider:
            # Fallback without LLM
            return False, "No LLM provider available"

        prompt = BOUNDARY_DETECTION_PROMPT.format(
            conversation_history=history_text, new_messages=new_text, custom_instructions=self.custom_instructions
        )

        print("[ConversationEpisodeBuilder] Sending boundary prompt to LLM…")

        try:
            resp = self.llm_provider.generate(prompt, temperature=0.3)
            print(f"[ConversationEpisodeBuilder] Boundary response length: {len(resp)} chars")

            # Parse JSON response from LLM boundary detection
            json_match = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("should_end", False), data.get("reason", "No reason provided")
            else:
                return False, "Failed to parse LLM response"

        except Exception as e:
            print(f"Boundary detection error: {e}")
            return False, "Failed to parse LLM response"

    def _format_conversation_dicts(self, messages: list[dict[str, str]]) -> str:
        """Format conversation from message dictionaries into plain text."""
        lines = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if content:
                lines.append(f"{content}")
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
