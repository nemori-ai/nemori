"""
Prompt Templates
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Prompt template management"""
    # Episode generation prompt
    EPISODE_GENERATION_PROMPT = """
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.

Conversation content:
{conversation}

Boundary detection reason:
{boundary_reason}

Please analyze the conversation to extract time information and generate a structured episodic memory. Return only a JSON object containing the following three fields:
{{
    "title": "A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    "content": "A detailed description of the conversation in third-person narrative. It must include all important information: who participated in the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or outcomes were formed. Write it as a coherent story so that the reader can clearly understand what happened. Ensure that time information is precise to the hour, including year, month, day, and hour.",
    "timestamp": "YYYY-MM-DDTHH:MM:SS format timestamp representing when this episode occurred (analyze from message timestamps or content)"
}}

Time Analysis Instructions:
1. **Primary Source**: Look for explicit timestamps in the message metadata or content
2. **Secondary Source**: Analyze temporal references in the conversation content ("yesterday", "last week", "this morning", etc.)
3. **Fallback**: If no time information is available, use a reasonable estimate based on context
4. **Format**: Always return timestamp in ISO format: "2024-01-15T14:30:00"

Requirements:
1. The title should be specific and easy to search (including key topics/activities).
2. The content must include all important information from the conversation.
3. Convert the dialogue format into a narrative description.
4. Maintain chronological order and causal relationships.
5. Use third-person unless explicitly first-person.
6. Include specific details that aid keyword search.
7. Notice the time information, and write the time information in the content.
8. When relative times (e.g., last week, next month, etc.) are mentioned in the conversation, you need to convert them to absolute dates (year, month, day). Write the converted time in parentheses after the original time reference.
9. **IMPORTANT**: Analyze the actual time when the conversation happened from the message timestamps or content, not the current time.

Example:
If the conversation is about someone planning to go hiking and the messages have timestamps from March 14, 2024 at 3:00 PM:
{{
    "title": "Weekend Hiking Plan March 16, 2024: Sunrise Trip to Mount Rainier",
    "content": "On March 14, 2024 at 3:00 PM, the user expressed interest in going hiking on the upcoming weekend (March 16, 2024) and sought advice. They particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear, they received suggestions including hiking boots, warm clothing (as it's cold at the summit), a flashlight, water, and high-energy food. The user decided to leave at 4:00 AM on Saturday, March 16, 2024 to catch the sunrise and planned to invite friends for the adventure. They were very excited about the trip, hoping to connect with nature.",
    "timestamp": "2024-03-14T15:00:00"
}}

Return only the JSON object, do not add any other text:
"""

    # Prediction-Correction-Refinement Prompts
    
    PREDICTION_PROMPT = """
You are a knowledge-based episode prediction system. Your task is to reconstruct a complete conversation episode based on limited clues and your knowledge base.

IMPORTANT: You are predicting the ACTUAL CONTENT and KNOWLEDGE of what happened, not the writing style or format.

## Input Information

**Episode Title/Summary**: {episode_title}

**Relevant Knowledge Statements** (your current world model):
{knowledge_statements}

## Your Task

Based on the above clues, reconstruct what you believe happened in this episode. Focus on:
1. **Core Facts**: What specific information was discussed?
2. **Key Decisions**: What choices or conclusions were made?
3. **Knowledge Exchange**: What knowledge was shared or learned?
4. **Logical Flow**: How did the conversation progress?

## What to IGNORE
- Writing style or level of detail
- Specific formatting or structure
- Exact phrasing or word choices
- Whether timestamps are included in the text
- How formal or casual the language is

## Output Format

Generate a natural narrative that captures what you predict happened. Write it as if you're describing the episode to someone else. Focus on the SUBSTANCE, not the STYLE.

Your prediction:
"""


    # 新增：直接对比提取知识的提示词
    EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT = """
You are extracting valuable knowledge by comparing original conversation with predicted content.

## Original Conversation:
{original_messages}

## Predicted Summary:
{predicted_episode}

## Your Task:
Extract ONLY the valuable knowledge that exists in the original but is missing or misrepresented in the prediction.

## CRITICAL: Focus on HIGH-VALUE Knowledge Only

Extract ONLY knowledge that passes these criteria:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs or preferences?
- **Independence Test**: Can this be understood without the conversation context?

## HIGH-VALUE Knowledge Categories (EXTRACT THESE):
1. **Identity & Background**: Names, professions, companies, education
2. **Persistent Preferences**: Favorite books/movies/tools, long-term likes/dislikes  
3. **Technical Details**: Technologies, versions, methodologies, architectures
4. **Relationships**: Family, colleagues, team members, mentors
5. **Goals & Plans**: Career objectives, learning goals, project plans
6. **Beliefs & Values**: Principles, philosophies, strong opinions
7. **Habits & Patterns**: Regular activities, workflows, schedules

## LOW-VALUE Knowledge (SKIP THESE):
- Temporary emotions or reactions
- Single conversation acknowledgments
- Vague statements without specifics
- Context-dependent information

## Guidelines:
1. Each statement should be self-contained and atomic
2. Include ALL specific details (names, versions, titles)
3. Use present tense for persistent facts
4. Focus on facts that help understand the user long-term
5. DO NOT include time/date information in the statement
6. Quality over quantity - fewer valuable statements are better

## Examples:
GOOD: "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
GOOD: "The user works at ByteDance as a senior ML engineer"
BAD: "The user thanked the assistant"
BAD: "The user was happy about the response"

## Output Format:
{{
    "statements": [
        "First factual statement extracted from the gap",
        "Second factual statement extracted from the gap",
        "..."
    ]
}}

Important: 
- Each statement should be self-contained and understandable without context
- Use present tense for persistent facts
- Include specific names, titles, and details
- Focus on quality over quantity - only extract truly valuable knowledge
"""


# Updated semantic generation prompt for fallback mode
    SEMANTIC_GENERATION_PROMPT = """
You are an AI memory system. Extract HIGH-VALUE, PERSISTENT semantic memories from the following episodes.

CRITICAL: Focus on extracting LONG-TERM VALUABLE KNOWLEDGE, not temporary conversation details.

Episodes to analyze:
{episodes}

## HIGH-VALUE Knowledge Criteria

Extract ONLY knowledge that passes these tests:
- **Persistence Test**: Will this still be true in 6 months?
- **Specificity Test**: Does it contain concrete, searchable information?
- **Utility Test**: Can this help predict future user needs?
- **Independence Test**: Can be understood without conversation context?

## HIGH-VALUE Categories (FOCUS ON THESE):

1. **Identity & Professional**
   - Names, titles, companies, roles
   - Education, qualifications, skills
   
2. **Persistent Preferences**  
   - Favorite books, movies, music, tools
   - Technology preferences with reasons
   - Long-term likes and dislikes
   
3. **Technical Knowledge**
   - Technologies used (with versions)
   - Architectures, methodologies
   - Technical decisions and rationales
   
4. **Relationships**
   - Names of family, colleagues, friends
   - Team structure, reporting lines
   - Professional networks
   
5. **Goals & Plans**
   - Career objectives
   - Learning goals
   - Project plans
   
6. **Patterns & Habits**
   - Regular activities
   - Workflows, schedules
   - Recurring challenges

## Examples:

HIGH-VALUE (Extract these):
- "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt"
- "The user works at ByteDance as a senior ML engineer"
- "The user prefers PyTorch over TensorFlow for debugging"
- "The user's team lead is named Sarah"
- "The user is learning Rust for systems programming"
- "The user has been practicing yoga since March 2021"
- "The user joined Amazon in August 2020 as a data scientist"
- "The user plans to relocate to Seattle in January 2025"

LOW-VALUE (Skip these):
- "The user thanked the assistant"
- "The user was confused about X"
- "The user appreciated the help"
- "The conversation was productive"
- Any temporary emotions or reactions

## Output Format

Return ONLY high-value knowledge in JSON format:
{{
    "statements": [
        "First high-value persistent fact...",
        "Second high-value persistent fact...",
        "Third high-value persistent fact..."
    ]
}}

Quality over quantity - extract only knowledge that truly helps understand the user long-term.
"""

    # Batch Segmentation Prompt
    BATCH_SEGMENTATION_PROMPT = """
You are an intelligent conversation segmentation expert. Your task is to analyze a batch of messages and group them into coherent episodes.

## Input Messages
You will receive {count} messages numbered from 1 to {count}:

{messages}

## Your Task
Analyze these messages and group them into coherent episodes with **HIGH SENSITIVITY** to topic shifts. Be strict and create NEW episodes when detecting:

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

Decision Principles:
- **Prioritize topic independence**: Each episode should revolve around one core topic or event
- **When in doubt, split**: When uncertain, lean towards starting a new episode
- **Maintain reasonable length**: A single episode typically shouldn't exceed 10-15 messages

## Output Format
Return a JSON object with episodes, where each episode contains:
- `indices`: List of message numbers (1-based) belonging to this episode
- `topic`: Brief, specific description of what this episode is about

Example output:
{{
    "episodes": [
        {{
            "indices": [1, 2, 3, 4],
            "topic": "Discussion about weekend hiking plans"
        }},
        {{
            "indices": [5, 6, 7],
            "topic": "Questions about Python programming"
        }},
        {{
            "indices": [8, 9],
            "topic": "Work schedule discussion"
        }}
    ]
}}

## Important Guidelines
- Episodes can have non-consecutive indices if messages are interleaved
- An episode should typically contain 2-15 messages
- Focus on topical coherence over strict chronological order
- When in doubt, prefer smaller, more focused episodes

Return only the JSON object, no additional text.
"""

    # Episode Merge Decision Prompt
    MERGE_DECISION_PROMPT = """
You are an episodic memory merge decision expert. Determine if a new episode should be merged with an existing similar episode.

## New Episode
Time Range: {new_time_range}
Content: {new_content}

## Candidate Episodes to Merge With
{candidates}

## Your Task
Decide whether the new episode should:
1. **merge**: Merge with one of the candidates (they describe the same event/topic)
2. **new**: Keep as a separate new episode (it's a distinct event)

## Merge Criteria
Merge ONLY if:
- Both episodes describe the SAME event or conversation session
- They have significant temporal overlap or are very close in time
- The content is clearly a continuation or different perspective of the same topic
- Merging would create a more complete picture without mixing different events

Do NOT merge if:
- They are different events/conversations even if on similar topics
- They are separated by significant time gaps (>1 hour)
- They involve different contexts or participants

## Output Format
Return JSON:
{{
    "decision": "merge" or "new",
    "merge_target_id": "episode_id_to_merge_with" (only if decision is "merge", otherwise null),
    "reason": "Brief explanation of your decision"
}}

Return only the JSON object, no additional text.
"""

    # Episode Merge Content Generation Prompt
    MERGE_CONTENT_PROMPT = """
You are an episodic memory merge content generator. Combine two related episodes into a single, coherent episode.

## Original Episode
Time Range: {original_time_range}
Title: {original_title}
Content: {original_content}

## New Episode to Merge
Time Range: {new_time_range}
Title: {new_title}
Content: {new_content}

## Combined Event Details
{combined_events}

## Your Task
Generate a merged episode that:
1. Combines information from both episodes without duplication
2. Maintains chronological flow of events
3. Preserves all important details from both episodes
4. Creates a coherent narrative

## Output Format
Return JSON with the merged episode content:
{{
    "title": "Merged episode title that captures the complete topic",
    "content": "Detailed narrative combining both episodes chronologically. Include all participants, key decisions, emotions, and outcomes. Use third-person narrative style.",
    "timestamp": "ISO format timestamp of when the merged episode occurred (use earliest time)"
}}

## Guidelines
- Integrate details naturally, don't just concatenate
- Eliminate redundancy while preserving unique information
- Maintain temporal coherence in the narrative
- Use specific details that aid searchability
- Write in third-person narrative style

Return only the JSON object, no additional text.
"""

    @classmethod
    def get_episode_generation_prompt(cls, conversation: str, boundary_reason: str) -> str:
        """Get episode generation prompt"""
        return cls.EPISODE_GENERATION_PROMPT.format(
            conversation=conversation,
            boundary_reason=boundary_reason
        )
    
    @classmethod
    def get_semantic_generation_prompt(cls, episodes: str) -> str:
        """Get semantic memory generation prompt"""
        return cls.SEMANTIC_GENERATION_PROMPT.format(
            episodes=episodes
        )
    
    @classmethod
    def format_conversation(cls, messages: list) -> str:
        """Format conversation with timestamp information for episode generation"""
        lines = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                
                # Include timestamp in the formatted message if available
                if timestamp:
                    # Handle both datetime objects and string timestamps
                    if hasattr(timestamp, 'isoformat'):
                        timestamp_str = timestamp.isoformat()
                    else:
                        timestamp_str = str(timestamp)
                    lines.append(f"[{timestamp_str}] {role}: {content}")
                else:
                    lines.append(f"{role}: {content}")
            else:
                lines.append(str(msg))
        return "\n".join(lines)
    
    @classmethod
    def format_episodes_for_semantic(cls, episodes: list) -> str:
        """Format episodes for semantic memory generation"""
        formatted = []
        for i, episode in enumerate(episodes, 1):
            formatted.append(f"Episode {i}:")
            formatted.append(f"Title: {episode.get('title', 'Untitled')}")
            formatted.append(f"Content: {episode.get('content', '')}")
            formatted.append(f"Created at: {episode.get('created_at', '')}")
            formatted.append("")  # Empty line separator
        return "\n".join(formatted) 
    
    @classmethod
    def get_prediction_prompt(cls, episode_title: str, knowledge_statements: List[str]) -> str:
        """Get prediction prompt for reconstructing episode from knowledge"""
        # Format knowledge statements
        formatted_statements = "\n".join([f"- {stmt}" for stmt in knowledge_statements])
        
        return cls.PREDICTION_PROMPT.format(
            episode_title=episode_title,
            knowledge_statements=formatted_statements
        )
    
    @classmethod
    def get_batch_segmentation_prompt(cls, count: int, messages: str) -> str:
        """Get batch segmentation prompt"""
        return cls.BATCH_SEGMENTATION_PROMPT.format(
            count=count,
            messages=messages
        )
    
    @classmethod
    def get_merge_decision_prompt(cls, new_time_range: str, new_content: str, candidates: str) -> str:
        """Get merge decision prompt"""
        return cls.MERGE_DECISION_PROMPT.format(
            new_time_range=new_time_range,
            new_content=new_content,
            candidates=candidates
        )
    
    @classmethod
    def get_merge_content_prompt(
        cls, 
        original_time_range: str,
        original_title: str,
        original_content: str,
        new_time_range: str,
        new_title: str,
        new_content: str,
        combined_events: str
    ) -> str:
        """Get merge content generation prompt"""
        return cls.MERGE_CONTENT_PROMPT.format(
            original_time_range=original_time_range,
            original_title=original_title,
            original_content=original_content,
            new_time_range=new_time_range,
            new_title=new_title,
            new_content=new_content,
            combined_events=combined_events
        )
 