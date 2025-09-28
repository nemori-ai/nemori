"""
Prompt Templates
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Prompt template management"""
    
    # Boundary detection prompt
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

    @classmethod
    def get_boundary_detection_prompt(cls, conversation_history: str, new_messages: str) -> str:
        """Get boundary detection prompt"""
        return cls.BOUNDARY_DETECTION_PROMPT.format(
            conversation_history=conversation_history,
            new_messages=new_messages
        )
    
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
    
 