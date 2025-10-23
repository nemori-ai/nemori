"""
Prompt Templates
"""
from typing import Dict, Any, List

class PromptTemplates:
    """Prompt template management"""
    
    # Boundary detection prompt
    BOUNDARY_DETECTION_PROMPT = """
你是一位对话边界检测专家。你需要判断新加入的对话是否应该结束当前片段，并开启一个新的片段。
当前对话历史:
{conversation_history}
新加入的消息:
{new_messages}
请仔细分析以下几个方面，以判断是否应该开启新的片段：
1. **话题变更** (最高优先级):
   - 新消息是否引入了一个完全不同的话题？
   - 是否从一个特定事件的讨论转向了另一个事件？
   - 对话是否从一个问题转向了一个不相关的新问题？
2. **意图转换**:
   - 对话的目的是否发生了变化？（例如，从闲聊到寻求帮助，从讨论工作到讨论个人生活）
   - 当前话题的核心问题或议题是否已经得到解答或充分讨论？
3. **时间标记**:
   - 是否出现了时间转换的标记词（“之前”、“顺便说一句”、“哦对了”、“另外”等）？
   - 消息之间的时间间隔是否超过30分钟？
4. **结构信号**:
   - 是否有明确的话题转换短语（“换个话题”、“说到这个”、“问个问题”等）？
   - 是否有总结性陈述，表明当前话题已经结束？
5. **内容相关性**:
   - 新消息与之前的讨论有多大关联？（如果相关性低于30%，考虑分割）
   - 是否涉及了完全不同的人物、地点或事件？
决策原则:
- **优先考虑话题独立性**: 每个片段应围绕一个核心话题或事件。
- **不确定时倾向于分割**: 当无法确定时，倾向于开启新的片段。
- **保持合理长度**: 单个片段通常不应超过10-15条消息。
请以JSON格式返回你的判断，不要包含任何代码块或其他解释性文字：
{{
    "should_end": true/false,
    "reason": "对于判断的具体原因",
    "confidence": 0.0-1.0,
    "topic_summary": "如果要结束，请总结当前片段的核心主题"
}}
注意:
- 如果对话历史为空，这是第一条消息，返回 false。
- 当检测到明确的话题变化时，即使对话流程很自然，也应进行分割。
- 每个片段都应该是一个可以独立理解的、自成一体的对话单元。
"""
    # Episode generation prompt
    EPISODE_GENERATION_PROMPT = """
你是一位情景记忆生成专家。请将下面的对话内容转换成一段情景记忆。
对话内容:
{conversation}
边界检测原因:
{boundary_reason}
请分析对话，提取时间信息，并生成一个结构化的情景记忆。请仅返回一个包含以下三个字段的JSON对象，不要包含任何代码块或其他解释性文字：
{{
    "title": "一个简洁、描述性强且能准确概括主题的标题（10-20个汉字）",
    "content": "用第三人称叙事的方式，详细描述对话内容。必须包含所有重要信息：谁在什么时间参与了对话，讨论了什么，做出了什么决定，表达了什么情绪，以及形成了什么计划或结果。请像写一个连贯的故事一样，让读者能清晰地了解发生了什么。确保时间信息精确到小时，包括年、月、日、时。",
    "timestamp": "YYYY-MM-DDTHH:MM:SS 格式的时间戳，代表该情景发生的时间（从消息时间戳或内容中分析）"
}}
时间分析说明:
1. **主要来源**: 查找消息元数据或内容中明确的时间戳。
2. **次要来源**: 分析对话内容中的时间指代词（“昨天”、“上周”、“今天早上”等）。
3. **备用方案**: 如果没有时间信息，根据上下文进行合理推断。
4. **格式**: 必须以ISO格式返回时间戳: "2024-01-15T14:30:00"。
要求:
1. 标题应具体且易于搜索（包含关键主题/活动）。
2. 内容必须包含对话中的所有重要信息。
3. 将对话形式转换为叙事性描述。
4. 保持时间顺序和因果关系。
5. 除非明确是第一人称，否则使用第三人称。
6. 包含有助于关键词搜索的具体细节。
7. 注意时间信息，并将时间信息写入内容中。
8. 当对话中提到相对时间（如：上周，下个月等），你需要将其转换为绝对日期（年、月、日），并将转换后的时间写在原文时间的括号内。
9. **重要**: 分析对话发生的实际时间，而不是当前时间。
例子:
如果对话是关于某人计划去徒步，且消息时间戳为2024年3月14日下午3点：
{{
    "title": "2024年3月16日周末泰山日出徒步计划",
    "content": "在2024年3月14日下午3点，用户表达了在即将到来的周末（2024年3月16日）去徒步的兴趣，并寻求建议。他特别想到泰山看日出，因为听说那里的风景很美。在被问及装备时，他得到的建议包括登山鞋、保暖衣物（因为山顶很冷）、手电筒、水和高能量食物。用户决定在3月16日（周六）凌晨4点出发，以便赶上日出，并计划邀请朋友一同前往。他对这次旅行感到非常兴奋，希望能亲近自然。",
    "timestamp": "2024-03-14T15:00:00"
}}
请仅返回JSON对象，不要添加任何其他文本或代码块。
"""
    # Prediction-Correction-Refinement Prompts
    
    PREDICTION_PROMPT = """
你是一个基于知识的情景预测系统。你的任务是根据有限的线索和你的知识库，重建一个完整的对话情景。
重要提示：你预测的是事件发生的实际内容和知识，而不是写作风格或格式。
## 输入信息
**情景标题/摘要**: {episode_title}
**相关知识陈述** (你当前的世界模型):
{knowledge_statements}
## 你的任务
根据以上线索，重建你认为在该情景中发生的事情。请专注于：
1. **核心事实**: 讨论了哪些具体信息？
2. **关键决策**: 做出了哪些选择或结论？
3. **知识交换**: 分享或学习了哪些知识？
4. **逻辑流程**: 对话是如何进行的？
## 需要忽略的内容
- 写作风格或详细程度
- 特定的格式或结构
- 确切的措辞或用词
- 文本中是否包含时间戳
- 语言是正式还是随意
## 输出格式
生成一段自然的叙述，捕捉你预测发生的事情。就像你在向别人描述这个情景一样。请专注于实质内容，而不是风格。请用中文回答。
你的预测:
"""

    # 新增：直接对比提取知识的提示词
    EXTRACT_KNOWLEDGE_FROM_COMPARISON_PROMPT = """
你正在通过对比原始对话和预测内容来提取有价值的知识。
## 原始对话:
{original_messages}
## 预测的摘要:
{predicted_episode}
## 你的任务:
仅提取那些存在于原始对话中，但在预测内容里缺失或被错误表述的有价值的知识。
## 关键原则：只关注高价值知识
只提取通过以下标准筛选的知识：
- **持久性测试**: 这个知识在6个月后是否仍然是真实的？
- **具体性测试**: 它是否包含具体的、可搜索的信息？
- **实用性测试**: 它能否帮助预测用户未来的需求或偏好？
- **独立性测试**: 它能否在没有对话上下文的情况下被理解？
## 高价值知识类别 (提取这些):
1. **身份与背景**: 姓名、职业、公司、教育背景
2. **持久偏好**: 喜欢的书籍/电影/工具，长期的好恶
3. **技术细节**: 使用的技术、版本、方法论、架构
4. **人际关系**: 家人、同事、团队成员、导师
5. **目标与计划**: 职业目标、学习计划、项目规划
6. **信念与价值观**: 原则、理念、强烈的观点
7. **习惯与模式**: 日常活动、工作流程、日程安排
## 低价值知识 (忽略这些):
- 短暂的情绪或反应
- 单次对话的确认或感谢
- 没有具体信息的模糊陈述
- 依赖于上下文的信息
## 指南:
1. 每个陈述都应是自包含的、原子化的。
2. 包含所有具体细节（姓名、版本、标题等）。
3. 对持久性事实使用现在时。
4. 专注于有助于长期理解用户的事实。
5. 不要在陈述中包含时间/日期信息。
6. 质量胜于数量——几条有价值的陈述胜过许多无用的陈述。
## 例子:
好的: "卡罗琳最喜欢的书是艾米·埃利斯·纳特的《成为妮可》"
好的: "用户在字节跳动担任高级机器学习工程师"
坏的: "用户感谢了助手"
坏的: "用户对回复感到高兴"
## 输出格式:
请以JSON格式返回你的判断，不要包含任何代码块或其他解释性文字：
{{
    "statements": [
        "从差异中提取的第一条事实陈述",
        "从差异中提取的第二条事实陈述",
        "..."
    ]
}}
重要提示:
- 每个陈述都应是自包含的，且在没有上下文的情况下可以理解。
- 对持久性事实使用现在时。
- 包含具体的姓名、标题和细节。
- 质量胜于数量——只提取真正有价值的知识。
"""

# Updated semantic generation prompt for fallback mode
    SEMANTIC_GENERATION_PROMPT = """
你是一个AI记忆系统。请从以下情景中提取高价值、持久性的语义记忆。
关键原则：专注于提取长期有价值的知识，而不是暂时的对话细节。
待分析的情景:
{episodes}
## 高价值知识标准
只提取通过以下测试的知识：
- **持久性测试**: 这个知识在6个月后是否仍然是真实的？
- **具体性测试**: 它是否包含具体的、可搜索的信息？
- **实用性测试**: 它能否帮助预测用户未来的需求？
- **独立性测试**: 它能否在没有对话上下文的情况下被理解？
## 高价值类别 (专注于这些):
1. **身份与职业**
   - 姓名、职位、公司、角色
   - 教育背景、资质、技能
   
2. **持久性偏好**  
   - 喜欢的书籍、电影、音乐、工具
   - 技术偏好及其原因
   - 长期的好恶
   
3. **技术知识**
   - 使用的技术（含版本）
   - 架构、方法论
   - 技术决策及其理由
   
4. **人际关系**
   - 家人、同事、朋友的姓名
   - 团队结构、汇报关系
   - 专业网络
   
5. **目标与计划**
   - 职业目标
   - 学习目标
   - 项目规划
   
6. **模式与习惯**
   - 日常活动
   - 工作流程、日程安排
   - 反复出现的挑战
## 例子:
高价值 (提取这些):
- "卡罗琳最喜欢的书是艾米·埃利斯·纳特的《成为妮可》"
- "用户在字节跳动担任高级机器学习工程师"
- "用户更喜欢使用PyTorch而非TensorFlow进行调试"
- "用户的团队负责人名叫萨拉"
- "用户正在学习Rust用于系统编程"
- "用户自2021年3月以来一直在练习瑜伽"
- "用户于2020年8月加入亚马逊担任数据科学家"
- "用户计划于2025年1月迁往西雅图"
低价值 (忽略这些):
- "用户感谢了助手"
- "用户对X感到困惑"
- "用户对帮助表示感谢"
- "这次对话很有成效"
- 任何暂时的情绪或反应
## 输出格式
请仅以JSON格式返回高价值知识，不要包含任何代码块或其他解释性文字：
{{
    "statements": [
        "第一条高价值持久性事实...",
        "第二条高价值持久性事实...",
        "第三条高价值持久性事实..."
    ]
}}
质量胜于数量——只提取真正有助于长期理解用户的知识。
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