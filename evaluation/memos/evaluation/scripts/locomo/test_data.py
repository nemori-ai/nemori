import datetime

# 原始对话列表
raw_conversations = [
    {"role": "user", "content": "你好，我想了解一下你们的智能客服系统。"},
    {"role": "assistant", "content": "您好！我们的智能客服系统能够7x24小时在线，处理常见的客户咨询，并且可以根据业务需求进行定制。"},
    {"role": "user", "content": "听起来不错，它支持哪些渠道的接入？"},
    {"role": "assistant", "content": "目前支持网页、App、微信公众号和小程序等多种主流渠道。"},
    {"role": "user", "content": "如果遇到复杂问题，机器人解决不了怎么办？"},
    {"role": "assistant", "content": "系统支持无缝转接到人工客服，机器人和人工的聊天记录会同步，确保服务连贯性。"}
]

def convert_conversations(raw_data, sample_id, speaker_a, speaker_b):
    """
    将原始对话列表转换为指定的目标格式。
    """
    conversation_data = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }
    
    # 为了演示，我们假设对话发生在两个不同的时间段
    # 您可以根据实际情况进行切分
    session_1_messages = []
    session_2_messages = []
    
    # 设置一个基础时间戳
    base_time = datetime.datetime.fromisoformat("2024-01-23T10:00:00")

    for i, message in enumerate(raw_data):
        timestamp = (base_time + datetime.timedelta(minutes=i*2)).isoformat() + "Z"
        speaker_name = speaker_a if message["role"] == "user" else speaker_b
        
        # 假设前4条消息在 session_1，其余在 session_2
        if i < 4:
            dia_id = f"D1:{len(session_1_messages) + 1}"
            session_1_messages.append({
                "speaker": speaker_name,
                "dia_id": dia_id,
                "text": message["content"],
                "timestamp": timestamp
            })
        else:
            if not session_2_messages: # 如果是session_2的第一条消息，重新设置时间
                base_time_s2 = datetime.datetime.fromisoformat("2024-01-23T14:00:00")
                timestamp = base_time_s2.isoformat() + "Z"

            dia_id = f"D2:{len(session_2_messages) + 1}"
            session_2_messages.append({
                "speaker": speaker_name,
                "dia_id": dia_id,
                "text": message["content"],
                "timestamp": timestamp
            })

    conversation_data["session_1"] = session_1_messages
    s1_time = datetime.datetime.fromisoformat(session_1_messages[0]["timestamp"].replace("Z", ""))
    conversation_data["session_1_date_time"] = s1_time.strftime("%-I:%M %p on %d %B, %Y")

    if session_2_messages:
        conversation_data["session_2"] = session_2_messages
        s2_time = datetime.datetime.fromisoformat(session_2_messages[0]["timestamp"].replace("Z", ""))
        conversation_data["session_2_date_time"] = s2_time.strftime("%-I:%M %p on %d %B, %Y")

    # 手动添加示例问答
    qa_pairs = [
        {
            "question": "智能客服系统支持哪些渠道？",
            "answer": "支持网页、App、微信公众号和小程序。",
            "evidence": ["D1:4"],
            "category": 1
        },
        {
            "question": "机器人无法解决问题时如何处理？",
            "answer": "系统会将对话无缝转接给人工客服，并同步聊天记录。",
            "evidence": ["D2:2"],
            "category": 1
        }
    ]

    return {
        "sample_id": sample_id,
        "conversation": conversation_data,
        "qa": qa_pairs
    }

# 执行转换
test_conversations = [
    convert_conversations(raw_conversations, "test_004", "客户", "客服机器人")
]

# 打印结果，以验证格式
import json
print(json.dumps(test_conversations, indent=4, ensure_ascii=False))