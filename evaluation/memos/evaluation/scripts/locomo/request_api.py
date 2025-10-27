import requests
import json
import time
import os

# --- 配置信息 ---
BASE_URL = "http://localhost:5001"

# 我们使用一个简单的对话作为例子
# 注意：这是个包含一个对话的列表
conversation_to_update = [
    {
        "sample_id": "test_001",
        "user_id": "77777",
        "conversation": {
            "speaker_a": "张三",
            "speaker_b": "李四",
            "session_1": [
                {
                    "speaker": "张三",
                    "dia_id": "D1:1", 
                    "text": "最近我在学习Python机器学习，你觉得哪些库比较重要？",
                    "timestamp": "2024-01-20T10:00:00Z"
                },
                {
                    "speaker": "李四",
                    "dia_id": "D1:2",
                    "text": "scikit-learn是入门必备，pandas用于数据处理，numpy处理数值计算。深度学习推荐PyTorch。",
                    "timestamp": "2024-01-20T10:02:00Z"
                },
                {
                    "speaker": "张三", 
                    "dia_id": "D1:3",
                    "text": "PyTorch和TensorFlow比较，哪个更适合初学者？",
                    "timestamp": "2024-01-20T10:04:00Z"
                },
                {
                    "speaker": "李四",
                    "dia_id": "D1:4", 
                    "text": "PyTorch的动态图更直观，调试容易。TensorFlow适合生产部署，但学习曲线陡峭。",
                    "timestamp": "2024-01-20T10:06:00Z"
                }
            ],
            "session_1_date_time": "10:00 AM on 20 January, 2024",
            "session_2": [
                {
                    "speaker": "张三",
                    "dia_id": "D2:1",
                    "text": "数据预处理方面有什么建议吗？我的数据集有很多缺失值。",
                    "timestamp": "2024-01-20T14:00:00Z"
                },
                {
                    "speaker": "李四", 
                    "dia_id": "D2:2",
                    "text": "缺失值可以用均值填充、前向填充，或者直接删除。要看具体业务场景，时间序列数据建议前向填充。",
                    "timestamp": "2024-01-20T14:02:00Z"
                }
            ],
            "session_2_date_time": "2:00 PM on 20 January, 2024"
        },
        "qa": [
            {
                "question": "张三在学习什么技术？",
                "answer": "Python机器学习", 
                "evidence": ["D1:1"],
                "category": 1
            },
            {
                "question": "李四推荐了哪些Python库？",
                "answer": "scikit-learn, pandas, numpy, PyTorch",
                "evidence": ["D1:2"],
                "category": 1
            }
        ]
    },
]

# 从对话数据中提取关键信息
# API中的很多操作是针对单个对话的，所以我们取列表的第一个元素
single_conversation_data = conversation_to_update[0]
USER_ID = single_conversation_data["user_id"]
SPEAKER_A = single_conversation_data["conversation"]["speaker_a"]
SPEAKER_B = single_conversation_data["conversation"]["speaker_b"]
# 使用 User ID 来创建一个唯一的 version
VERSION = f"session_for_user_{USER_ID}"

# 初始化所需的配置（与服务器端代码保持一致）
# 从环境变量读取，如果不存在则使用默认值
LLM_CONFIG = {
    "api_key": os.getenv("JENIYA_API_KEY", "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"),
    "base_url": os.getenv("JENIYA_BASE_URL", "https://jeniya.cn/v1"),
    "model": "gpt-4o-mini"
}
EMBED_CONFIG = {
    "emb_api_key": "EMPTY",
    "emb_base_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:6007/v1"),
    "embed_model": "qwen3-emb"
}


def print_response(name, response):
    """格式化打印响应的辅助函数"""
    print(f"--- Response from {name} ---")
    print(f"Status Code: {response.status_code}")
    # 使用 json.dumps 美化输出
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    print("-" * 30 + "\n")


def main():
    """主执行函数，按顺序调用 API"""
    
    # ----------------------------------------------------
    # 步骤 1: 初始化 Experiment 和 Client
    # ----------------------------------------------------
    print(f"🚀 STEP 1: Initializing services for version: {VERSION}\n")

    # 初始化 Experiment
    exp_payload = {
        "version": VERSION,
        "llm_config": LLM_CONFIG,
        "embed_config": EMBED_CONFIG
    }

    response = requests.post(f"{BASE_URL}/api/initialize/experiment", json=exp_payload)
    response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
    print_response("Initialize Experiment", response)

    # 初始化 Client
    client_payload = {
        "version": VERSION,
        "embed_config": EMBED_CONFIG
    }
    response = requests.post(f"{BASE_URL}/api/initialize/client", json=client_payload)
    response.raise_for_status()
    print_response("Initialize Client", response)
    
    # ----------------------------------------------------
    # 步骤 2: 检测对话边界
    # ----------------------------------------------------
    print("🚀 STEP 2: Detecting conversation boundaries...\n")
    boundaries_payload = {
        "version": VERSION,
        # 注意：此接口处理单个对话，所以我们传入 `single_conversation_data`
        "messages": single_conversation_data
    }
    boundaries = None
    response = requests.post(f"{BASE_URL}/api/boundaries/detect", json=boundaries_payload)
    response.raise_for_status()
    print_response("Detect Boundaries", response)
    boundaries = response.json().get("episode_boundaries")
    if not boundaries:
        print("⚠️ Warning: Did not receive boundaries from the API.")

    # ----------------------------------------------------
    # 步骤 3: 使用 V2 接口更新记忆
    # ----------------------------------------------------
    if boundaries:
        print("🚀 STEP 3: Updating memory using V2 endpoint with detected boundaries...\n")
        update_v2_payload = {
            "version": VERSION,
            "messages": single_conversation_data,
            "boundaries": boundaries
        }
        response = requests.post(f"{BASE_URL}/api/memory/update-v2", json=update_v2_payload)
        response.raise_for_status()
        print_response("Update Memory V2", response)
    
    # --- （可选）使用传统的 /api/memory/update 接口 ---
    # print("🚀 (Optional) STEP 3.A: Updating memory using the standard endpoint...")
    # update_payload = {
    #     "version": VERSION,
    #     "conversations": conversation_to_update # 此接口需要一个对话列表
    # }
    # try:
    #     response = requests.post(f"{BASE_URL}/api/memory/update", json=update_payload)
    #     response.raise_for_status()
    #     print_response("Update Memory (Standard)", response)
    #     # 注意: 这个接口是异步的 (返回 202)，立即查询可能查不到最新数据
    #     # 在实际应用中，您可能需要一个回调或轮询机制来确认任务完成
    #     print("Waiting 5 seconds for the async update to process...")
    #     time.sleep(5)
    # except requests.exceptions.RequestException as e:
    #     print(f"❌ Error updating memory (Standard): {e}")
    #     return


    # ----------------------------------------------------
    # 步骤 4: 查询记忆库
    # ----------------------------------------------------
    print("🚀 STEP 4: Querying memory...\n")
    query_payload = {
        "version": VERSION,
        "user_id": USER_ID,
        "query": "李四推荐了哪些用于机器学习的Python库？",
        "speaker_a": SPEAKER_A,
        "speaker_b": SPEAKER_B,
        "top_k": 20
    }
    response = requests.post(f"{BASE_URL}/api/memory/query", json=query_payload)
    response.raise_for_status()
    print_response("Query Memory", response)



if __name__ == "__main__":
    main()