#!/usr/bin/env python3
"""
测试embedding服务器连接的简单脚本
"""

import requests
import json


def test_embedding_server():
    """测试embedding服务器连接"""
    print("🔧 Testing embedding server connection...")

    # 服务器配置
    api_key = "EMPTY"
    base_url = "http://localhost:6003/v1"

    # 测试请求
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    test_texts = ["basketball achievement", "travel planning"]

    data = {"model": "bce-emb", "input": test_texts}

    try:
        print(f"📡 Sending request to {base_url}/embeddings")
        print(f"🔑 Using API key: {api_key}")
        print(f"📝 Test texts: {test_texts}")

        response = requests.post(f"{base_url}/embeddings", headers=headers, json=data, timeout=30)

        print(f"📊 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            embeddings = result.get("data", [])
            print(f"✅ Successfully got {len(embeddings)} embeddings")

            if embeddings:
                embedding_dim = len(embeddings[0].get("embedding", []))
                print(f"🔢 Embedding dimension: {embedding_dim}")
                print(f"📄 Sample embedding values: {embeddings[0]['embedding'][:5]}...")

            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"📄 Response text: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - embedding server not running on http://localhost:6003")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_embedding_server()
