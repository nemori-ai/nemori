#!/usr/bin/env python3
"""
测试 detect boundaries API 的示例脚本
"""

import requests
import json
from datetime import datetime, timedelta

# API 配置
API_BASE_URL = "http://localhost:5001"
DETECT_BOUNDARIES_ENDPOINT = f"{API_BASE_URL}/api/boundaries/detect"

def create_test_messages():
    """创建测试消息数据"""
    base_time = datetime.now()
    
    # 测试消息 - 包含多个主题转换点，应该能检测出边界
    messages = [
        {
            "speaker": "张三",
            "content": "最近我在学习Python机器学习，你觉得哪些库比较重要？",
            "timestamp": (base_time + timedelta(minutes=0)).isoformat() + "Z"
        },
        {
            "speaker": "李四", 
            "content": "scikit-learn是入门必备，pandas用于数据处理，numpy处理数值计算。深度学习推荐PyTorch。",
            "timestamp": (base_time + timedelta(minutes=2)).isoformat() + "Z"
        },
        {
            "speaker": "张三",
            "content": "PyTorch和TensorFlow比较，哪个更适合初学者？",
            "timestamp": (base_time + timedelta(minutes=4)).isoformat() + "Z"
        },
        {
            "speaker": "李四",
            "content": "PyTorch的动态图更直观，调试容易。TensorFlow适合生产部署，但学习曲线陡峭。",
            "timestamp": (base_time + timedelta(minutes=6)).isoformat() + "Z"
        },
        # 主题转换 - 从机器学习库转到数据预处理
        {
            "speaker": "张三", 
            "content": "数据预处理方面有什么建议吗？我的数据集有很多缺失值。",
            "timestamp": (base_time + timedelta(hours=4)).isoformat() + "Z"  # 4小时后
        },
        {
            "speaker": "李四",
            "content": "缺失值可以用均值填充、前向填充，或者直接删除。要看具体业务场景，时间序列数据建议前向填充。",
            "timestamp": (base_time + timedelta(hours=4, minutes=2)).isoformat() + "Z"
        }
    ]
    
    return messages

def test_detect_boundaries():
    """测试边界检测 API"""
    print("🧪 测试边界检测 API")
    print("=" * 50)
    
    # 创建测试数据
    test_messages = create_test_messages()
    
    # 构造请求数据
    request_data = {
        "version": "boundary_test",
        "messages": test_messages
    }
    
    print(f"📋 发送 {len(test_messages)} 条消息进行边界检测...")
    print("消息预览:")
    for i, msg in enumerate(test_messages):
        preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  {i+1}. {msg['speaker']}: {preview}")
    
    try:
        # 发送请求
        print(f"\n🚀 发送请求到: {DETECT_BOUNDARIES_ENDPOINT}")
        response = requests.post(
            DETECT_BOUNDARIES_ENDPOINT,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60  # 60秒超时
        )
        
        print(f"📡 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 边界检测成功!")
            print(f"📊 检测结果:")
            print(f"   • 总消息数: {result.get('total_messages', 0)}")
            print(f"   • 检测到的段落数: {result.get('segments_detected', 0)}")
            
            # 显示每个段落的详细信息
            boundaries = result.get('boundaries', [])
            for i, boundary in enumerate(boundaries):
                print(f"\n📄 段落 {i+1}:")
                print(f"   • 消息范围: {boundary['start_index']} - {boundary['end_index']}")
                print(f"   • 消息数量: {boundary['message_count']}")
                print(f"   • 分割原因: {boundary['reason']}")
                
                # 显示段落中的消息概览
                segment_messages = boundary.get('messages', [])
                print(f"   • 消息内容:")
                for j, msg in enumerate(segment_messages):
                    content_preview = msg.get("content", msg.get("text", ""))[:40] + "..."
                    print(f"     {j+1}. {msg.get('speaker', 'Unknown')}: {content_preview}")
            
        elif response.status_code == 400:
            error_result = response.json()
            print(f"❌ 请求错误: {error_result.get('message', 'Unknown error')}")
            
        elif response.status_code == 500:
            error_result = response.json()
            print(f"❌ 服务器错误: {error_result.get('message', 'Internal server error')}")
            
        else:
            print(f"❌ 未知错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器。请确保 test_api.py 服务正在运行 (python test_api.py)")
    except requests.exceptions.Timeout:
        print("❌ 请求超时。边界检测可能需要较长时间。")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_invalid_requests():
    """测试无效请求的处理"""
    print(f"\n🧪 测试无效请求处理")
    print("=" * 30)
    
    test_cases = [
        {
            "name": "缺少 version 字段",
            "data": {"messages": [{"speaker": "test", "content": "test"}]}
        },
        {
            "name": "缺少 messages 字段", 
            "data": {"version": "test"}
        },
        {
            "name": "空的 messages 列表",
            "data": {"version": "test", "messages": []}
        },
        {
            "name": "消息格式错误 - 缺少 content/text",
            "data": {"version": "test", "messages": [{"speaker": "test"}]}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧮 测试: {test_case['name']}")
        try:
            response = requests.post(
                DETECT_BOUNDARIES_ENDPOINT,
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 400:
                result = response.json()
                print(f"   ✅ 正确返回 400 错误: {result.get('message', 'No message')}")
            else:
                print(f"   ❌ 意外的响应状态: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")

def main():
    """主函数"""
    print("🚀 Nemori 边界检测 API 测试")
    print("=" * 60)
    
    # 测试正常的边界检测
    test_detect_boundaries()
    
    # 测试无效请求处理
    test_invalid_requests()
    
    print(f"\n🎉 测试完成!")
    print("=" * 60)
    print("💡 API 使用说明:")
    print("   • 端点: POST /api/boundaries/detect")
    print("   • 必需字段: version, messages")
    print("   • 消息格式: {speaker, content/text, timestamp(可选)}")
    print("   • 返回: 检测到的对话段落边界信息")

if __name__ == "__main__":
    main()