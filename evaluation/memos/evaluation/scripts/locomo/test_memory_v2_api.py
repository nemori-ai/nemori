#!/usr/bin/env python3
"""
测试 update_memory V2 API 的示例脚本
该脚本展示如何使用预分割的边界信息直接进行记忆更新
"""

import requests
import json
from datetime import datetime, timedelta

# API 配置
API_BASE_URL = "http://localhost:5001"
UPDATE_MEMORY_V2_ENDPOINT = f"{API_BASE_URL}/api/memory/update-v2"
DETECT_BOUNDARIES_ENDPOINT = f"{API_BASE_URL}/api/boundaries/detect"

def create_test_data():
    """创建测试数据"""
    base_time = datetime.now()
    
    # 测试消息 - 包含两个不同主题
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
        # 主题转换 - 从机器学习转到数据预处理
        {
            "speaker": "张三", 
            "content": "数据预处理方面有什么建议吗？我的数据集有很多缺失值。",
            "timestamp": (base_time + timedelta(hours=4)).isoformat() + "Z"
        },
        {
            "speaker": "李四",
            "content": "缺失值可以用均值填充、前向填充，或者直接删除。要看具体业务场景，时间序列数据建议前向填充。",
            "timestamp": (base_time + timedelta(hours=4, minutes=2)).isoformat() + "Z"
        }
    ]
    
    return messages

def create_manual_boundaries():
    """创建手动分割的边界 - 演示跳过自动检测"""
    boundaries = [
        {
            "start_index": 0,
            "end_index": 3,
            "reason": "机器学习库讨论段落"
        },
        {
            "start_index": 4, 
            "end_index": 5,
            "reason": "数据预处理讨论段落"
        }
    ]
    
    return boundaries

def test_complete_workflow():
    """测试完整的工作流程：先检测边界，然后使用V2更新记忆"""
    print("🔄 测试完整工作流程（边界检测 + V2记忆更新）")
    print("=" * 60)
    
    messages = create_test_data()
    version = "workflow_test"
    
    print(f"📋 消息数量: {len(messages)}")
    for i, msg in enumerate(messages):
        preview = msg["content"][:40] + "..." if len(msg["content"]) > 40 else msg["content"]
        print(f"   {i}. {msg['speaker']}: {preview}")
    
    # 第一步：检测边界
    print(f"\n🔍 第一步：检测边界")
    try:
        boundary_response = requests.post(
            DETECT_BOUNDARIES_ENDPOINT,
            json={
                "version": version,
                "messages": messages
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if boundary_response.status_code != 200:
            print(f"❌ 边界检测失败: {boundary_response.status_code}")
            print(boundary_response.json())
            return
        
        boundary_result = boundary_response.json()
        detected_boundaries = boundary_result.get("boundaries", [])
        
        print(f"✅ 检测到 {len(detected_boundaries)} 个边界段落")
        for i, boundary in enumerate(detected_boundaries):
            print(f"   段落 {i+1}: 消息 {boundary['start_index']}-{boundary['end_index']}")
            print(f"   原因: {boundary['reason']}")
        
        # 第二步：使用检测到的边界进行记忆更新
        print(f"\n🏗️ 第二步：使用检测到的边界进行记忆更新")
        
        # 简化边界格式（只保留必需字段）
        simplified_boundaries = []
        for boundary in detected_boundaries:
            simplified_boundaries.append({
                "start_index": boundary["start_index"],
                "end_index": boundary["end_index"],
                "reason": boundary["reason"]
            })
        
        memory_response = requests.post(
            UPDATE_MEMORY_V2_ENDPOINT,
            json={
                "version": version,
                "messages": messages,
                "boundaries": simplified_boundaries
            },
            headers={"Content-Type": "application/json"},
            timeout=120  # 更长的超时，因为要进行LLM处理
        )
        
        if memory_response.status_code == 200:
            memory_result = memory_response.json()
            print("✅ 记忆更新成功!")
            print(f"📊 处理结果:")
            print(f"   • Episodes 创建: {memory_result.get('episodes_created', 0)}")
            print(f"   • 语义概念发现: {memory_result.get('semantic_concepts', 0)}")
            print(f"   • 处理的说话人: {memory_result.get('processed_speakers', 0)}")
            print(f"   • 使用的边界段落: {memory_result.get('boundary_segments_used', 0)}")
            print(f"   • 处理方法: {memory_result.get('method', 'unknown')}")
        else:
            print(f"❌ 记忆更新失败: {memory_response.status_code}")
            print(memory_response.json())
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器。请确保 test_api.py 服务正在运行")
    except requests.exceptions.Timeout:
        print("❌ 请求超时。处理可能需要较长时间。")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_manual_boundaries():
    """测试使用手动分割的边界"""
    print("\n🎯 测试手动边界分割（跳过自动检测）")
    print("=" * 60)
    
    messages = create_test_data()
    manual_boundaries = create_manual_boundaries()
    version = "manual_boundaries_test"
    
    print(f"📋 消息数量: {len(messages)}")
    print(f"🔧 手动边界数量: {len(manual_boundaries)}")
    
    for i, boundary in enumerate(manual_boundaries):
        print(f"   边界 {i+1}: 消息 {boundary['start_index']}-{boundary['end_index']}")
        print(f"   原因: {boundary['reason']}")
    
    try:
        response = requests.post(
            UPDATE_MEMORY_V2_ENDPOINT,
            json={
                "version": version,
                "messages": messages,
                "boundaries": manual_boundaries
            },
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 记忆更新成功!")
            print(f"📊 处理结果:")
            print(f"   • Episodes 创建: {result.get('episodes_created', 0)}")
            print(f"   • 语义概念发现: {result.get('semantic_concepts', 0)}")
            print(f"   • 处理的说话人: {result.get('processed_speakers', 0)}")
            print(f"   • 使用的边界段落: {result.get('boundary_segments_used', 0)}")
        else:
            print(f"❌ 记忆更新失败: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def test_invalid_requests():
    """测试无效请求的处理"""
    print(f"\n🧪 测试无效请求处理")
    print("=" * 30)
    
    test_cases = [
        {
            "name": "缺少 version 字段",
            "data": {
                "messages": [{"speaker": "test", "content": "test"}],
                "boundaries": [{"start_index": 0, "end_index": 0}]
            }
        },
        {
            "name": "缺少 messages 字段", 
            "data": {
                "version": "test",
                "boundaries": [{"start_index": 0, "end_index": 0}]
            }
        },
        {
            "name": "缺少 boundaries 字段",
            "data": {
                "version": "test",
                "messages": [{"speaker": "test", "content": "test"}]
            }
        },
        {
            "name": "边界索引超出范围",
            "data": {
                "version": "test",
                "messages": [{"speaker": "test", "content": "test"}],
                "boundaries": [{"start_index": 0, "end_index": 5}]  # 超出消息范围
            }
        },
        {
            "name": "无效的边界索引类型",
            "data": {
                "version": "test", 
                "messages": [{"speaker": "test", "content": "test"}],
                "boundaries": [{"start_index": "0", "end_index": "0"}]  # 字符串而非整数
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧮 测试: {test_case['name']}")
        try:
            response = requests.post(
                UPDATE_MEMORY_V2_ENDPOINT,
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
    print("🚀 Nemori Memory Update V2 API 测试")
    print("=" * 70)
    
    # 测试完整工作流程
    test_complete_workflow()
    
    # 测试手动边界
    test_manual_boundaries()
    
    # 测试无效请求
    test_invalid_requests()
    
    print(f"\n🎉 测试完成!")
    print("=" * 70)
    print("💡 V2 API 优势:")
    print("   • 跳过边界检测：直接使用预分割的边界信息")
    print("   • 提高效率：减少LLM调用次数")
    print("   • 更多控制：可以使用自定义或外部边界检测结果")
    print("   • 灵活性：支持人工调整或其他算法的边界结果")
    print("\n📋 两种使用模式:")
    print("   1. 完整流程：/api/boundaries/detect → /api/memory/update-v2")
    print("   2. 直接更新：手动边界 → /api/memory/update-v2")

if __name__ == "__main__":
    main()