#!/usr/bin/env python3
"""
Clean redundant metadata from existing episode files
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

def clean_message_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean redundant information from message metadata
    
    Args:
        metadata: Original metadata
        
    Returns:
        Cleaned metadata with redundant fields removed
    """
    if not metadata:
        return {}
    
    # 需要移除的重复字段
    redundant_fields = {
        'original_text',  # 与content重复
        'timestamp'       # 与主timestamp重复
    }
    
    # 保留有价值的唯一字段
    valuable_fields = {
        'original_speaker',     # 可能与role不同，保留原始说话人信息
        'dataset_timestamp',    # 原始数据集时间格式，有追溯价值
        'image_url',           # 多模态内容
        'blip_caption',        # 图像描述
        'search_query',        # 搜索查询
        'has_multimodal_content'  # 内容类型标识
    }
    
    # 只保留有价值的字段，移除重复字段
    cleaned_metadata = {}
    for key, value in metadata.items():
        if key not in redundant_fields:
            # 只保留非空且有价值的字段
            if value is not None or key in valuable_fields:
                cleaned_metadata[key] = value
    
    return cleaned_metadata

def clean_episode_file(file_path: Path) -> int:
    """
    Clean a single episode file
    
    Args:
        file_path: Path to the episode file
        
    Returns:
        Number of episodes processed
    """
    print(f"Processing {file_path}")
    
    episodes = []
    processed_count = 0
    
    try:
        # Read all episodes from JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    episode = json.loads(line)
                    
                    # Clean original_messages metadata
                    if 'original_messages' in episode:
                        for message in episode['original_messages']:
                            if 'metadata' in message:
                                message['metadata'] = clean_message_metadata(message['metadata'])
                    
                    episodes.append(episode)
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {file_path}: {e}")
                    continue
        
        # Write cleaned episodes back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            for episode in episodes:
                f.write(json.dumps(episode, ensure_ascii=False) + '\n')
        
        print(f"✅ Cleaned {processed_count} episodes in {file_path}")
        return processed_count
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return 0

def clean_directory(directory: Path) -> Dict[str, int]:
    """
    Clean all episode files in a directory
    
    Args:
        directory: Directory containing episode files
        
    Returns:
        Statistics of cleaning process
    """
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return {"files": 0, "episodes": 0}
    
    stats = {"files": 0, "episodes": 0}
    
    # Find all .jsonl files in episodes subdirectory
    episodes_dir = directory / "episodes"
    if not episodes_dir.exists():
        print(f"Episodes directory {episodes_dir} does not exist")
        return stats
    
    episode_files = list(episodes_dir.glob("*.jsonl"))
    
    if not episode_files:
        print(f"No .jsonl files found in {episodes_dir}")
        return stats
    
    print(f"Found {len(episode_files)} episode files to clean")
    
    for file_path in episode_files:
        episode_count = clean_episode_file(file_path)
        if episode_count > 0:
            stats["files"] += 1
            stats["episodes"] += episode_count
    
    return stats

def main():
    """Main function"""
    print("🧹 Starting redundant metadata cleanup...")
    
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Define directories to clean
    directories_to_clean = [
        project_root / "evaluation" / "memories_1",
        project_root / "evaluation" / "memories_w"
    ]
    
    total_stats = {"files": 0, "episodes": 0}
    
    for directory in directories_to_clean:
        print(f"\n📁 Cleaning directory: {directory}")
        dir_stats = clean_directory(directory)
        total_stats["files"] += dir_stats["files"]
        total_stats["episodes"] += dir_stats["episodes"]
    
    print(f"\n✨ Cleanup completed!")
    print(f"📊 Total files cleaned: {total_stats['files']}")
    print(f"📊 Total episodes processed: {total_stats['episodes']}")

if __name__ == "__main__":
    main()
