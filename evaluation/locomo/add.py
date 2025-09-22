"""
Memory System
"""

import json
import os
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Import new memory system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src import MemorySystem, MemoryConfig
from src.models import Message

load_dotenv()


class MemorySystemAdd:
    """
    Improved Memory System v3.0 - Add Memory for Evaluation
    
    Configured for Prediction-Correction Engine (Simplified Two-Step):
    - Uses prediction-correction engine for semantic memory generation
    - Two-step process: prediction → direct knowledge extraction
    - Cold start support for users without existing knowledge
    - Configurable semantic generation workers
    - Progress monitoring for semantic tasks
    - Automatic retry for failed semantic generation
    """
    
    def __init__(self, 
                 data_path=None, 
                 batch_size=1,
                 storage_path="./evaluation_memories_v3",
                 model="gpt-4.1-mini",
                 language="en",
                 enable_semantic_extraction=True,
                 semantic_generation_workers=8,  
                 semantic_wait_timeout=120):     
        """
        Initialize improved memory system
        """
        # Create configuration
        self.config = MemoryConfig(
            # LLM and Embedding settings
            llm_model=model,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            
            # Storage settings
            storage_path=storage_path,
            
            # Buffer settings
            buffer_size_min=2,
            buffer_size_max=50,
            
            # Boundary detection settings
            boundary_confidence_threshold=0,
            enable_smart_boundary=True,
            
            # Episode settings
            episode_min_messages=2,
            episode_max_messages=50,
            
            # Semantic memory settings
            enable_semantic_memory=enable_semantic_extraction,
            enable_prediction_correction=True,    # Enable prediction-correction engine (simplified two-step)
            extract_semantic_per_episode=False,   # Use two-step prediction-correction method instead of single episode  
            
            # Search settings
            search_top_k_episodes=15,
            search_top_k_semantic=15,
            
            # Performance settings
            use_faiss=True,
            faiss_index_type="Flat",  
            batch_size=64,
            max_workers=8,
            semantic_generation_workers=semantic_generation_workers,  
            
            # Cache settings
            enable_cache=True,
            cache_size=1000,
            cache_ttl_seconds=3600,
            
            # Language settings
            language=language
        )
        
        # Initialize memory system
        self.memory_system = MemorySystem(self.config)
        
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.semantic_wait_timeout = semantic_wait_timeout
        
        self.semantic_generation_tracker = {}
        self.tracker_lock = threading.Lock()
        
        if data_path:
            self.load_data()
            
        # Print configuration
        print(f"\n=== Improved Memory System v3.0 Configuration ===")
        print(f"Model: {model}")
        print(f"Language: {language}")
        print(f"Multimodal Support: Enabled (blip_caption, query, img_url)")
        print(f"Semantic Extraction: {'Enabled' if enable_semantic_extraction else 'Disabled'}")
        if enable_semantic_extraction:
            print(f"  - Mode: Prediction-Correction Engine (Simplified Two-Step)")
            print(f"  - Step 1: Predict episode content from existing knowledge")
            print(f"  - Step 2: Compare prediction with actual content and extract new knowledge")
            print(f"  - Cold start: Direct extraction for users with no existing knowledge")
            print(f"  - Semantic Generation Workers: {semantic_generation_workers}")
            print(f"  - Semantic Wait Timeout: {semantic_wait_timeout}s")
        print(f"Batch Size: {batch_size}")
        print(f"Storage Path: {storage_path}")
        print("===================================================\n")

    def load_data(self):
        """Load data file"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def track_semantic_generation(self, user_id: str, episodes_created: int):
        with self.tracker_lock:
            if user_id not in self.semantic_generation_tracker:
                self.semantic_generation_tracker[user_id] = {
                    "episodes_created": 0,
                    "semantic_tasks_scheduled": 0,
                    "semantic_tasks_completed": 0,
                    "start_time": time.time()
                }
            self.semantic_generation_tracker[user_id]["episodes_created"] += episodes_created

    def add_memory(self, user_id: str, messages: List[Message], retries: int = 3) -> Dict[str, Any]:
        """
        Add memory to system with retry logic and tracking
        """
        for attempt in range(retries):
            try:
                # Convert Message objects to dictionaries
                message_dicts = []
                for msg in messages:
                    message_dict = {
                        "role": msg.role,
                        "content": msg.content,
                        "metadata": msg.metadata.copy()
                    }
                    if hasattr(msg, 'timestamp') and msg.timestamp:
                        message_dict["timestamp"] = msg.timestamp.isoformat()
                        message_dict["metadata"]["timestamp"] = msg.timestamp.isoformat()
                    message_dicts.append(message_dict)
                
                # Add messages to memory system
                result = self.memory_system.add_messages(user_id, message_dicts)
                
                if result.get("episodes_created"):
                    self.track_semantic_generation(user_id, len(result["episodes_created"]))
                    if result.get("semantic_tasks_scheduled"):
                        with self.tracker_lock:
                            self.semantic_generation_tracker[user_id]["semantic_tasks_scheduled"] += result["semantic_tasks_scheduled"]
                
                return result
            except Exception as e:
                if attempt < retries - 1:
                    print(f"  Retry {attempt + 1}/{retries} for user {user_id}: {e}")
                    time.sleep(1)
                    continue
                else:
                    print(f"  Failed to add memory for user {user_id} after {retries} attempts: {e}")
                    raise e

    def parse_dataset_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from dataset format"""
        try:
            if isinstance(timestamp_str, str):
                # Remove extra spaces and normalize
                timestamp_str = ' '.join(timestamp_str.split())
                
                # Handle formats like "1:56 pm on 8 May, 2023"
                if ' on ' in timestamp_str:
                    time_part, date_part = timestamp_str.split(' on ')
                    
                    # Parse time part (e.g., "1:56 pm")
                    time_part = time_part.strip()
                    if 'pm' in time_part.lower():
                        time_str = time_part.lower().replace('pm', '').strip()
                        is_pm = True
                    elif 'am' in time_part.lower():
                        time_str = time_part.lower().replace('am', '').strip()
                        is_pm = False
                    else:
                        time_str = time_part.strip()
                        is_pm = False
                    
                    # Parse hour and minute
                    if ':' in time_str:
                        hour_str, minute_str = time_str.split(':')
                        hour = int(hour_str)
                        minute = int(minute_str)
                    else:
                        hour = int(time_str)
                        minute = 0
                    
                    # Convert to 24-hour format
                    if is_pm and hour != 12:
                        hour += 12
                    elif not is_pm and hour == 12:
                        hour = 0
                    
                    # Parse date part (e.g., "8 May, 2023")
                    date_part = date_part.strip()
                    
                    # Handle formats like "8 May, 2023" or "May 8, 2023"
                    month_names = {
                        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
                        'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                        'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
                    }
                    
                    parts = date_part.replace(',', '').split()
                    
                    # Try different date formats
                    day, month, year = None, None, None
                    
                    for part in parts:
                        if part.lower() in month_names:
                            month = month_names[part.lower()]
                        elif part.isdigit():
                            num = int(part)
                            if num > 31:  # Likely year
                                year = num
                            else:  # Likely day
                                day = num
                    
                    # Default values if parsing failed
                    if not day:
                        day = 1
                    if not month:
                        month = 1
                    if not year:
                        year = 2023
                    
                    # Create datetime object
                    dt = datetime(year=year, month=month, day=day, hour=hour, minute=minute)
                    return dt
                
                # Try ISO format as fallback
                elif 'T' in timestamp_str:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Try other common formats
                else:
                    # Fallback to current time
                    print(f"Warning: Could not parse timestamp '{timestamp_str}', using current time")
                    return datetime.now()
            
            else:
                return datetime.now()
                
        except Exception as e:
            print(f"Warning: Error parsing timestamp '{timestamp_str}': {e}, using current time")
            return datetime.now()

    def add_memories_for_speaker(self, speaker_id: str, messages: List[Dict[str, Any]], 
                                timestamp: str, desc: str) -> List[Dict[str, Any]]:
        """Add memories for specific speaker with improved tracking"""
        episodes_created = []
        
        # Convert message dicts to Message objects
        message_objects = []
        for msg_data in messages:
            msg_timestamp = msg_data.get("timestamp", timestamp)
            
            # Prepare metadata with multimodal information
            metadata = {
                "original_speaker": msg_data.get("speaker", "unknown"),
                "dataset_timestamp": msg_timestamp,
                "timestamp": msg_timestamp  # Also add timestamp to metadata for compatibility
            }
            
            # Add multimodal information to metadata if available
            if "multimodal_info" in msg_data:
                multimodal_info = msg_data["multimodal_info"]
                metadata.update({
                    "original_text": multimodal_info.get("original_text"),
                    "blip_caption": multimodal_info.get("blip_caption"),
                    "search_query": multimodal_info.get("query"),
                    "image_url": multimodal_info.get("img_url"),
                    "has_multimodal_content": bool(
                        multimodal_info.get("blip_caption") or 
                        multimodal_info.get("query") or 
                        multimodal_info.get("img_url")
                    )
                })
            
            message_obj = Message(
                role=msg_data.get("role", "user"),
                content=msg_data.get("content", ""),
                timestamp=self.parse_dataset_timestamp(msg_timestamp),
                metadata=metadata
            )
            message_objects.append(message_obj)
        
        # Add messages in batches
        for i in tqdm(range(0, len(message_objects), self.batch_size), desc=desc):
            batch_messages = message_objects[i:i + self.batch_size]
            
            try:
                # Add batch to memory system
                result = self.add_memory(speaker_id, batch_messages)
                
                # Track episodes created
                if result.get("episodes_created"):
                    episodes_created.extend(result["episodes_created"])
                    for ep in result["episodes_created"]:
                        print(f"  Created episode: {ep.get('title', 'Untitled')}")
                
                # Track semantic generation
                if result.get("semantic_tasks_scheduled"):
                    print(f"  Scheduled {result['semantic_tasks_scheduled']} semantic generation tasks")
                        
            except Exception as e:
                print(f"  Error adding batch {i//self.batch_size + 1}: {e}")
                continue
        
        # Force episode creation if needed
        try:
            print(f"  Forcing episode creation for any remaining messages...")
            final_episode = self.memory_system.force_episode_creation(speaker_id)
            if final_episode:
                episodes_created.append(final_episode)
                print(f"  Created final episode: {final_episode.get('title', 'Untitled')}")
                
                self.track_semantic_generation(speaker_id, 1)
                if final_episode.get("semantic_generation_scheduled"):
                    with self.tracker_lock:
                        self.semantic_generation_tracker[speaker_id]["semantic_tasks_scheduled"] += 1
                        
        except Exception as e:
            print(f"  Error forcing episode creation: {e}")
            
        print(f"  Total episodes created: {len(episodes_created)}")
        return episodes_created

    def monitor_semantic_generation_progress(self):
        try:
            while True:
                time.sleep(5) 
                
                with self.tracker_lock:
                    all_completed = True
                    total_scheduled = 0
                    total_active = 0
                    
                    for user_id, tracker in self.semantic_generation_tracker.items():
                        status = self.memory_system.get_semantic_generation_status(user_id)
                        active_tasks = status.get("active_tasks", 0)
                        
                        if active_tasks > 0:
                            all_completed = False
                            total_active += active_tasks
                        
                        total_scheduled += tracker["semantic_tasks_scheduled"]
                    
                    if total_scheduled > 0:
                        progress = (total_scheduled - total_active) / total_scheduled * 100
                        logger.debug(f"Semantic Generation Background Progress: {progress:.1f}% ({total_scheduled - total_active}/{total_scheduled} completed)")
                    
                    if all_completed and total_scheduled > 0:
                        logger.info("Background monitor: All semantic memory generation tasks completed!")
                        break
        except Exception as e:
            logger.error(f"Error in background semantic generation monitor: {e}")

    def process_conversation(self, item: Dict[str, Any], idx: int):
        """Process single conversation"""
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        print(f"\nProcessing conversation {idx}: {speaker_a} vs {speaker_b}")

        # Clear existing data for speaker A only (search uses single user)
        try:
            self.memory_system.delete_user_data(speaker_a_user_id)
        except Exception as e:
            print(f"  Warning: Error clearing user data: {e}")

        # Process each conversation segment
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            if date_time_key not in conversation:
                continue
                
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            if not chats:
                continue

            messages_a = []
            
            # Build message lists with multimodal content
            for chat in chats:
                speaker = chat["speaker"]
                text = chat["text"]
                
                # Build enhanced content with multimodal information
                content_parts = [text]
                
                # Add image caption if available
                if "blip_caption" in chat and chat["blip_caption"]:
                    content_parts.append(f"[Image: {chat['blip_caption']}]")
                
                # Add search query if available
                if "query" in chat and chat["query"]:
                    content_parts.append(f"[Search: {chat['query']}]")
            
                
                enhanced_content = " ".join(content_parts)
                
                if speaker == speaker_a:
                    messages_a.append({
                        "role": f"{speaker_a}",
                        "content": enhanced_content,
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "multimodal_info": {
                            "original_text": text,
                            "blip_caption": chat.get("blip_caption"),
                            "query": chat.get("query"),
                        }
                    })
                elif speaker == speaker_b:
                    messages_a.append({
                        "role": f"{speaker_b}",
                        "content": enhanced_content,
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "multimodal_info": {
                            "original_text": text,
                            "blip_caption": chat.get("blip_caption"),
                            "query": chat.get("query"),
                        }
                    })
                else:
                    print(f"  Warning: Unknown speaker: {speaker}")
                    continue

            # Add memories only for speaker A
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages_a, timestamp, f"Adding Episodes for {speaker_a}"),
            )

            thread_a.start()
            thread_a.join()

        print(f"Conversation {idx} processed successfully")

    def process_all_conversations(self, max_workers: int = 10):
        """Process all conversations with improved semantic generation monitoring"""
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
            
        print(f"Processing {len(self.data)} conversations with {max_workers} workers...")
        
        monitor_thread = None
        if self.config.enable_semantic_memory and os.environ.get("ENABLE_BACKGROUND_MONITOR", "false").lower() == "true":
            monitor_thread = threading.Thread(target=self.monitor_semantic_generation_progress, daemon=True)
            monitor_thread.start()
            print("Background semantic generation monitor started (set ENABLE_BACKGROUND_MONITOR=false to disable)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_conversation, item, idx) 
                for idx, item in enumerate(self.data)
            ]

            # Wait for all to complete
            for future in tqdm(futures, desc="Processing conversations"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing conversation: {e}")
                    continue
                
        print("\nAll conversations processed successfully!")
        
        if self.config.enable_semantic_memory:
            print("\nWaiting for async semantic memory generation to complete...")
            
            all_user_ids = set()
            for idx, item in enumerate(self.data):
                conversation = item["conversation"]
                speaker_a = conversation["speaker_a"]
                speaker_b = conversation["speaker_b"]
                all_user_ids.add(f"{speaker_a}_{idx}")
                all_user_ids.add(f"{speaker_b}_{idx}")

            completed_count = 0
            failed_users = []
            start_wait_time = time.time()
            
            pbar = tqdm(total=len(all_user_ids), desc="Semantic generation progress")
            
            remaining_users = all_user_ids.copy()
            last_active_count = -1
            
            while remaining_users and (time.time() - start_wait_time) < self.semantic_wait_timeout:
                completed_this_round = []
                total_active = 0
                
                for user_id in list(remaining_users):
                    status = self.memory_system.get_semantic_generation_status(user_id)
                    active_tasks = status.get("active_tasks", 0)
                    
                    if active_tasks == 0:
                        completed_this_round.append(user_id)
                        completed_count += 1
                    else:
                        total_active += active_tasks
                
                for user_id in completed_this_round:
                    remaining_users.remove(user_id)
                    pbar.update(1)
                
                if total_active != last_active_count:
                    pbar.set_postfix({"active_tasks": total_active})
                    last_active_count = total_active
                
                if remaining_users:
                    time.sleep(0.5)
            
            pbar.close()
            
            if remaining_users:
                failed_users = list(remaining_users)
                print(f"\n⚠️ {len(failed_users)} users timed out after {self.semantic_wait_timeout}s")
            
            print(f"\nSemantic generation completed for {completed_count}/{len(all_user_ids)} users")
            
            if failed_users:
                print(f"\n⚠️ {len(failed_users)} users had semantic generation issues:")
                print(f"   You can run resume_semantic_generation.py to complete them")
                
                failed_users_file = os.path.join(self.config.storage_path, "failed_semantic_users.json")
                with open(failed_users_file, "w") as f:
                    json.dump(failed_users, f, indent=2)
                print(f"   Failed users saved to: {failed_users_file}")
        
        # Print final statistics
        self.print_statistics()
        
        print("Closing memory system...")
        self.memory_system.__exit__(None, None, None)

    def print_statistics(self):
        """Print system statistics with semantic generation details"""
        try:
            stats = self.memory_system.get_stats()
            system_stats = stats.get('system', {})
            
            print(f"\n=== Final Statistics ===")
            print(f"Messages processed: {system_stats.get('messages_processed', 0)}")
            print(f"Episodes created: {system_stats.get('episodes_created', 0)}")  
            print(f"Semantic memories created: {system_stats.get('semantic_memories_created', 0)}")
            
            if self.semantic_generation_tracker:
                print(f"\nSemantic Generation Tracking:")
                total_episodes = 0
                total_scheduled = 0
                
                for user_id, tracker in self.semantic_generation_tracker.items():
                    total_episodes += tracker["episodes_created"]
                    total_scheduled += tracker["semantic_tasks_scheduled"]
                
                print(f"  Total episodes triggering semantic generation: {total_episodes}")
                print(f"  Total semantic tasks scheduled: {total_scheduled}")
            
            print(f"\nStorage path: {self.config.storage_path}")
            print("============================\n")
        except Exception as e:
            print(f"Error getting statistics: {e}")


def main():
    """Main function for testing"""
    # Example usage
    adder = MemorySystemAdd(
        data_path="./dataset/locomo10.json",  # Update with actual path
        batch_size=1,
        storage_path="./memories",
        model="gpt-4.1-mini",
        language="en",
        enable_semantic_extraction=True,
        semantic_generation_workers=20,  
        semantic_wait_timeout=3600  
    )
    
    adder.process_all_conversations(max_workers=10)


if __name__ == "__main__":
    main() 