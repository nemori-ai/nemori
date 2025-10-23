import json
import os
import sys
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.core.memory_system import MemorySystem
from src.config import MemoryConfig

load_dotenv()

logger = logging.getLogger(__name__)


class LongMemEvalMemorySystemAdd:
    def __init__(self, data_path, batch_size=5,
                 storage_path="./longmemeval_memories",
                 model=os.getenv("OPENAI_MODEL"),
                 language="en",
                 enable_semantic_memory=True,
                 semantic_generation_workers=8,  # New: Number of semantic generation worker threads
                 semantic_wait_timeout=3600,      # New: Semantic generation wait timeout
                 max_workers=10):
        """
        Initialize LongMemEval New Memory System Adder
        
        Args:
            data_path: Path to data file
            batch_size: Batch processing size
            storage_path: Storage path
            model: LLM model
            language: Language setting
            enable_semantic_memory: Whether to enable semantic memory
            semantic_generation_workers: Number of semantic generation worker threads
            semantic_wait_timeout: Semantic generation wait timeout (seconds)
            max_workers: Maximum number of parallel worker threads
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.storage_path = storage_path
        self.semantic_wait_timeout = semantic_wait_timeout
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"\n=== LongMemEval Memory System v3 Improved Configuration ===")
        print(f"Total questions: {len(self.data)}")
        print(f"Model: {model}")
        print(f"Language: {language}")
        print(f"Semantic Memory: {'Enabled' if enable_semantic_memory else 'Disabled'}")
        if enable_semantic_memory:
            print(f"  - Prediction-Correction Mode: Enabled")
            print(f"  - Semantic Generation Workers: {semantic_generation_workers}")
            print(f"  - Semantic Wait Timeout: {semantic_wait_timeout}s")
            print(f"  - Use Simplified Extraction: True")
        print(f"Batch Size: {batch_size}")
        print(f"Storage Path: {storage_path}")
        print(f"Max Workers: {max_workers}")
        print("========================================================\n")
        
        # Save configuration
        self.model = model
        self.language = language
        self.enable_semantic_memory = enable_semantic_memory
        self.max_workers = max_workers
        
        # Track semantic memory generation progress
        self.semantic_generation_tracker = {}
        self.tracker_lock = threading.Lock()
        
        # Create shared configuration
        self.config = MemoryConfig(
            # LLM and Embedding settings
            llm_model=model,
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
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
            enable_semantic_memory=enable_semantic_memory,
            semantic_similarity_threshold=0.5,
            enable_prediction_correction=enable_semantic_memory,
            
            # Search settings
            search_top_k_episodes=15,
            search_top_k_semantic=15,
            
            # Vector Database Configuration
            vector_db_type="chroma",
            chroma_persist_directory=f"{storage_path}/chroma_db",
            chroma_collection_prefix="nemori_longmem",
            
            # Performance settings
            batch_size=64,
            max_workers=8,
            semantic_generation_workers=semantic_generation_workers,  # New configuration
            
            # Cache settings
            enable_cache=True,
            cache_size=1000,
            cache_ttl_seconds=3600,
            
            # Language settings
            language=language
        )
        
    def track_semantic_generation(self, user_id: str, episodes_created: int):
        """Track semantic memory generation progress"""
        with self.tracker_lock:
            if user_id not in self.semantic_generation_tracker:
                self.semantic_generation_tracker[user_id] = {
                    "episodes_created": 0,
                    "semantic_tasks_scheduled": 0,
                    "semantic_tasks_completed": 0,
                    "start_time": time.time()
                }
            self.semantic_generation_tracker[user_id]["episodes_created"] += episodes_created

    def monitor_semantic_generation_progress(self, user_id: str):
        """Monitor semantic memory generation progress for a single user"""
        try:
            start_time = time.time()
            while (time.time() - start_time) < self.semantic_wait_timeout:
                status = self.memory_system.get_semantic_generation_status(user_id)
                active_tasks = status.get("active_tasks", 0)
                
                if active_tasks == 0:
                    logger.debug(f"Semantic generation completed for user {user_id}")
                    return True
                
                # Check once per second
                time.sleep(1)
            
            logger.warning(f"Semantic generation timeout for user {user_id} after {self.semantic_wait_timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"Error monitoring semantic generation for user {user_id}: {e}")
            return False
        
    def process_question(self, question_data, question_idx):
        """
        Create independent memory system for a single question
        
        Args:
            question_data: Question data
            question_idx: Question index
        """
        question_id = question_data['question_id']
        
        # Create memory system dedicated to this question
        memory_system = MemorySystem(self.config)
        self.memory_system = memory_system  # Save reference for monitoring
        
        # Use question ID as user ID
        user_id = f"question_{question_id}"
        
        # Clean up old data
        memory_system.delete_user_data(user_id)
        
        print(f"\nProcessing question {question_idx + 1}/{len(self.data)}: {question_id}")
        print(f"Question: {question_data['question'][:100]}...")
        
        # 获取所有相关会话
        haystack_sessions = question_data['haystack_sessions']
        haystack_dates = question_data['haystack_dates']
        haystack_session_ids = question_data['haystack_session_ids']
        
        total_messages = 0
        episodes_created = 0
        semantic_memories_created = 0
        
        # Process each session
        for session_idx, (session, date, session_id) in enumerate(
            zip(haystack_sessions, haystack_dates, haystack_session_ids)
        ):
            print(f"  Processing session {session_idx + 1}/{len(haystack_sessions)}: {session_id}")
            
            # Build message list
            messages = []
            for turn_idx, turn in enumerate(session):
                role = turn['role']
                content = turn['content']
                
                # Parse date string to datetime object
                try:
                    # LongMemEval 使用格式: "2023/05/20 (Sat) 02:21"
                    timestamp = datetime.strptime(date, '%Y/%m/%d (%a) %H:%M')
                except Exception as e:
                    print(f"    Warning: Failed to parse date '{date}': {e}, using current time")
                    timestamp = datetime.now()
                
                # Add message
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "metadata": {
                        "session_id": session_id,
                        "turn_index": turn_idx
                    }
                }
                messages.append(message)
            
            # Add messages in batches
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                
                try:
                    result = memory_system.add_messages(
                        owner_id=user_id,
                        messages=batch,
                        metadata={
                            "session_id": session_id,
                            "question_id": question_id,
                            "session_date": date
                        }
                    )
                    
                    # Track semantic memory generation
                    if result.get("episodes_created"):
                        episodes_count = len(result["episodes_created"])
                        episodes_created += episodes_count
                        self.track_semantic_generation(user_id, episodes_count)
                        
                        for ep in result["episodes_created"]:
                            print(f"    Created episode: {ep.get('title', 'Untitled')}")
                    
                    # Track semantic task scheduling
                    if result.get("semantic_tasks_scheduled"):
                        with self.tracker_lock:
                            if user_id in self.semantic_generation_tracker:
                                self.semantic_generation_tracker[user_id]["semantic_tasks_scheduled"] += result["semantic_tasks_scheduled"]
                        print(f"    Scheduled {result['semantic_tasks_scheduled']} semantic generation tasks")
                    
                    total_messages += len(batch)
                    
                except Exception as e:
                    print(f"    Error adding batch: {e}")
                    continue
        
        # Force creation of final episode
        try:
            print(f"  Forcing episode creation for any remaining messages...")
            final_result = memory_system.force_episode_creation(user_id)
            if final_result and final_result.get("episode_created"):
                episodes_created += 1
                print(f"  Created final episode: {final_result.get('title', 'N/A')}")
                
                # Track forcibly created episode
                self.track_semantic_generation(user_id, 1)
                if final_result.get("semantic_generation_scheduled"):
                    with self.tracker_lock:
                        if user_id in self.semantic_generation_tracker:
                            self.semantic_generation_tracker[user_id]["semantic_tasks_scheduled"] += 1
        except Exception as e:
            print(f"  Error forcing episode creation: {e}")
        
        # Improved semantic memory generation waiting logic
        if self.enable_semantic_memory:
            print(f"  Waiting for semantic memory generation...")
            
            # Use improved monitoring method
            success = self.monitor_semantic_generation_progress(user_id)
            
            if success:
                print(f"  ✅ Semantic memory generation completed")
            else:
                print(f"  ⚠️ Semantic memory generation timed out after {self.semantic_wait_timeout}s")
            
            # Get final count of semantic memories
            try:
                semantic_memories = memory_system.storage["semantic"].list_user_items(user_id)
                semantic_memories_created = len(semantic_memories)
            except Exception as e:
                print(f"  Warning: Could not get semantic memories count: {e}")
                semantic_memories_created = 0
        
        print(f"  Total messages: {total_messages}, Episodes: {episodes_created}, Semantic memories: {semantic_memories_created}")
        
        # Save question metadata
        metadata = {
            "question_id": question_id,
            "question": question_data['question'],
            "answer": question_data['answer'],
            "question_type": question_data['question_type'],
            "question_date": question_data['question_date'],
            "total_sessions": len(haystack_sessions),
            "total_messages": total_messages,
            "episodes_created": episodes_created,
            "semantic_memories_created": semantic_memories_created,
            "answer_session_ids": question_data.get('answer_session_ids', []),
            "processing_timestamp": datetime.now().isoformat(),
            "semantic_generation_successful": success if self.enable_semantic_memory else True
        }
        
        # Use new system's storage path
        question_storage_path = os.path.join(self.storage_path, "metadata", f"question_{question_id}")
        os.makedirs(question_storage_path, exist_ok=True)
        metadata_path = os.path.join(question_storage_path, "metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Clean up memory system resources
        memory_system.__exit__(None, None, None)
        
        return {
            "question_id": question_id,
            "success": True,
            "episodes_created": episodes_created,
            "semantic_memories_created": semantic_memories_created,
            "total_messages": total_messages,
            "semantic_generation_successful": success if self.enable_semantic_memory else True
        }
    
    def process_all_questions(self, start_idx=0, end_idx=None):
        """
        Process all questions (improved version with semantic generation monitoring)
        
        Args:
            start_idx: Start index
            end_idx: End index (None means process to the end)
        """
        if end_idx is None:
            end_idx = len(self.data)
        
        questions_to_process = self.data[start_idx:end_idx]
        
        print(f"\nProcessing questions {start_idx} to {end_idx}")
        print(f"Total questions to process: {len(questions_to_process)}")
        
        results = []
        
        # Start background monitoring thread (optional, mainly for debugging)
        monitor_thread = None
        if self.enable_semantic_memory and os.environ.get("ENABLE_BACKGROUND_MONITOR", "false").lower() == "true":
            monitor_thread = threading.Thread(target=self.monitor_background_progress, daemon=True)
            monitor_thread.start()
            print("Background semantic generation monitor started (set ENABLE_BACKGROUND_MONITOR=false to disable)")
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for idx, question_data in enumerate(questions_to_process):
                actual_idx = start_idx + idx
                future = executor.submit(self.process_question, question_data, actual_idx)
                futures.append((future, actual_idx, question_data['question_id']))
            
            # Collect results
            for future, idx, question_id in tqdm(futures, desc="Processing questions"):
                try:
                    result = future.result(timeout=6000)  # 10分钟超时
                    results.append(result)
                except Exception as e:
                    print(f"\nError processing question {idx} ({question_id}): {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "question_id": question_id,
                        "success": False,
                        "error": str(e),
                        "semantic_generation_successful": False
                    })
        
        print("\nAll questions processed successfully!")
        
        # Improved semantic memory generation completion check
        if self.enable_semantic_memory:
            print("\nChecking semantic memory generation status...")
            
            # Collect all user IDs
            all_user_ids = []
            for idx, question_data in enumerate(questions_to_process):
                question_id = question_data['question_id']
                user_id = f"question_{question_id}"
                all_user_ids.append(user_id)
            
            # Check failed semantic generation
            failed_semantic_users = []
            successful_semantic = 0
            
            for result in results:
                if result.get("success", False):
                    if result.get("semantic_generation_successful", False):
                        successful_semantic += 1
                    else:
                        failed_semantic_users.append(result["question_id"])
            
            print(f"Semantic generation summary:")
            print(f"  Successful: {successful_semantic}/{len([r for r in results if r.get('success', False)])}")
            
            if failed_semantic_users:
                print(f"  Failed: {len(failed_semantic_users)} questions had semantic generation issues")
                
                # Save list of failed questions
                failed_questions_file = os.path.join(self.storage_path, "failed_semantic_questions.json")
                with open(failed_questions_file, "w") as f:
                    json.dump(failed_semantic_users, f, indent=2)
                print(f"  Failed questions saved to: {failed_questions_file}")
                print(f"  You can use a resume script to complete them")
        
        # Save processing results
        results_path = os.path.join(self.storage_path, f"processing_results_{start_idx}_{end_idx}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print detailed statistics
        self.print_detailed_statistics(results)
        
        print(f"Results saved to: {results_path}")

    def monitor_background_progress(self):
        """Monitor semantic memory generation progress for all users (background thread version)"""
        try:
            while True:
                time.sleep(10)  # Check every 10 seconds
                
                with self.tracker_lock:
                    if not self.semantic_generation_tracker:
                        continue
                    
                    total_scheduled = 0
                    total_active = 0
                    active_users = 0
                    
                    for user_id, tracker in self.semantic_generation_tracker.items():
                        # Get semantic generation status for this user
                        try:
                            status = self.memory_system.get_semantic_generation_status(user_id)
                            active_tasks = status.get("active_tasks", 0)
                            
                            if active_tasks > 0:
                                active_users += 1
                                total_active += active_tasks
                            
                            total_scheduled += tracker["semantic_tasks_scheduled"]
                        except:
                            # Ignore closed memory_system
                            continue
                    
                    if total_scheduled > 0:
                        progress = (total_scheduled - total_active) / total_scheduled * 100
                        logger.debug(f"Background Progress: {progress:.1f}% ({total_scheduled - total_active}/{total_scheduled} completed), {active_users} active users")
                    
                    if total_active == 0 and total_scheduled > 0:
                        logger.info("Background monitor: All semantic memory generation tasks completed!")
                        break
                        
        except Exception as e:
            logger.error(f"Error in background semantic generation monitor: {e}")

    def print_detailed_statistics(self, results):
        """Print detailed statistics"""
        try:
            print(f"\n=== Detailed Processing Statistics ===")
            
            # Basic statistics
            successful = sum(1 for r in results if r.get('success', False))
            total_episodes = sum(r.get('episodes_created', 0) for r in results if r.get('success', False))
            total_semantic = sum(r.get('semantic_memories_created', 0) for r in results if r.get('success', False))
            total_messages = sum(r.get('total_messages', 0) for r in results if r.get('success', False))
            
            print(f"Questions processed: {successful}/{len(results)}")
            print(f"Total episodes created: {total_episodes}")
            print(f"Total semantic memories created: {total_semantic}")
            print(f"Total messages processed: {total_messages}")
            
            # Semantic generation statistics
            if self.enable_semantic_memory:
                successful_semantic = sum(1 for r in results if r.get('success', False) and r.get('semantic_generation_successful', False))
                failed_semantic = sum(1 for r in results if r.get('success', False) and not r.get('semantic_generation_successful', False))
                
                print(f"\nSemantic Generation Statistics:")
                print(f"  Successful: {successful_semantic}")
                print(f"  Failed/Timeout: {failed_semantic}")
                print(f"  Success Rate: {successful_semantic/(successful_semantic+failed_semantic)*100:.1f}%" if (successful_semantic+failed_semantic) > 0 else "  Success Rate: N/A")
            
            # Semantic memory generation tracking statistics
            if self.semantic_generation_tracker:
                print(f"\nSemantic Generation Tracking:")
                total_episodes_tracked = 0
                total_scheduled_tracked = 0
                
                for user_id, tracker in self.semantic_generation_tracker.items():
                    total_episodes_tracked += tracker["episodes_created"]
                    total_scheduled_tracked += tracker["semantic_tasks_scheduled"]
                
                print(f"  Total episodes triggering semantic generation: {total_episodes_tracked}")
                print(f"  Total semantic tasks scheduled: {total_scheduled_tracked}")
            
            print(f"\nStorage path: {self.storage_path}")
            print("=========================================\n")
            
        except Exception as e:
            print(f"Error generating detailed statistics: {e}")
    
    def add_memory_with_retry(self, memory_system, user_id: str, messages: list, metadata: dict, retries: int = 3):
        """
        Add memory to system with retry logic
        
        Args:
            memory_system: Memory system instance
            user_id: User ID
            messages: Message list
            metadata: Metadata
            retries: Number of retries
            
        Returns:
            Addition result
        """
        for attempt in range(retries):
            try:
                result = memory_system.add_messages(
                    owner_id=user_id,
                    messages=messages,
                    metadata=metadata
                )
                return result
            except Exception as e:
                if attempt < retries - 1:
                    print(f"    Retry {attempt + 1}/{retries} for user {user_id}: {e}")
                    time.sleep(1)
                    continue
                else:
                    print(f"    Failed to add memory for user {user_id} after {retries} attempts: {e}")
                    raise e
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            if hasattr(self, 'memory_system') and self.memory_system:
                stats = self.memory_system.get_stats()
                return stats.get('system', {})
            return {}
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def cleanup_resources(self):
        """Clean up system resources"""
        try:
            if hasattr(self, 'memory_system') and self.memory_system:
                print("Cleaning up memory system resources...")
                self.memory_system.__exit__(None, None, None)
                self.memory_system = None
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")


def main():
    """Test function"""
    # Example usage
    adder = LongMemEvalMemorySystemAdd(
        data_path="./dataset/longmemeval_s.json",
        batch_size=1,
        storage_path="./longmemeval_memories_v3",
        model=os.getenv("OPENAI_MODEL"),
        language="en",
        enable_semantic_memory=True,
        semantic_generation_workers=8,
        semantic_wait_timeout=120,
        max_workers=10
    )
    
    # Process first 5 questions as test
    adder.process_all_questions(start_idx=0, end_idx=5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add LongMemEval data to memory system v3 (Improved)")
    parser.add_argument("--data_path", type=str, default="./dataset/longmemeval_s.json",
                        help="Path to longmemeval_s.json")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for adding messages")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Maximum number of parallel workers")
    parser.add_argument("--semantic_generation_workers", type=int, default=20,
                        help="Number of workers for semantic memory generation")
    parser.add_argument("--semantic_wait_timeout", type=int, default=3600,
                        help="Timeout for semantic memory generation (seconds)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for processing (default: process all)")
    parser.add_argument("--disable_semantic", action="store_true",
                        help="Disable semantic memory generation")
    parser.add_argument("--storage_path", type=str, default="./longmemeval_memories",
                        help="Storage path for memories")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL"),
                        help="LLM model to use")
    parser.add_argument("--language", type=str, default="en",
                        help="Language setting (en or zh)")
    
    args = parser.parse_args()
    
    try:
        # Create adder
        adder = LongMemEvalMemorySystemAdd(
            data_path=args.data_path,
            batch_size=args.batch_size,
            storage_path=args.storage_path,
            model=args.model,
            language=args.language,
            enable_semantic_memory=not args.disable_semantic,
            semantic_generation_workers=args.semantic_generation_workers,
            semantic_wait_timeout=args.semantic_wait_timeout,
            max_workers=args.max_workers
        )
        
        # Process all questions
        adder.process_all_questions(
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        print("🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        try:
            if 'adder' in locals():
                adder.cleanup_resources()
        except:
            pass 