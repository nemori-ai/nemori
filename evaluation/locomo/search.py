"""
New Memory System v3.0 - Search Memory for Evaluation  
"""

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

# Import new memory system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.core.memory_system import MemorySystem
from src.config import MemoryConfig

# Import prompts - you may need to create this file

    # Fallback prompt if prompts.py doesn't exist
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation between two speakers. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from the conversation
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

Conversation memories:
{{conversation_memories}}

Question: {{question}}

Answer:
"""

load_dotenv()


class MemorySystemSearch:
    """
    New Memory System v3.0 - Search Memory for Evaluation
    
    Features:
    - Supports searching both episodic and semantic memories
    - Semantic memories now use unified "knowledge" type
    - Better timestamp handling (stored separately from content)
    - Flexible search with BM25 and vector similarity
    - Modified to search only one user's memories since both speakers have similar memories
    """
    
    def __init__(self, 
                 output_path="episodic_v3_results.json",
                 storage_path="./evaluation_memories_v3",
                 model="gpt-4.1-mini",
                 language="en",
                 top_k_episodes=5,
                 top_k_semantic=5,
                 include_original_messages_top_k=2,
                 max_workers=10,
                 save_batch_size=30,
                 search_method="bm25",
                 enable_memory_cleanup=False,
                 force_reload_indices=False):
        """
        Initialize new memory system for searching memories
        
        Args:
            output_path: Output path for results
            storage_path: Storage base path
            model: LLM model to use
            language: Language for tokenization ("en" or "zh")
            top_k_episodes: Number of episodic memories to retrieve
            top_k_semantic: Number of semantic memories to retrieve
            include_original_messages_top_k: Number of top episode results to include original messages
            max_workers: Max parallel workers
            save_batch_size: Batch size for saving
            enable_memory_cleanup: Enable memory cleanup for large datasets
            force_reload_indices: Force reload indices without using cache
        """
        # Create configuration
        self.config = MemoryConfig(
            # LLM and Embedding settings
            llm_model=model,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            
            # Storage settings
            storage_path=storage_path,
            
            # Search settings
            search_top_k_episodes=top_k_episodes,
            search_top_k_semantic=top_k_semantic,
            
            # Performance settings
            use_faiss=False,
            faiss_index_type="IVF",
            batch_size=32,
            max_workers=max_workers,
            # Cache settings
            enable_cache=True,
            cache_size=1000,
            cache_ttl_seconds=3600,
            
            # Language settings
            language=language
        )
        
        # Initialize memory system with optimized loading enabled
        self.memory_system = MemorySystem(self.config)
        
        # Search configuration
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.include_original_messages_top_k = include_original_messages_top_k
        self.search_method = search_method  # Default use vector search
        
        # OpenAI client for answer generation
        self.openai_client = OpenAI()
        
        # Results storage
        self.results = defaultdict(list)
        self.output_path = output_path
        self.ANSWER_PROMPT = ANSWER_PROMPT
        
        # Parallel processing config
        self.max_workers = max_workers
        self.save_batch_size = save_batch_size
        self.enable_memory_cleanup = enable_memory_cleanup
        
        # Cache for loaded users to avoid repeated loading
        self.loaded_users = set()
        
        # Force reload indices setting
        self.force_reload_indices = force_reload_indices
        
        # Print configuration
        print(f"\n=== New Memory System v3.0 Search Configuration ===")
        print(f"Model: {model}")
        print(f"Language: {language}")
        print(f"Episodic Memories: {top_k_episodes}")
        print(f"Semantic Memories: {top_k_semantic}")
        print(f"Include Original Messages Top K: {include_original_messages_top_k}")
        print(f"Max Workers: {max_workers}")
        print(f"Storage Path: {storage_path}")
        print(f"Search Mode: Single user (optimized for LoCoMo dataset)")
        print(f"Force Reload Indices: {force_reload_indices}")
        print("==================================================\n")

    def _clear_cache_completely(self):
        """
        如果启用了强制重新加载，清空整个缓存
        这是最安全的方式，避免操作内部缓存结构
        """
        if self.force_reload_indices:
            if hasattr(self.memory_system, 'performance_optimizer') and hasattr(self.memory_system.performance_optimizer, 'cache'):
                self.memory_system.performance_optimizer.cache.clear()
                print("Cache cleared for fresh data loading")

    def search_memory(self, user_id: str, query: str, max_retries: int = 5, 
                     retry_delay: int = 2) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search memories using new system API
        
        Args:
            user_id: User ID
            query: Query string
            max_retries: Max retry attempts (increased from 3 to 5)
            retry_delay: Retry delay in seconds (increased from 1 to 2)
            
        Returns:
            Tuple of (formatted_memories, search_time)
        """
        start_time = time.time()
        
        if self.top_k_episodes == 0 and self.top_k_semantic == 0:
            return [], time.time() - start_time
         
        retries = 0
        
        while retries < max_retries:
            try:
                episodes = self.memory_system.storage["episode"].get_user_episodes(user_id)
                semantic_memories = self.memory_system.storage["semantic"].list_user_items(user_id)
                
                if len(episodes) == 0 and len(semantic_memories) == 0:
                    return [], time.time() - start_time
                
                # Use search_all method which returns properly structured results
                formatted_search_results = self.memory_system.search_all(
                    user_id=user_id,
                    query=query,
                    top_k_episodes=self.top_k_episodes,
                    top_k_semantic=self.top_k_semantic,
                    search_method=self.search_method
                )
                
                if not formatted_search_results:    
                    raise Exception("Search returned None or empty results")
                
                episodic_count = len(formatted_search_results.get('episodic', []))
                semantic_count = len(formatted_search_results.get('semantic', []))
                
                if episodic_count == 0 and semantic_count == 0 and (len(episodes) > 0 or len(semantic_memories) > 0):
                    raise Exception(f"User {user_id} has data but search returned empty results")
                
                break
                
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    return [], time.time() - start_time
                
                time.sleep(retry_delay * (2 ** retries))

        end_time = time.time()
        search_time = end_time - start_time
        
        # Format results according to evaluation requirements
        formatted_memories = self._format_search_results(formatted_search_results)
        
        return formatted_memories, search_time

    def _format_search_results(self, search_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format search results for evaluation
        
        Args:
            search_results: Raw search results from memory system
            
        Returns:
            Formatted memory list
        """
        formatted_memories = []
        
        # Process episodic memories
        episodic_results = search_results.get("episodic", [])
        for idx, episode in enumerate(episodic_results):
            # Handle both new format and legacy format
            content = episode.get("content", "")
            if not content:
                content = episode.get("summary", "")  # Fallback to summary
            
            timestamp = episode.get("timestamp", "")
            if not timestamp:
                timestamp = episode.get("created_at", "")  # Fallback
                
            memory_data = {
                "memory": content,
                "timestamp": timestamp,
                "score": round(episode.get("score", episode.get("fused_score", 0)), 2),
                "episode_title": episode.get("title", episode.get("episode_title", "")),
                "episode_id": episode.get("episode_id", ""),
                "level": 1,  # Episodic memories are level 1
                "memory_type": "episodic",
                "search_method": episode.get("search_method", "unknown")
            }
            
            # Add original messages for top results
            if idx < self.include_original_messages_top_k:
                original_messages = episode.get("original_messages", [])
                if original_messages:
                    memory_data["original_messages"] = original_messages
                    memory_data["has_original_messages"] = True
                else:
                    memory_data["original_messages"] = []
                    memory_data["has_original_messages"] = False
            else:
                memory_data["has_original_messages"] = False
            
            formatted_memories.append(memory_data)
        
        # Process semantic memories
        semantic_results = search_results.get("semantic", [])
        for semantic in semantic_results:
            # Handle both new format and legacy format
            content = semantic.get("content", "")
            timestamp = semantic.get("created_at", semantic.get("timestamp", ""))
            
            memory_data = {
                "memory": content,
                "timestamp": timestamp,
                "score": round(semantic.get("score", semantic.get("fused_score", 0)), 2),
                "episode_title": "",  # Semantic memories don't have episode titles
                "episode_id": semantic.get("memory_id", semantic.get("semantic_id", "")),
                "level": 2,  # Semantic memories are level 2
                "memory_type": "semantic",
                "knowledge_type": semantic.get("knowledge_type", "knowledge"),  # Default to "knowledge" for new format
                "confidence": semantic.get("confidence", 0.0),
                "search_method": semantic.get("search_method", "unknown"),
                "has_original_messages": False,  # Semantic memories don't have original messages
                "related_episodes": semantic.get("related_episodes", semantic.get("source_episodes", []))
            }
            
            formatted_memories.append(memory_data)
        
        return formatted_memories

    def answer_question(self, speaker_1_user_id: str, speaker_2_user_id: str, 
                       question: str, answer: str, category: int) -> Tuple[str, List, List, float, float, List, List, float]:
        """
        Answer question based on memories (modified to search only one user)
        
        Args:
            speaker_1_user_id: Speaker 1 user ID (will be used for search)
            speaker_2_user_id: Speaker 2 user ID (not used, kept for compatibility)
            question: Question text
            answer: Standard answer
            category: Question category
            
        Returns:
            Tuple of (response, speaker_1_memories, speaker_2_memories, 
                     speaker_1_memory_time, speaker_2_memory_time, 
                     speaker_1_graph_memories, speaker_2_graph_memories, response_time)
        """
        # Search memories for only the first speaker (since both have similar memories)
        conversation_memories, memory_search_time = self.search_memory(speaker_1_user_id, question)

        # Format memories for prompt
        conversation_memory_strings = self._format_memories_for_prompt(conversation_memories)

        # Generate answer using LLM
        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            conversation_memories=conversation_memory_strings,
            question=question,
        )

        # Call LLM to generate answer
        t1 = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=0.0
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            # print(f"Error generating answer: {e}")
            llm_response = "Error generating answer"
        
        t2 = time.time()
        response_time = t2 - t1
        
        return (
            llm_response,
            conversation_memories,      # speaker_1_memories (contains all conversation memories)
            [],                        # speaker_2_memories (empty since we only search one user)
            memory_search_time,        # speaker_1_memory_time
            0.0,                       # speaker_2_memory_time (0 since no search)
            [],                        # speaker_1_graph_memories - not used in this system
            [],                        # speaker_2_graph_memories - not used in this system
            response_time,
        )

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories for inclusion in prompt
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Single formatted string with episodic and semantic memories
        """
        # Separate episodic and semantic memories
        episodic_memories = [m for m in memories if m.get('memory_type') == 'episodic']
        semantic_memories = [m for m in memories if m.get('memory_type') == 'semantic']
        
        # 🚨 Critical fix: If no memories exist, return explicit "no memories" message
        if not episodic_memories and not semantic_memories:
            return "No memories available."
        
        # Start with introduction
        formatted_output = """The following are episodic and semantic memories related to the question. Episodic memories generally contain accurate original information, especially temporal information. Semantic memories contain conceptual information from which you can infer user preferences and other facts:

"""
        
        # Format episodic memories
        if episodic_memories:
            formatted_output += "Episodic Memories:\n"
            for memory in episodic_memories:
                timestamp = memory.get('timestamp', '')
                episode_title = memory.get('episode_title', '')
                content = memory['memory']
                
                # Format: timestamp, title
                title_line = f"{timestamp}"
                if episode_title:
                    title_line += f", {episode_title}"
                formatted_output += f"{title_line}\n"
                
                # Content
                formatted_output += f"{content}\n"
                
                # Add original messages if available
                if memory.get('has_original_messages', False) and memory.get('original_messages'):
                    original_messages = memory['original_messages']
                    formatted_output += "Original Messages:\n"
                    for j, msg in enumerate(original_messages):
                        role = msg.get('role', 'unknown')
                        msg_content = msg.get('content', '')
                        msg_timestamp = msg.get('timestamp', '')
                        formatted_output += f"  {j+1}. [{msg_timestamp}] {role}: {msg_content}\n"
                
                formatted_output += "\n"
        
        # Format semantic memories
        if semantic_memories:
            formatted_output += "Semantic Memories:\n"
            for memory in semantic_memories:
                timestamp = memory.get('timestamp', '')
                content = memory['memory']
                
                # Format: timestamp: content
                if timestamp:
                    formatted_output += f"{timestamp}: {content}\n"
                else:
                    formatted_output += f"{content}\n"
            
            formatted_output += "\n"
        
        return formatted_output.strip()

    def process_data_file(self, file_path: str):
        """
        Process data file with parallel processing and periodic saving
        
        Args:
            file_path: Path to evaluation data file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect all tasks and unique users (only speaker_a since we search one user)
        all_tasks = []
        all_users = set()
        for idx, item in enumerate(data):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            
            # Only collect speaker_a since we'll only search from one user
            all_users.add(speaker_a_user_id)

            # Create task for each question
            for question_item in qa:
                all_tasks.append((idx, question_item, speaker_a_user_id, speaker_b_user_id))

        # Pre-load all user data and indices to avoid concurrency issues
        force_reload_msg = " (force reload enabled)" if self.force_reload_indices else ""
        print(f"Pre-loading data for {len(all_users)} users (search method: {self.search_method}){force_reload_msg}...")
        
        # 如果启用强制重新加载，清除整个缓存
        if self.force_reload_indices:
            self._clear_cache_completely()
        
        for user_id in tqdm(all_users, desc="Pre-loading user data"):
            try:
                self.memory_system.load_user_data_and_indices_for_method(user_id, self.search_method)
            except Exception as e:
                print(f"Error pre-loading user {user_id}: {e}")

        # print(f"Total tasks to process: {len(all_tasks)}")
        # print(f"Configuration: max_workers={self.max_workers}, save_batch_size={self.save_batch_size}")

        # Process in batches
        batch_size = self.save_batch_size
        processed_count = 0
        
        if self.enable_memory_cleanup:
            temp_results = defaultdict(list)
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(all_tasks), desc="Processing all questions") as pbar:
                # Process in batches
                for i in range(0, len(all_tasks), batch_size):
                    batch_tasks = all_tasks[i:i + batch_size]
                    
                    # Submit current batch
                    futures = []
                    for task in batch_tasks:
                        idx, question_item, speaker_a_user_id, speaker_b_user_id = task
                        future = executor.submit(
                            self._process_single_task,
                            idx, question_item, speaker_a_user_id, speaker_b_user_id
                        )
                        futures.append((future, idx))
                    
                    # Collect results
                    batch_results = []
                    for future, idx in futures:
                        try:
                            result = future.result(timeout=120)  # 2 minute timeout
                            if self.enable_memory_cleanup:
                                temp_results[idx].append(result)
                            else:
                                self.results[idx].append(result)
                            batch_results.append((idx, result))
                            processed_count += 1
                            pbar.update(1)
                        except Exception as e:
                            # print(f"\nError processing task: {e}")
                            pbar.update(1)
                            continue
                    
                    # Save progress
                    # print(f"\nSaving progress... ({processed_count}/{len(all_tasks)} completed)")
                    
                    if self.enable_memory_cleanup:
                        self._incremental_save(batch_results)
                        temp_results.clear()
                    else:
                        with open(self.output_path, "w", encoding="utf-8") as f:
                            json.dump(self.results, f, indent=4, ensure_ascii=False)

        # Final save
        if not self.enable_memory_cleanup:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
            
        # print(f"\nResults saved to {self.output_path}")
        # self._print_statistics()
        
        self._cleanup_resources()
        
    def _process_single_task(self, idx: int, question_item: Dict[str, Any], 
                           speaker_a_user_id: str, speaker_b_user_id: str) -> Dict[str, Any]:
        """
        Process single task for parallel execution
        
        Args:
            idx: Conversation index
            question_item: Question data
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            
        Returns:
            Processing result
        """
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        category = question_item.get("category", -1)
        evidence = question_item.get("evidence", [])
        adversarial_answer = question_item.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            "system_version": "v3.0-single-user"
        }

        return result

    def _incremental_save(self, batch_results: List[Tuple[int, Dict[str, Any]]]):
        """
        Incremental save for memory cleanup mode
        
        Args:
            batch_results: List of (idx, result) tuples
        """
        # Load existing results
        existing_results = {}
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            except:
                existing_results = {}
        
        # Merge new results
        for idx, result in batch_results:
            idx_str = str(idx)
            if idx_str not in existing_results:
                existing_results[idx_str] = []
            existing_results[idx_str].append(result)
        
        # Save updated results
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4, ensure_ascii=False)

    def _print_statistics(self):
        """Print processing statistics"""
        try:
            total_conversations = len(self.results)
            total_questions = sum(len(questions) for questions in self.results.values())
            
            # print(f"\n=== Processing Statistics ===")
            # print(f"Total conversations: {total_conversations}")
            # print(f"Total questions: {total_questions}")
            # print(f"Output file: {self.output_path}")
            # print("============================\n")
        except Exception as e:
            # print(f"Error printing statistics: {e}")
            pass

    def _cleanup_resources(self):
        """Clean up resources to ensure proper program termination"""
        try:
            if hasattr(self, 'memory_system'):
                if hasattr(self.memory_system, 'search_engine'):
                    if hasattr(self.memory_system.search_engine, 'executor'):
                        self.memory_system.search_engine.executor.shutdown(wait=True)
                
                if hasattr(self.memory_system, 'performance_optimizer'):
                    self.memory_system.performance_optimizer.cache.clear()
            
            if hasattr(self, 'openai_client'):
                if hasattr(self.openai_client, 'close'):
                    self.openai_client.close()
            
            print("Resources cleaned up successfully.")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
        import sys
        sys.exit(0)


def main():
    """Main function for testing"""
    import sys
    import os
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    try:
        # Example usage
        searcher = MemorySystemSearch(
            output_path="locomo/results.json",
            storage_path="memories",  # Use correct data directory
            model="gpt-4.1-mini",
            language="en",
            top_k_episodes=20,
            top_k_semantic=20,
            include_original_messages_top_k=2,
            max_workers=100,  # Further reduced from 20 to 6 to avoid concurrency issues
            save_batch_size=200,  # Reduced from 100 to 30 for better stability
            enable_memory_cleanup=False,
            search_method="vector",
            force_reload_indices=True  # 强制重新加载索引，不使用缓存
        )
        
        # Process evaluation data
        searcher.process_data_file("./dataset/locomo10.json")  # Update with actual path
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Program finished.")
        sys.exit(0)


if __name__ == "__main__":
    main() 