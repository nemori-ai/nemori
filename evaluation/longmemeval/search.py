import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from jinja2 import Template
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.core.memory_system import MemorySystem
from src.core.lightweight_search_system import LightweightSearchSystem
from src.config import MemoryConfig

load_dotenv()


class LongMemEvalMemorySystemSearch:
    def __init__(self, data_path, output_path="longmemeval/results.json",
                 storage_path="./longmemeval_memories",
                 model="gpt-4o-mini",
                 language="en",
                 top_k_episodes=10,
                 top_k_semantic=10,
                 include_original_messages_top_k=2,
                 max_workers=10,
                 save_batch_size=100,
                 search_method="vector",
                 use_preload=True):
        """
        Initialize LongMemEval New Memory System Search
        
        Args:
            data_path: Path to data file
            output_path: Result output path
            storage_path: Storage path
            model: LLM model
            language: Language setting
            top_k_episodes: Number of episodic memories to return
            top_k_semantic: Number of semantic memories to return
            include_original_messages_top_k: Include original message information for top K episodes
            max_workers: Maximum number of parallel processing worker threads
            save_batch_size: How many questions to process before saving once
            search_method: Search method ("vector", "bm25", "hybrid")
            use_preload: Whether to use preload optimization (only supports vector search)
        """
        self.data_path = data_path
        self.output_path = output_path
        self.storage_path = storage_path
        self.model = model
        self.language = language
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.include_original_messages_top_k = include_original_messages_top_k
        self.max_workers = max_workers
        self.save_batch_size = save_batch_size
        self.search_method = search_method
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Result storage
        self.results = defaultdict(list)
        
        print(f"\n=== LongMemEval Memory System v3 Search Configuration ===")
        print(f"Total questions: {len(self.data)}")
        print(f"Model: {model}")
        print(f"Language: {language}")
        print(f"Episodic Memories: {top_k_episodes}")
        print(f"Semantic Memories: {top_k_semantic}")
        print(f"Include Original Messages Top K: {include_original_messages_top_k}")
        print(f"Search Method: {search_method}")
        print(f"Max Workers: {max_workers}")
        print(f"Storage Path: {storage_path}")
        
        # Validate storage path
        if os.path.exists(storage_path):
            episodes_dir = os.path.join(storage_path, "episodes")
            semantic_dir = os.path.join(storage_path, "semantic")
            print(f"\n[VALIDATION] Storage path exists: {storage_path}")
            print(f"[VALIDATION] Episodes directory exists: {os.path.exists(episodes_dir)}")
            print(f"[VALIDATION] Semantic directory exists: {os.path.exists(semantic_dir)}")
            
            if os.path.exists(episodes_dir):
                # Check number of JSONL files instead of directories
                episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('_episodes.jsonl')]
                print(f"[VALIDATION] Found {len(episode_files)} episode files")
            
            if os.path.exists(semantic_dir):
                # Check number of JSONL files instead of directories
                semantic_files = [f for f in os.listdir(semantic_dir) if f.endswith('_semantic.jsonl')]
                print(f"[VALIDATION] Found {len(semantic_files)} semantic files")
        else:
            print(f"\n[WARNING] Storage path does not exist: {storage_path}")
            print(f"[WARNING] Please ensure the memory data has been generated using add_longmemeval_v3.py")
        
        print("========================================================\n")
        
        # No longer create a shared memory system, but save configuration for independent use by each question
        # This avoids lock contention and enables true parallel loading
        self.preload_all_data = use_preload and search_method == "vector"  # Only vector search supports preload
        self.preloaded_data = {}  # Store preloaded data
        
        # Create lightweight search system (for fast search)
        self.lightweight_search = None
        if self.preload_all_data:
            print(f"\n[INFO] Preload optimization enabled for vector search")
            self._initialize_lightweight_search()
        elif use_preload and search_method != "vector":
            print(f"\n[WARN] Preload optimization only supports vector search, disabled for {search_method}")
        
        # Answer generation prompt
        self.ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from conversations. These memories contain timestamped information that may be relevant to answering the question.

# MEMORIES:

## Episodic Memories (Conversation Episodes):
{{episodic_memories}}

## Semantic Memories (Knowledge Statements):
{{semantic_memories}}

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both types
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. The answer should be concise and directly address the question
8. The question date refers to the time when this question was asked, which can help you make judgments about time-related questions
9. Some episodic memories may include "Original Conversation Messages" which contain the exact dialogue. Use these when you need specific quotes or exact wording
10. Semantic memories provide extracted knowledge that may help answer general questions about facts, preferences, or patterns

# QUESTION_DATE:
{{question_date}}

# QUESTION:
The User: {{question}}

# ANSWER:"""
        

    
    def _initialize_lightweight_search(self):
        """Initialize lightweight search system and preload all user data"""
        print("\n[INFO] Initializing lightweight search system...")
        start_time = time.time()
        
        # Create configuration
        config = MemoryConfig(
            llm_model=self.model,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            storage_path=self.storage_path,
            language=self.language,
            search_top_k_episodes=self.top_k_episodes,
            search_top_k_semantic=self.top_k_semantic,
            use_faiss=True,
            enable_cache=True
        )
        
        # Create lightweight search system
        self.lightweight_search = LightweightSearchSystem(config, preload_data=True)
        
        # Collect all unique user IDs
        unique_user_ids = set()
        for item in self.data:
            user_id = f"question_{item['question_id']}"
            unique_user_ids.add(user_id)
        
        print(f"[INFO] Found {len(unique_user_ids)} unique users to preload")
        
        # Batch preload all user data
        print("[INFO] Preloading all user data into memory...")
        # Use reasonable parallelism: for preload, I/O-intensive tasks can use more threads
        max_preload_workers = min(len(unique_user_ids), self.max_workers * 2)
        preload_results = self.lightweight_search.batch_preload_users(
            list(unique_user_ids), 
            max_workers=max_preload_workers
        )
        
        # Statistics preload results
        successful = sum(1 for v in preload_results.values() if v)
        failed = len(preload_results) - successful
        
        total_time = time.time() - start_time
        print(f"[INFO] Preload completed in {total_time:.2f}s")
        print(f"[INFO] Successfully preloaded: {successful} users")
        print(f"[INFO] Failed to preload: {failed} users")
        
        # Show memory usage
        memory_usage = self.lightweight_search.get_memory_usage()
        print(f"[INFO] Memory usage: {memory_usage}")
        print("======================================\n")
    
    
    def search_memory_for_question(self, question_data: Dict[str, Any]) -> Tuple[Dict[str, List[Dict]], float, str]:
        """
        Search memory for a single question
        
        Args:
            question_data: Question data
            
        Returns:
            (search results, search time, transformed question)
        """
        question_id = question_data['question_id']
        original_question = question_data['question']
        
        print(f"\n[DEBUG] Starting search for question: {question_id}")
        print(f"[DEBUG] Original question: {original_question[:100]}...")
        
        # Use original question directly, without transformation
        question = original_question
        print(f"[DEBUG] Using original question directly (no transformation)")
        
        # Use question ID as user ID
        user_id = f"question_{question_id}"
        print(f"[DEBUG] User ID: {user_id}")
        
        # Search memory
        start_time = time.time()
        
        try:
            if self.preload_all_data and self.lightweight_search:
                # Use lightweight search system (preloaded data, no lock, fast)
                print(f"[DEBUG] Using lightweight search system (preloaded data)")
                
                if self.search_method == "vector":
                    # Execute vector search
                    search_start = time.time()
                    search_results = self.lightweight_search.search_vector(
                        user_id=user_id,
                        query=question,
                        top_k_episodes=self.top_k_episodes,
                        top_k_semantic=self.top_k_semantic
                    )
                    search_time = time.time() - search_start
                    print(f"[DEBUG] Vector search took: {search_time:.3f}s")
                else:
                    # For non-vector search, fall back to standard system
                    print(f"[WARN] Lightweight search only supports vector search, falling back to standard system")
                    return self._search_with_standard_system(question_id, question, user_id)
            else:
                # Use standard memory system
                return self._search_with_standard_system(question_id, question, user_id)
            
            # Print search result statistics
            episodic_count = len(search_results.get("episodic", []))
            semantic_count = len(search_results.get("semantic", []))
            print(f"[DEBUG] Found {episodic_count} episodic memories, {semantic_count} semantic memories")
            
            # Format search results
            format_start = time.time()
            formatted_results = self._format_search_results(search_results)
            format_time = time.time() - format_start
            print(f"[DEBUG] Result formatting took: {format_time:.3f}s")
            
            total_search_time = time.time() - start_time
            print(f"[DEBUG] Total search time: {total_search_time:.3f}s")
            
            return formatted_results, total_search_time, question
            
        except Exception as e:
            print(f"[ERROR] Error searching memory for question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"episodic": [], "semantic": []}, 0, original_question
    
    def _search_with_standard_system(self, question_id: str, question: str, user_id: str) -> Tuple[Dict[str, List[Dict]], float, str]:
        """Use standard memory system for search (compatible with all search methods)"""
        start_time = time.time()
        
        # Create independent configuration and memory system for each question
        config = MemoryConfig(
            llm_model=self.model,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            storage_path=self.storage_path,
            language=self.language,
            search_top_k_episodes=self.top_k_episodes,
            search_top_k_semantic=self.top_k_semantic,
            use_faiss=True,
            enable_cache=True
        )
        
        # Create independent memory system instance
        system_start = time.time()
        memory_system = MemorySystem(config)
        system_time = time.time() - system_start
        print(f"[DEBUG] Memory system creation took: {system_time:.3f}s")
        
        try:
            # Load user data and indices
            load_start = time.time()
            print(f"[DEBUG] Loading user data and indices for method: {self.search_method}")
            memory_system.load_user_data_and_indices_for_method(user_id, self.search_method)
            load_time = time.time() - load_start
            print(f"[DEBUG] Data loading took: {load_time:.3f}s")
            
            # Execute search
            search_start = time.time()
            print(f"[DEBUG] Executing search with top_k_episodes={self.top_k_episodes}, top_k_semantic={self.top_k_semantic}")
            search_results = memory_system.search_all(
                user_id=user_id,
                query=question,
                top_k_episodes=self.top_k_episodes,
                top_k_semantic=self.top_k_semantic,
                search_method=self.search_method
            )
            search_time = time.time() - search_start
            print(f"[DEBUG] Search execution took: {search_time:.3f}s")
            
            # Format search results
            format_start = time.time()
            formatted_results = self._format_search_results(search_results)
            format_time = time.time() - format_start
            print(f"[DEBUG] Result formatting took: {format_time:.3f}s")
            
            total_search_time = time.time() - start_time
            
            # Clean up memory system resources
            cleanup_start = time.time()
            memory_system.__exit__(None, None, None)
            cleanup_time = time.time() - cleanup_start
            print(f"[DEBUG] Cleanup took: {cleanup_time:.3f}s")
            
            return formatted_results, total_search_time, question
            
        except Exception as e:
            print(f"[ERROR] Error in standard system search: {e}")
            import traceback
            traceback.print_exc()
            # Ensure cleanup resources
            try:
                memory_system.__exit__(None, None, None)
            except:
                pass
            return {"episodic": [], "semantic": []}, time.time() - start_time, question
    
    def _format_search_results(self, search_results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Format search results
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            Formatted results {"episodic": [...], "semantic": [...]}
        """
        formatted = {"episodic": [], "semantic": []}
        
        # Process episodic memories
        episodic_results = search_results.get("episodic", [])
        for idx, episode in enumerate(episodic_results):
            memory_data = {
                "content": episode.get("content", ""),
                "timestamp": episode.get("timestamp", ""),
                "score": round(episode.get("score", 0), 3),
                "episode_title": episode.get("title", ""),
                "episode_id": episode.get("episode_id", ""),
                "level": 1,
                "search_method": episode.get("search_method", "unknown")
            }
            
            # Add original messages (if within top_k range)
            if idx < self.include_original_messages_top_k:
                original_messages = episode.get("original_messages", [])
                if original_messages:
                    memory_data["original_messages"] = original_messages
                    memory_data["has_original_messages"] = True
                else:
                    memory_data["has_original_messages"] = False
            else:
                memory_data["has_original_messages"] = False
            
            formatted["episodic"].append(memory_data)
        
        # Process semantic memories
        semantic_results = search_results.get("semantic", [])
        for semantic in semantic_results:
            memory_data = {
                "content": semantic.get("content", ""),
                "timestamp": semantic.get("created_at", ""),  # Semantic memories use created_at as timestamp
                "score": round(semantic.get("score", 0), 3),
                "memory_id": semantic.get("memory_id", ""),
                "confidence": semantic.get("confidence", 0.0),
                "knowledge_type": semantic.get("knowledge_type", "knowledge"),
                "source_episodes": semantic.get("source_episodes", []),
                "level": 2,
                "search_method": semantic.get("search_method", "unknown")
            }
            
            formatted["semantic"].append(memory_data)
        
        return formatted
    
    def answer_question(self, question_data: Dict[str, Any], search_results: Dict[str, List[Dict]]) -> Tuple[str, float]:
        """
        Answer question based on memories
        
        Args:
            question_data: Question data
            search_results: Search results
            
        Returns:
            (generated answer, response time)
        """
        # Format episodic memories
        episodic_texts = []
        for i, mem in enumerate(search_results.get("episodic", [])):
            memory_text = f"{i+1}. [{mem['timestamp']}] {mem['content']}"
            if mem.get('episode_title'):
                memory_text = f"   Title: {mem['episode_title']}\n   " + memory_text
            
            # Add original messages
            if mem.get('has_original_messages') and mem.get('original_messages'):
                memory_text += "\n   Original Messages:"
                for j, msg in enumerate(mem['original_messages']):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')
                    memory_text += f"\n     - [{timestamp}] {role}: {content}"
            episodic_texts.append(memory_text)
        
        episodic_str = "\n\n".join(episodic_texts) if episodic_texts else "No episodic memories found."
        
        # Format semantic memories
        semantic_texts = []
        for i, mem in enumerate(search_results.get("semantic", [])):
            memory_text = f"{i+1}. [{mem['timestamp']}]  {mem['content']}"
            semantic_texts.append(memory_text)
        
        semantic_str = "\n\n".join(semantic_texts) if semantic_texts else "No semantic memories found."
        
        # Build question with date (same as zep original implementation)
        question_with_date = f"(date: {question_data['question_date']}) {question_data['question']}"
        
        # Build prompt
        template = Template(self.ANSWER_PROMPT)
        prompt = template.render(
            episodic_memories=episodic_str,
            semantic_memories=semantic_str,
            question=question_with_date,
            question_date=question_data['question_date']
        )
        
        # Call LLM to generate answer
        start_time = time.time()
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            response_time = time.time() - start_time
            
            return answer, response_time
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating answer", 0
    
    def process_question(self, question_data: Dict[str, Any], question_idx: int) -> Dict[str, Any]:
        """
        Process a single question
        
        Args:
            question_data: Question data
            question_idx: Question index
            
        Returns:
            Process result
        """
        question_id = question_data['question_id']
        print(f"\n[INFO] Processing question {question_idx}: {question_id}")
        
        # Search memory
        print(f"[INFO] Starting memory search...")
        search_results, search_time, question_used = self.search_memory_for_question(question_data)
        print(f"[INFO] Memory search completed in {search_time:.3f}s")
        
        # Generate answer
        print(f"[INFO] Generating answer...")
        response, response_time = self.answer_question(question_data, search_results)
        print(f"[INFO] Answer generated in {response_time:.3f}s")
        
        # Build result
        result = {
            "question_id": question_id,
            "question": question_data['question'],
            "answer": question_data['answer'],
            "question_type": question_data['question_type'],
            "question_date": question_data['question_date'],
            "response": response,
            "episodic_memories": search_results.get("episodic", []),
            "semantic_memories": search_results.get("semantic", []),
            "num_episodic_memories": len(search_results.get("episodic", [])),
            "num_semantic_memories": len(search_results.get("semantic", [])),
            "search_time": search_time,
            "response_time": response_time,
            "total_time": search_time + response_time,
            "answer_session_ids": question_data.get('answer_session_ids', []),
            "system_version": "v3.0"
        }
        
        return result
    
    def process_all_questions(self, start_idx=0, end_idx=None):
        """
        Process all questions
        
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
        
        # Use thread pool to process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for idx, question_data in enumerate(questions_to_process):
                actual_idx = start_idx + idx
                future = executor.submit(self.process_question, question_data, actual_idx)
                futures.append((future, actual_idx))
            
            # Collect results and save periodically
            for i, (future, idx) in enumerate(tqdm(futures, desc="Processing questions")):
                try:
                    result = future.result(timeout=300)  # Increase to 5 minutes timeout
                    results.append(result)
                    
                    # Save results periodically
                    if (i + 1) % self.save_batch_size == 0:
                        self._save_results(results)
                        print(f"\nSaved progress: {i + 1}/{len(futures)} questions processed")
                        
                except Exception as e:
                    print(f"\nError processing question {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    results.append({
                        "question_id": self.data[idx]['question_id'],
                        "question": self.data[idx]['question'],
                        "answer": self.data[idx]['answer'],
                        "response": f"Error: {str(e)}",
                        "error": True
                    })
        
        # Save final results
        self._save_results(results)
        
        # Print statistics
        self._print_statistics(results)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to file"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {self.output_path}")
    
    def _print_statistics(self, results: List[Dict[str, Any]]):
        """Print statistics"""
        total = len(results)
        errors = sum(1 for r in results if r.get('error', False))
        successful = total - errors
        
        # Calculate average metrics
        avg_search_time = sum(r.get('search_time', 0) for r in results) / total if total > 0 else 0
        avg_response_time = sum(r.get('response_time', 0) for r in results) / total if total > 0 else 0
        avg_total_time = sum(r.get('total_time', 0) for r in results) / total if total > 0 else 0
        
        avg_episodic = sum(r.get('num_episodic_memories', 0) for r in results) / total if total > 0 else 0
        avg_semantic = sum(r.get('num_semantic_memories', 0) for r in results) / total if total > 0 else 0
        
        print(f"\n=== Processing Statistics ===")
        print(f"Total questions: {total}")
        print(f"Successful: {successful}")
        print(f"Errors: {errors}")
        print(f"Average search time: {avg_search_time:.3f}s")
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Average total time: {avg_total_time:.3f}s")
        print(f"Average episodic memories retrieved: {avg_episodic:.1f}")
        print(f"Average semantic memories retrieved: {avg_semantic:.1f}")
        print("=============================\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search and evaluate LongMemEval with memory system v3")
    parser.add_argument("--data_path", type=str, default="./dataset/longmemeval_s.json",
                        help="Path to longmemeval_s.json")
    parser.add_argument("--output_path", type=str, default="longmemeval/results.json",
                        help="Path to save results")
    parser.add_argument("--storage_path", type=str, default="./longmemeval_memories",
                        help="Path to memory storage")
    parser.add_argument("--top_k_episodes", type=int, default=10,
                        help="Number of episodic memories to retrieve")
    parser.add_argument("--top_k_semantic", type=int, default=20,
                        help="Number of semantic memories to retrieve")
    parser.add_argument("--include_original_top_k", type=int, default=2,
                        help="Number of top episodes to include original messages")
    parser.add_argument("--max_workers", type=int, default=40,
                        help="Maximum number of parallel workers")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for processing")
    parser.add_argument("--search_method", type=str, default="vector",
                        choices=["vector", "bm25", "hybrid"],
                        help="Search method to use")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="LLM model to use")
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh"],
                        help="Language setting")
    parser.add_argument("--save_batch_size", type=int, default=100,
                        help="Save results every N questions")
    parser.add_argument("--use_preload", type=bool, default=True,
                        help="Whether to preload all data into memory for faster search")
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = LongMemEvalMemorySystemSearch(
        data_path=args.data_path,
        output_path=args.output_path,
        storage_path=args.storage_path,
        model=args.model,
        language=args.language,
        top_k_episodes=args.top_k_episodes,
        top_k_semantic=args.top_k_semantic,
        include_original_messages_top_k=args.include_original_top_k,
        max_workers=args.max_workers,
        save_batch_size=args.save_batch_size,
        search_method=args.search_method,
        use_preload=args.use_preload
    )
    
    # Process all questions
    searcher.process_all_questions(
        start_idx=args.start_idx,
        end_idx=args.end_idx
    ) 