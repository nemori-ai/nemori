"""
Performance Optimization Tools
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry"""
    value: Any
    timestamp: float
    ttl: float
    
    def is_expired(self) -> bool:
        """Check if expired"""
        return time.time() - self.timestamp > self.ttl


class OptimizedLRUCache:
    """Sharded LRU cache - uses multiple locks to reduce contention"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600, num_shards: int = 16):
        """
        Initialize optimized LRU cache
        
        Args:
            max_size: Maximum cache size
            default_ttl: Default expiration time (seconds)
            num_shards: Number of shards (reduce lock contention)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.num_shards = num_shards
        
        # Shard storage and locks
        self.shards: List[Dict[str, CacheEntry]] = [{} for _ in range(num_shards)]
        self.access_orders: List[List[str]] = [[] for _ in range(num_shards)]
        self.shard_locks: List[threading.RLock] = [threading.RLock() for _ in range(num_shards)]
        
        # Maximum size per shard
        self.shard_max_size = max(1, max_size // num_shards)
        
        logger.debug(f"Optimized LRU cache initialized with {num_shards} shards, {self.shard_max_size} per shard")
    
    def _get_shard_index(self, key: str) -> int:
        """Get shard index based on key"""
        return hash(key) % self.num_shards
    
    def _move_to_end(self, shard_idx: int, key: str):
        """Move key to end of shard"""
        access_order = self.access_orders[shard_idx]
        if key in access_order:
            access_order.remove(key)
        access_order.append(key)
    
    def _evict_expired(self, shard_idx: int):
        """Clean up expired entries in shard"""
        shard = self.shards[shard_idx]
        access_order = self.access_orders[shard_idx]
        
        expired_keys = []
        for key, entry in shard.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            shard.pop(key, None)
            if key in access_order:
                access_order.remove(key)
    
    def _evict_lru(self, shard_idx: int):
        """Clean up least recently used entries in shard"""
        shard = self.shards[shard_idx]
        access_order = self.access_orders[shard_idx]
        
        while len(shard) >= self.shard_max_size:
            if access_order:
                lru_key = access_order.pop(0)
                shard.pop(lru_key, None)
            else:
                break
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value (shard locking)"""
        shard_idx = self._get_shard_index(key)
        
        with self.shard_locks[shard_idx]: 
            shard = self.shards[shard_idx]
            
            if key in shard:
                entry = shard[key]
                if not entry.is_expired():
                    self._move_to_end(shard_idx, key)
                    return entry.value
                else:

                    shard.pop(key, None)
                    access_order = self.access_orders[shard_idx]
                    if key in access_order:
                        access_order.remove(key)
            
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store cache value (shard locking)"""
        shard_idx = self._get_shard_index(key)
        
        with self.shard_locks[shard_idx]:  # Only lock relevant shard
            # Clean up expired entries
            self._evict_expired(shard_idx)
            
            # If shard is full, clean up LRU entries
            if len(self.shards[shard_idx]) >= self.shard_max_size:
                self._evict_lru(shard_idx)
            
            # Store new entry
            ttl = ttl or self.default_ttl
            self.shards[shard_idx][key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            self._move_to_end(shard_idx, key)
    
    def contains(self, key: str) -> bool:
        """Check if key is contained (shard locking)"""
        shard_idx = self._get_shard_index(key)
        
        with self.shard_locks[shard_idx]:  # Only lock relevant shard
            shard = self.shards[shard_idx]
            
            if key in shard:
                entry = shard[key]
                if not entry.is_expired():
                    return True
                else:
                    # Delete expired entry
                    shard.pop(key, None)
                    access_order = self.access_orders[shard_idx]
                    if key in access_order:
                        access_order.remove(key)
            
            return False
    
    def clear(self):
        """Clear all shard caches"""
        for i in range(self.num_shards):
            with self.shard_locks[i]:
                self.shards[i].clear()
                self.access_orders[i].clear()
    
    def size(self) -> int:
        """Get total cache size"""
        total_size = 0
        for i in range(self.num_shards):
            with self.shard_locks[i]:
                total_size += len(self.shards[i])
        return total_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = 0
        shard_sizes = []
        
        for i in range(self.num_shards):
            with self.shard_locks[i]:
                shard_size = len(self.shards[i])
                total_size += shard_size
                shard_sizes.append(shard_size)
        
        return {
            "size": total_size,
            "max_size": self.max_size,
            "usage_ratio": total_size / self.max_size,
            "default_ttl": self.default_ttl,
            "num_shards": self.num_shards,
            "shard_sizes": shard_sizes,
            "avg_shard_size": total_size / self.num_shards,
            "cache_type": "sharded_lru"
        }


# Keep original class name for backward compatibility
LRUCache = OptimizedLRUCache


class PerformanceOptimizer:
    """Optimized performance optimizer - supports concurrent processing"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: float = 3600, 
                 max_workers: int = 4, num_cache_shards: int = 16):
        """
        Initialize performance optimizer
        
        Args:
            cache_size: Cache size
            cache_ttl: Cache expiration time
            max_workers: Maximum number of worker threads
            num_cache_shards: Number of cache shards
        """
        self.cache = OptimizedLRUCache(
            max_size=cache_size, 
            default_ttl=cache_ttl,
            num_shards=num_cache_shards
        )
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Statistics (protected by fine-grained locks)
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_tasks": 0,
            "total_time_saved": 0.0
        }
        self.stats_lock = threading.Lock()  # Lightweight lock for statistics
        
        logger.info(f"Performance optimizer initialized with {num_cache_shards} cache shards")
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key (optimized version)"""
        try:
            # Simplified version: direct string concatenation, avoid JSON serialization overhead
            key_parts = [str(func_name)]
            
            # Process args
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(str(hash(str(arg))))
            
            # Process kwargs
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                for k, v in sorted_kwargs:
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}:{v}")
                    else:
                        key_parts.append(f"{k}:{hash(str(v))}")
            
            # Generate final key
            key_str = "|".join(key_parts)
            return hashlib.md5(key_str.encode()).hexdigest()
            
        except Exception as e:
            # Fallback to simpler key generation method
            logger.warning(f"Cache key generation failed: {e}, using fallback")
            return hashlib.md5(f"{func_name}_{time.time()}".encode()).hexdigest()
    
    def cached_call(self, func: Callable, func_name: str, *args, **kwargs) -> Any:
        """
        Cache function call (optimized version)
        
        Args:
            func: Function
            func_name: Function name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        # Generate cache key
        cache_key = self.cache_key(func_name, *args, **kwargs)
        
        # Try to get from cache (shard locking)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            return cached_result
        
        # Cache miss, execute function
        with self.stats_lock:
            self.stats["cache_misses"] += 1
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Store in cache (shard locking)
        self.cache.put(cache_key, result)
        
        return result
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    max_workers: Optional[int] = None) -> List[Any]:
        """
        Parallel map function (optimized version)
        
        Args:
            func: Function
            items: List of items
            max_workers: Maximum number of worker threads
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        max_workers = max_workers or self.max_workers
        
        with self.stats_lock:
            self.stats["parallel_tasks"] += 1
        
        start_time = time.time()
        
        # Smart threshold: smaller threshold uses parallel processing
        if len(items) <= 1:
            results = [func(item) for item in items]
        else:
            # Parallel execution (using existing thread pool)
            try:
                future_to_index = {}
                for i, item in enumerate(items):
                    future = self.executor.submit(func, item)
                    future_to_index[future] = i
                
                # Collect results in original order
                results = [None] * len(items)
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        index = future_to_index[future]
                        results[index] = result
                    except Exception as e:
                        index = future_to_index[future]
                        logger.error(f"Parallel task failed (index {index}): {e}")
                        results[index] = None
                        
            except Exception as e:
                logger.error(f"Parallel processing failed, falling back to serial processing: {e}")
                results = [func(item) for item in items]
        
        execution_time = time.time() - start_time
        
        with self.stats_lock:
            # More accurate time savings estimation
            if len(items) > 1:
                estimated_serial_time = execution_time * min(max_workers, len(items))
                self.stats["total_time_saved"] += max(0, estimated_serial_time - execution_time)
        
        return results
    
    def batch_process(self, func: Callable, items: List[Any], 
                     batch_size: int = 32, max_workers: Optional[int] = None) -> List[Any]:
        """
        Batch processing (optimized version)
        
        Args:
            func: Function
            items: List of items
            batch_size: Batch processing size
            max_workers: Maximum number of worker threads
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Adaptive batch size
        optimal_batch_size = max(1, min(batch_size, len(items) // (max_workers or self.max_workers) + 1))
        
        # Process in batches
        batches = [items[i:i + optimal_batch_size] for i in range(0, len(items), optimal_batch_size)]
        
        def process_batch(batch):
            return [func(item) for item in batch]
        
        # Process batches in parallel
        batch_results = self.parallel_map(process_batch, batches, max_workers)
        
        # Merge results
        results = []
        for batch_result in batch_results:
            if batch_result:
                results.extend(batch_result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics (optimized version)"""
        with self.stats_lock:
            cache_stats = self.cache.get_stats()
            total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
            hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "cache": cache_stats,
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_hit_rate": hit_rate,
                "parallel_tasks": self.stats["parallel_tasks"],
                "total_time_saved": self.stats["total_time_saved"],
                "max_workers": self.max_workers,
                "optimization_type": "sharded_concurrent"
            }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "parallel_tasks": 0,
                "total_time_saved": 0.0
            }
    
    def __del__(self):
        """Destructor"""
        try:
            self.executor.shutdown(wait=True)
        except:
            pass 