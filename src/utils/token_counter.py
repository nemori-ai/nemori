"""
Token Counter - 使用tiktoken统计LLM调用的token使用情况
"""

import tiktoken
import threading
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class TokenStats:
    """Token统计数据"""
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    category: str = ""


class TokenCounter:
    """全局Token统计器（线程安全的单例模式）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.stats: Dict[str, TokenStats] = {}
            self.stats_lock = threading.Lock()
            self.initialized = True
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数
        
        Args:
            text: 要计数的文本
            
        Returns:
            token数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def add_llm_call(self, category: str, input_text: str, output_text: str):
        """记录一次LLM调用的token使用
        
        Args:
            category: 调用类别（batch_segmentation, episode_generation等）
            input_text: 输入文本（prompt）
            output_text: 输出文本（response）
        """
        if not category:
            return
            
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        
        with self.stats_lock:
            if category not in self.stats:
                self.stats[category] = TokenStats(category=category)
            
            self.stats[category].input_tokens += input_tokens
            self.stats[category].output_tokens += output_tokens
            self.stats[category].call_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要
        
        Returns:
            包含各类别统计信息的字典
        """
        with self.stats_lock:
            summary = {}
            total_input = 0
            total_output = 0
            total_calls = 0
            
            for cat, stats in self.stats.items():
                summary[cat] = {
                    "calls": stats.call_count,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "total_tokens": stats.input_tokens + stats.output_tokens
                }
                total_input += stats.input_tokens
                total_output += stats.output_tokens
                total_calls += stats.call_count
            
            summary["TOTAL"] = {
                "calls": total_calls,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output
            }
            
            return summary
    
    def print_summary(self, title: str = "Token Usage Summary"):
        """打印统计摘要（格式化输出）
        
        Args:
            title: 标题
        """
        summary = self.get_summary()
        
        if not summary or len(summary) <= 1:  # 只有TOTAL或为空
            print(f"\n=== {title} ===")
            print("No token usage data collected.")
            return
        
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print('='*80)
        print(f"{'Category':<25} {'Calls':>8} {'Input Tokens':>15} {'Output Tokens':>15} {'Total':>15}")
        print('-'*80)
        
        # 按类别排序（TOTAL除外）
        categories = sorted([k for k in summary.keys() if k != "TOTAL"])
        
        for cat in categories:
            stats = summary[cat]
            print(
                f"{cat:<25} "
                f"{stats['calls']:>8,} "
                f"{stats['input_tokens']:>15,} "
                f"{stats['output_tokens']:>15,} "
                f"{stats['total_tokens']:>15,}"
            )
        
        # 打印总计
        if "TOTAL" in summary:
            print('-'*80)
            total = summary["TOTAL"]
            print(
                f"{'TOTAL':<25} "
                f"{total['calls']:>8,} "
                f"{total['input_tokens']:>15,} "
                f"{total['output_tokens']:>15,} "
                f"{total['total_tokens']:>15,}"
            )
        
        print('='*80)
    
    def reset(self):
        """重置所有统计数据"""
        with self.stats_lock:
            self.stats.clear()


__all__ = ["TokenCounter", "TokenStats"]

