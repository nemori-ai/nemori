#!/usr/bin/env python3
"""
LongMemEval Result Evaluation Script

Use zep's LongMemEval evaluation standards to evaluate the results of memory systems and baselines.
Support evaluation of different types of questions and generate detailed evaluation reports.
"""

import json
import os
import sys
import time
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class Grade(BaseModel):
    is_correct: str = Field(description='yes or no')


class LongMemEvalEvaluator:
    """LongMemEval Result Evaluator"""
    
    def __init__(self, model=os.getenv("OPENAI_MODEL"), max_concurrent=10):
        """
        Initialize evaluator
        
        Args:
            model: Model used for evaluation
            max_concurrent: Maximum concurrency
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self.client = AsyncOpenAI()
        
        # Evaluation prompt template
        self.TEMPORAL_REASONING_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""
        
        self.KNOWLEDGE_UPDATE_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""
        
        self.SINGLE_SESSION_PREFERENCE_PROMPT = """
I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
B: {question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
A: {response}
</RESPONSE>
"""
        
        self.DEFAULT_PROMPT = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""
    
    async def evaluate_single_response(self, question: str, gold_answer: str, 
                                     response: str, question_type: str) -> bool:
        """
        Evaluate single response
        
        Args:
            question: Question
            gold_answer: Gold standard answer
            response: Model response
            question_type: Question type
            
        Returns:
            Whether correct
        """
        system_prompt = """
You are an expert grader that determines if answers to questions match a gold standard answer
"""
        
        # Select prompt based on question type
        if question_type == 'temporal-reasoning':
            prompt = self.TEMPORAL_REASONING_PROMPT.format(
                question=question, gold_answer=gold_answer, response=response
            )
        elif question_type == 'knowledge-update':
            prompt = self.KNOWLEDGE_UPDATE_PROMPT.format(
                question=question, gold_answer=gold_answer, response=response
            )
        elif question_type == 'single-session-preference':
            prompt = self.SINGLE_SESSION_PREFERENCE_PROMPT.format(
                question=question, gold_answer=gold_answer, response=response
            )
        else:
            prompt = self.DEFAULT_PROMPT.format(
                question=question, gold_answer=gold_answer, response=response
            )
        
        try:
            response_obj = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=Grade,
                temperature=0,
            )
            
            result = response_obj.choices[0].message.parsed
            return result.is_correct.strip().lower() == 'yes'
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return False
    
    async def evaluate_results_file(self, results_file: str) -> Dict[str, Any]:
        """
        Evaluate results file
        
        Args:
            results_file: Path to results file
            
        Returns:
            Evaluation results
        """
        print(f"\n=== Evaluate file: {results_file} ===")
        
        # Load results data
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total questions: {len(data)}")
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for item in data:
            if item.get('error', False):
                continue
                
            task = self.evaluate_single_response(
                question=item['question'],
                gold_answer=item['answer'],
                response=item['response'],
                question_type=item.get('question_type', 'default')
            )
            evaluation_tasks.append((task, item))
        
        print(f"Valid questions: {len(evaluation_tasks)}")
        
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_with_semaphore(task_and_item):
            task, item = task_and_item
            async with semaphore:
                is_correct = await task
                return is_correct, item
        
        # Execute evaluation
        print("Start evaluation...")
        results = await tqdm.gather(
            *[evaluate_with_semaphore(task_and_item) for task_and_item in evaluation_tasks],
            desc="Evaluation progress"
        )
        
        # Statistics results
        return self._calculate_statistics(results, results_file)
    
    def _calculate_statistics(self, results: List[tuple], results_file: str) -> Dict[str, Any]:
        """
        Calculate statistics
        
        Args:
            results: Evaluation results list
            results_file: Results file name
            
        Returns:
            Statistics data
        """
        # Basic statistics
        total_questions = len(results)
        correct_count = sum(1 for is_correct, _ in results if is_correct)
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        # Statistics by question type
        type_stats = {}
        for is_correct, item in results:
            question_type = item.get('question_type', 'unknown')
            if question_type not in type_stats:
                type_stats[question_type] = {'correct': 0, 'total': 0}
            
            type_stats[question_type]['total'] += 1
            if is_correct:
                type_stats[question_type]['correct'] += 1
        
        # Calculate accuracy by question type
        for qtype in type_stats:
            stats = type_stats[qtype]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Error case analysis
        error_cases = []
        for is_correct, item in results:
            if not is_correct:
                error_cases.append({
                    'question_id': item['question_id'],
                    'question_type': item.get('question_type', 'unknown'),
                    'question': item['question'],
                    'gold_answer': item['answer'],
                    'model_response': item['response']
                })
        
        # Time performance statistics (if any)
        response_times = [item.get('response_time', 0) for _, item in results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # System information
        system_version = results[0][1].get('system_version', 'unknown') if results else 'unknown'
        
        return {
            'file_name': os.path.basename(results_file),
            'system_version': system_version,
            'evaluation_model': self.model,
            'total_questions': total_questions,
            'correct_answers': correct_count,
            'overall_accuracy': accuracy,
            'accuracy_by_type': type_stats,
            'average_response_time': avg_response_time,
            'error_cases_count': len(error_cases),
            'error_cases': error_cases[:10],  # Only save the first 10 error cases
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def print_evaluation_summary(self, stats: Dict[str, Any]):
        """
        Print evaluation summary
        
        Args:
            stats: Statistics data
        """
        print(f"\n" + "="*60)
        print(f"LongMemEval evaluation results - {stats['file_name']}")
        print("="*60)
        
        print(f"System version: {stats['system_version']}")
        print(f"Evaluation model: {stats['evaluation_model']}")
        print(f"Evaluation time: {stats['evaluation_timestamp']}")
        
        print(f"\n--- Overall performance ---")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Correct answers: {stats['correct_answers']}")
        print(f"Overall accuracy: {stats['overall_accuracy']:.3f} ({stats['overall_accuracy']*100:.1f}%)")
        print(f"Average response time: {stats['average_response_time']:.2f} seconds")
        
        print(f"\n--- Analysis by question type ---")
        for qtype, type_stats in stats['accuracy_by_type'].items():
            print(f"{qtype}: {type_stats['correct']}/{type_stats['total']} "
                  f"= {type_stats['accuracy']:.3f} ({type_stats['accuracy']*100:.1f}%)")
        
        print(f"\n--- Error analysis ---")
        print(f"Total error cases: {stats['error_cases_count']}")
        if stats['error_cases']:
            print(f"First few error cases:")
            for i, case in enumerate(stats['error_cases'][:3], 1):
                print(f"\n{i}. Question ID: {case['question_id']} ({case['question_type']})")
                print(f"   Question: {case['question'][:100]}...")
                print(f"   Gold answer: {case['gold_answer']}")
                print(f"   Model response: {case['model_response'][:100]}...")
    
    def save_detailed_report(self, stats: Dict[str, Any], output_file: str):
        """
        Save detailed evaluation report
        
        Args:
            stats: Statistics data
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
            print(f"\nDetailed evaluation report saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval Results Evaluator")
    parser.add_argument("results_files", nargs="+", help="Results file path", default="results.json")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Evaluation model (default: gpt-4o-mini)")
    parser.add_argument("--output", type=str, 
                        help="Path to save evaluation report")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Maximum concurrent evaluation count (default: 10)")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LongMemEvalEvaluator(
        model=args.model,
        max_concurrent=args.max_concurrent
    )
    
    all_stats = []
    
    # Evaluate each file
    for results_file in args.results_files:
        if not os.path.exists(results_file):
            print(f"File not found: {results_file}")
            continue
        
        try:
            stats = await evaluator.evaluate_results_file(results_file)
            evaluator.print_evaluation_summary(stats)
            all_stats.append(stats)
            
        except Exception as e:
            print(f"Error evaluating file {results_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save evaluation report
    if args.output and all_stats:
        if len(all_stats) == 1:
            evaluator.save_detailed_report(all_stats[0], args.output)
        else:
            # Multiple file evaluation, save summary report
            summary_report = {
                'evaluation_summary': {
                    'total_files': len(all_stats),
                    'evaluation_model': args.model,
                    'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'individual_results': all_stats
            }
            evaluator.save_detailed_report(summary_report, args.output)
    
    # Print comparison summary
    if len(all_stats) > 1:
        print(f"\n" + "="*60)
        print("Multiple file comparison summary")
        print("="*60)
        for stats in all_stats:
            print(f"{stats['file_name']}: {stats['overall_accuracy']:.3f} "
                  f"({stats['overall_accuracy']*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main()) 