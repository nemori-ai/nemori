import argparse
import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any

import nltk
import numpy as np
import torch  # <--- MODIFICATION: Import torch for device check
import transformers
from bert_score import score as bert_score
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm as async_tqdm  # <--- MODIFICATION: Use async-compatible tqdm

# --- Setup Section (largely unchanged) ---
logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()

try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK resources: {e}")

# <--- MODIFICATION: Smarter model loading with device placement ---
sentence_model = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model_name = "Qwen/Qwen3-Embedding-0.6B"
    sentence_model = SentenceTransformer(sentence_model_name, device=device)
    print(f"SentenceTransformer model '{sentence_model_name}' loaded successfully on '{device}'.")
except Exception as e:
    print(f"Failed to load SentenceTransformer model: {e}")


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


# --- locomo_grader (unchanged) ---
async def locomo_grader(llm_client, question: str, gold_answer: str, response: str) -> bool:
    # ... (no changes needed here)
    system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

    accuracy_prompt = f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You williolw23 be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": accuracy_prompt},
        ],
        temperature=0,
    )
    message_content = response.choices[0].message.content
    label = json.loads(message_content)["label"]
    parsed = LLMGrade(llm_judgment=label, llm_reasoning="")

    return parsed.llm_judgment.strip().lower() == "correct"


# --- NLP Metric Functions (largely unchanged, but now they are pure CPU functions) ---
def calculate_rouge_scores(gold_answer, response):
    # ... (no changes)
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = rouge_scores["rouge1"].fmeasure
        metrics["rouge2_f"] = rouge_scores["rouge2"].fmeasure
        metrics["rougeL_f"] = rouge_scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens, response_tokens):
    # ... (no changes)
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    try:
        smoothing = SmoothingFunction().method1
        weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens, weights=weight, smoothing_function=smoothing
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")

    return metrics


def calculate_meteor_score(gold_tokens, response_tokens):
    # ... (no changes)
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_f1_score(gold_tokens, response_tokens):
    # ... (no changes)
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


# <--- MODIFICATION: This function now accepts pre-computed semantic scores ---
def calculate_nlp_metrics(gold_answer, response, context, precomputed_semantic: Dict = None):
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""

    metrics = {"context_tokens": len(nltk.word_tokenize(context)) if context else 0}

    # Lexical metrics (fast, can stay here)
    gold_tokens = nltk.word_tokenize(gold_answer.lower())
    response_tokens = nltk.word_tokenize(response.lower())

    metrics["lexical"] = {}
    metrics["lexical"]["f1"] = calculate_f1_score(gold_tokens, response_tokens)
    metrics["lexical"].update(calculate_rouge_scores(gold_answer, response))
    metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))
    metrics["lexical"]["meteor"] = calculate_meteor_score(gold_tokens, response_tokens)

    # Use pre-computed semantic scores if available
    metrics["semantic"] = precomputed_semantic or {}

    return metrics


# <--- MODIFICATION: New function to run CPU-bound tasks in a separate thread ---
async def run_cpu_bound_nlp_metrics(gold_answer, response, context, precomputed_semantic):
    return await asyncio.to_thread(calculate_nlp_metrics, gold_answer, response, context, precomputed_semantic)


# <--- MODIFICATION: Core processing logic for a single item, now fully async ---
async def process_response_item(
    response_item: Dict, llm_client: AsyncOpenAI, num_runs: int, semaphore: asyncio.Semaphore
) -> Dict:
    async with semaphore:
        question = response_item.get("question")
        answer = response_item.get("answer")
        ground_truth = response_item.get("golden_answer")

        # Perform I/O-bound LLM grading concurrently
        grading_tasks = [locomo_grader(llm_client, question, ground_truth, answer) for _ in range(num_runs)]
        judgments = await asyncio.gather(*grading_tasks)
        judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

        # Perform CPU-bound NLP metrics calculation in a thread
        nlp_metrics = await run_cpu_bound_nlp_metrics(
            ground_truth, answer, response_item.get("search_context", ""), response_item.get("precomputed_semantic", {})
        )

        response_item["llm_judgments"] = judgments_dict
        response_item["nlp_metrics"] = nlp_metrics
        response_item["total_duration_ms"] = response_item.get("response_duration_ms", 0.0) + response_item.get(
            "search_duration_ms", 0.0
        )

        return response_item


def convert_numpy_types(obj):
    # ... (no changes)
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


# <--- MODIFICATION: The main function is heavily refactored for performance ---
async def main(frame, version="default", num_runs=1, llm_concurrency=10, batch_size=32):
    print(f"\n=== Starting LoCoMo evaluation for {frame} (v: {version}) with {num_runs} run(s) ===")
    print(f"LLM Concurrency: {llm_concurrency}, Embedding Batch Size: {batch_size}")

    results_dir = f"results/locomo/{frame}-{version}"
    response_path = f"{results_dir}/{frame}_locomo_responses.json"
    judged_path = f"{results_dir}/{frame}_locomo_judged.json"
    os.makedirs(results_dir, exist_ok=True)

    load_dotenv()
    llm_api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
    llm_base_url = "https://jeniya.cn/v1"

    oai_client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_base_url)

    with open(response_path) as file:
        locomo_responses = json.load(file)

    # 1. Flatten all responses into a single list for global processing
    all_responses = []
    for group_id, group_responses in locomo_responses.items():
        for response in group_responses:
            if response.get("golden_answer") is not None:
                response["group_id"] = group_id
                all_responses.append(response)

    print(f"Found {len(all_responses)} total valid responses to evaluate.")
    if not all_responses:
        print("No responses to evaluate. Exiting.")
        return

    # 2. BATCHED SEMANTIC CALCULATIONS (CPU-intensive)
    print("Pre-computing semantic scores in batches (this may take a while)...")
    gold_texts = [str(r["golden_answer"]) for r in all_responses]
    response_texts = [str(r["answer"]) for r in all_responses]

    # Batch SentenceTransformer embeddings
    if sentence_model:
        all_embeddings = sentence_model.encode(
            gold_texts + response_texts, batch_size=batch_size, show_progress_bar=True
        )
        gold_embeddings = all_embeddings[: len(gold_texts)]
        response_embeddings = all_embeddings[len(gold_texts) :]

        similarities = [1 - cosine(g, r) for g, r in zip(gold_embeddings, response_embeddings)]
    else:
        similarities = [0.0] * len(all_responses)

    # Batch BERTScore
    _, _, bert_f1s = bert_score(
        response_texts,
        gold_texts,
        lang="en",
        rescale_with_baseline=True,
        device=sentence_model.device if sentence_model else "cpu",
        batch_size=batch_size,
        verbose=True,
    )

    # Add pre-computed scores back to each response item
    for i, response in enumerate(all_responses):
        response["precomputed_semantic"] = {"similarity": similarities[i], "bert_f1": bert_f1s[i].item()}
    print("✅ Semantic scores pre-computation complete.")

    # 3. CONCURRENT PROCESSING (I/O-bound and threaded CPU)
    print(f"Starting concurrent evaluation of {len(all_responses)} items...")
    semaphore = asyncio.Semaphore(llm_concurrency)

    tasks = [process_response_item(response, oai_client, num_runs, semaphore) for response in all_responses]

    # Use async_tqdm for a progress bar
    results = await async_tqdm.gather(*tasks, desc="Evaluating responses")

    # 4. Re-group results and save
    all_grades = {group_id: [] for group_id in locomo_responses.keys()}
    for res in results:
        group_id = res.pop("group_id")
        res.pop("precomputed_semantic", None)  # Clean up temporary key
        all_grades[group_id].append(res)

    print("\n=== Evaluation Complete: Calculating final scores ===")
    # ... (score calculation logic remains the same)
    run_scores = []
    evaluated_count = 0
    if num_runs > 0:
        for i in range(1, num_runs + 1):
            judgment_key = f"judgment_{i}"
            current_run_correct_count = 0
            current_run_total_count = 0
            for group in all_grades.values():
                for response in group:
                    if judgment_key in response["llm_judgments"]:
                        if response["llm_judgments"][judgment_key]:
                            current_run_correct_count += 1
                        current_run_total_count += 1

            if current_run_total_count > 0:
                run_accuracy = current_run_correct_count / current_run_total_count
                run_scores.append(run_accuracy)

        evaluated_count = current_run_total_count

    if evaluated_count > 0:
        mean_of_scores = np.mean(run_scores)
        std_of_scores = np.std(run_scores)
        print(f"LLM-as-a-Judge Mean Score: {mean_of_scores:.4f}")
        print(f"LLM-as-a-Judge Standard Deviation: {std_of_scores:.4f}")
        print(f"(Calculated from {num_runs} separate runs over {evaluated_count} questions)")
        print(f"Individual run scores: {[round(s, 4) for s in run_scores]}")
    else:
        print("No responses were evaluated")
        print("LLM-as-a-Judge score: N/A (0/0)")

    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)
        print(f"Saved detailed evaluation results to {judged_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (arguments are the same, but 'workers' is renamed to 'llm_concurrency')
    parser.add_argument("--lib", type=str, required=True)
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument(
        "--options",
        nargs="+",
        default=["lexical", "semantic"],
        help="This is now unused as all metrics are calculated.",
    )
    parser.add_argument("--llm_concurrency", type=int, default=10, help="Max concurrent requests to the LLM API")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding and BERTScore calculation")
    args = parser.parse_args()

    asyncio.run(main(args.lib, args.version, args.num_runs, args.llm_concurrency, args.batch_size))
