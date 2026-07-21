from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from evaluation.locomo.metrics.llm_judge import evaluate_llm_judge
    from evaluation.locomo.metrics.utils import calculate_bleu_scores, calculate_metrics
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from metrics.llm_judge import evaluate_llm_judge
    from metrics.utils import calculate_bleu_scores, calculate_metrics

CHECKPOINT_REPLACE_ATTEMPTS = 8
CHECKPOINT_RETRY_BASE_SECONDS = 0.05


def flatten_scored_items(data: dict[str, list[dict[str, Any]]]) -> list[tuple[str, int, dict[str, Any]]]:
    items: list[tuple[str, int, dict[str, Any]]] = []
    for key, records in data.items():
        for index, item in enumerate(records):
            if str(item.get("category")) == "5":
                continue
            items.append((str(key), index, item))
    return items


def record_identity(key: str, item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Identify a scored question without depending on concurrent output order."""
    return (
        str(key),
        str(item.get("question", "")),
        str(item.get("answer", "")),
        str(item.get("response", "")),
        str(item.get("category", "")),
    )


def resume_checkpoint(
    output_path: Path,
    work_items: list[tuple[str, int, dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, int, dict[str, Any]]]]:
    """Load valid completed records and return only work that still needs judging."""
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not output_path.exists():
        return results, list(work_items)

    try:
        checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return results, list(work_items)
    if not isinstance(checkpoint, dict):
        return results, list(work_items)

    expected = Counter(record_identity(key, item) for key, _index, item in work_items)
    restored: Counter[tuple[str, str, str, str, str]] = Counter()
    for key, records in checkpoint.items():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            identity = record_identity(str(key), record)
            if restored[identity] >= expected[identity]:
                continue
            results[str(key)].append(record)
            restored[identity] += 1

    remaining_restored = restored.copy()
    pending: list[tuple[str, int, dict[str, Any]]] = []
    for item in work_items:
        identity = record_identity(item[0], item[2])
        if remaining_restored[identity] > 0:
            remaining_restored[identity] -= 1
        else:
            pending.append(item)
    return results, pending


def evaluate_record(item_data: tuple[str, int, dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
    key, index, item = item_data
    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
    category = str(item["category"])
    question = str(item["question"])
    metrics = calculate_metrics(pred_answer, gt_answer)
    bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
    llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)
    return (
        key,
        index,
        {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
        },
    )


def emit_progress(completed: int, total: int) -> None:
    print(
        "NEMORI_PROGRESS "
        + json.dumps({"phase": "evaluate", "completed": completed, "total": total}),
        flush=True,
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(value, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for attempt in range(CHECKPOINT_REPLACE_ATTEMPTS):
            try:
                tmp_path.replace(path)
                return
            except OSError as exc:
                last_error = exc
                if attempt + 1 < CHECKPOINT_REPLACE_ATTEMPTS:
                    time.sleep(min(CHECKPOINT_RETRY_BASE_SECONDS * (2**attempt), 1.0))
        raise OSError(f"Could not replace {path}; recovery kept at {tmp_path}") from last_error
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo results")
    parser.add_argument("--input_file", type=str, default="locomo/results.json")
    parser.add_argument("--output_file", type=str, default="locomo/metrics.json")
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    work_items = flatten_scored_items(data)
    total = len(work_items)
    results, pending_items = resume_checkpoint(output_path, work_items)
    completed = total - len(pending_items)
    if completed:
        print(
            f"Resuming judge evaluation from checkpoint ({completed}/{total} questions)",
            flush=True,
        )
    emit_progress(completed, total)

    max_workers = max(1, int(args.max_workers))
    checkpoint_every = max(1, int(args.checkpoint_every))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_record, item) for item in pending_items]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Judge questions",
        ):
            key, _index, result = future.result()
            results[key].append(result)
            completed += 1
            emit_progress(completed, total)
            if completed % checkpoint_every == 0 or completed == total:
                atomic_json(output_path, results)

    atomic_json(output_path, results)
    print(f"Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
