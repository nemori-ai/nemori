"""Run the official SWE-bench harness and normalize its artifacts for Nemori."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from swebench.harness import run_evaluation


def emit(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def individual_report(run_id: str, instance_id: str) -> tuple[dict[str, Any], str | None]:
    root = Path("logs") / "run_evaluation" / run_id
    matches = list(root.glob(f"*/{instance_id}/report.json"))
    for path in matches:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        detail = payload.get(instance_id)
        if isinstance(detail, dict):
            return detail, path.parent.parent.name.replace("__", "/")
    return {}, None


def normalize_report(report: dict[str, Any], instance_ids: list[str]) -> dict[str, Any]:
    selected = set(instance_ids)

    def selected_ids(key: str) -> list[str]:
        return sorted(selected.intersection(str(item) for item in report.get(key, [])))

    completed = selected_ids("completed_ids")
    resolved = selected_ids("resolved_ids")
    unresolved = selected_ids("unresolved_ids")
    empty = selected_ids("empty_patch_ids")
    errors = selected_ids("error_ids")
    submitted = selected_ids("submitted_ids")
    incomplete = sorted(selected - set(submitted))
    return {
        "total_instances": len(instance_ids),
        "submitted_instances": len(submitted),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(errors),
        "completed_ids": completed,
        "incomplete_ids": incomplete,
        "empty_patch_ids": empty,
        "submitted_ids": submitted,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "error_ids": errors,
        "schema_version": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", choices=("none", "base", "env", "instance"), default="env")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--namespace", default="swebench")
    args = parser.parse_args()

    instances = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(instances, list) or not instances:
        raise ValueError("The selected SWE-bench input is empty")
    instance_ids = [str(item["instance_id"]) for item in instances]
    emit(
        "NEMORI_TRACE",
        {
            "event": "swebench_prepare",
            "flow": "ingest",
            "nodes": ["dataset", "image", "checkout", "dependencies"],
            "instances": len(instance_ids),
            "summary": f"准备 {len(instance_ids)} 个 SWE-bench Docker 任务环境",
        },
    )

    report_path = run_evaluation.main(
        # The official harness accepts a JSON path. Using the isolated local
        # subset avoids a second Hugging Face download from inside WSL2 and
        # makes retries deterministic when WSL networking is unavailable.
        dataset_name=str(args.input),
        split=args.split,
        instance_ids=instance_ids,
        predictions_path=args.predictions,
        max_workers=max(1, args.max_workers),
        force_rebuild=False,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=4096,
        run_id=args.run_id,
        timeout=max(60, args.timeout),
        namespace=None if args.namespace.lower() == "none" else args.namespace,
        rewrite_reports=False,
        modal=False,
        report_dir=".",
    )
    if report_path is None:
        raise RuntimeError("SWE-bench did not produce a run report")
    raw_report_path = Path(report_path)
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    report = normalize_report(raw_report, instance_ids)
    atomic_json(args.report, report)

    results: list[dict[str, Any]] = []
    completed_ids = set(report["completed_ids"])
    resolved_ids = set(report["resolved_ids"])
    error_ids = set(report["error_ids"])
    for index, item in enumerate(instances, start=1):
        instance_id = str(item["instance_id"])
        detail, model_name = individual_report(args.run_id, instance_id)
        resolved = instance_id in resolved_ids or bool(detail.get("resolved"))
        if instance_id in error_ids:
            status = "harness error"
        elif instance_id not in completed_ids:
            status = "not completed"
        elif resolved:
            status = "resolved"
        elif detail.get("patch_successfully_applied"):
            status = "patch applied; tests failed"
        else:
            status = "patch failed"
        results.append(
            {
                "question_id": instance_id,
                "question": item.get("problem_statement", ""),
                "answer": "gold patch" if args.predictions == "gold" else model_name,
                "response": status,
                "category": item.get("repo", "unknown"),
                "llm_score": 1 if resolved else 0,
                "resolved": resolved,
                "patch_successfully_applied": detail.get("patch_successfully_applied"),
                "model_name_or_path": model_name or ("gold" if args.predictions == "gold" else None),
            }
        )
        emit(
            "NEMORI_PROGRESS",
            {"phase": "evaluate", "completed": index, "total": len(instances)},
        )
    atomic_json(args.results, results)
    emit(
        "NEMORI_TRACE",
        {
            "event": "swebench_complete",
            "flow": "search",
            "nodes": ["prediction", "apply", "tests", "grade", "report"],
            "resolved": report["resolved_instances"],
            "completed": report["completed_instances"],
            "summary": (
                f"SWE-bench 完成 {report['completed_instances']}/{len(instance_ids)}，"
                f"解决 {report['resolved_instances']}"
            ),
        },
    )

    if raw_report_path.resolve() != args.report.resolve():
        raw_report_path.unlink(missing_ok=True)
    if report["error_instances"] or report["completed_instances"] < report["submitted_instances"]:
        raise RuntimeError(
            "SWE-bench harness did not complete every submitted instance: "
            f"completed={report['completed_instances']}, "
            f"submitted={report['submitted_instances']}, errors={report['error_instances']}"
        )


if __name__ == "__main__":
    main()
