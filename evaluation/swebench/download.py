"""Download SWE-bench Lite through the Hugging Face Dataset Viewer API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DATASET = "princeton-nlp/SWE-bench_Lite"
CONFIG = "default"
SPLIT = "test"
REVISION = "main"
API_ROOT = "https://datasets-server.huggingface.co"
HUB_ROOT = "https://huggingface.co"
REQUIRED_FIELDS = {
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "dataset",
    "config",
    "split",
    "requested_revision",
    "resolved_revision",
    "rows",
    "dataset_sha256",
}


def fetch_json(endpoint: str, **params: Any) -> dict[str, Any]:
    url = f"{API_ROOT}/{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "nemori-swebench-downloader/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS host
        return json.load(response)


def resolve_revision(dataset: str, revision: str = REVISION) -> str:
    """Resolve a branch/tag to an immutable Hugging Face dataset commit."""

    clean = revision.strip()
    if re.fullmatch(r"[0-9a-f]{40}", clean):
        return clean
    dataset_path = quote(dataset, safe="/")
    revision_path = quote(clean, safe="")
    url = f"{HUB_ROOT}/api/datasets/{dataset_path}/revision/{revision_path}"
    request = Request(url, headers={"User-Agent": "nemori-swebench-downloader/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS host
        payload = json.load(response)
    resolved = str(payload.get("sha") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError(f"Could not resolve dataset revision {dataset}@{revision}")
    return resolved


def download_rows(
    dataset: str = DATASET,
    config: str = CONFIG,
    split: str = SPLIT,
    revision: str = REVISION,
) -> list[dict[str, Any]]:
    split_info = fetch_json("splits", dataset=dataset, revision=revision)
    available = {
        (item.get("config"), item.get("split"))
        for item in split_info.get("splits", [])
    }
    if (config, split) not in available:
        raise ValueError(f"Dataset split is unavailable: {dataset}/{config}/{split}")

    records: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        payload = fetch_json(
            "rows",
            dataset=dataset,
            config=config,
            split=split,
            revision=revision,
            offset=offset,
            length=100,
        )
        page = [item.get("row") for item in payload.get("rows", [])]
        page = [item for item in page if isinstance(item, dict)]
        if not page:
            break
        records.extend(page)
        offset += len(page)
        total = int(payload.get("num_rows_total") or len(records))
        print(f"Downloaded {len(records)}/{total} rows", flush=True)

    if total is not None and len(records) != total:
        raise RuntimeError(f"Incomplete download: expected {total}, got {len(records)}")
    if not records:
        raise RuntimeError("The Dataset Viewer returned no rows")
    validate_records(records)
    return records


def validate_records(records: Any) -> list[dict[str, Any]]:
    """Validate the complete dataset payload before it can be committed."""

    if not isinstance(records, list) or not records:
        raise RuntimeError("SWE-bench dataset must be a non-empty JSON array")
    if not all(isinstance(item, dict) for item in records):
        raise RuntimeError("Every SWE-bench dataset row must be a JSON object")
    missing = [
        str(item.get("instance_id") or index)
        for index, item in enumerate(records)
        if not REQUIRED_FIELDS.issubset(item)
    ]
    if missing:
        raise RuntimeError(f"Rows are missing required fields: {', '.join(missing[:5])}")
    instance_ids = [str(item["instance_id"]).strip() for item in records]
    if any(not instance_id for instance_id in instance_ids):
        raise RuntimeError("Dataset contains an empty instance_id")
    if len(instance_ids) != len(set(instance_ids)):
        raise RuntimeError("Dataset contains duplicate instance_id values")
    return records


def atomic_write(path: Path, value: Any) -> None:
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


def write_manifest(
    path: Path,
    *,
    dataset: str,
    config: str,
    split: str,
    requested_revision: str,
    resolved_revision: str,
    rows: int,
) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"SWE-bench dataset is missing: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SWE-bench dataset is not valid JSON: {path}") from exc
    validate_records(records)
    if isinstance(rows, bool) or rows != len(records):
        raise RuntimeError(
            f"Dataset row count mismatch: declared {rows}, actual {len(records)}"
        )
    resolved_revision = resolved_revision.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        raise RuntimeError("resolved_revision must be an immutable 40-character commit SHA")
    if not all(value.strip() for value in (dataset, config, split, requested_revision)):
        raise RuntimeError("Dataset manifest identity fields must not be empty")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "config": config,
        "split": split,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "rows": rows,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path = path.with_suffix(".manifest.json")
    atomic_write(manifest_path, manifest)
    validate_bundle(
        path,
        expected_dataset=dataset,
        expected_config=config,
        expected_split=split,
        expected_revision=resolved_revision,
    )
    return manifest_path


def validate_bundle(
    path: Path,
    *,
    expected_dataset: str | None = None,
    expected_config: str | None = None,
    expected_split: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the dataset and its immutable sidecar form one bundle."""

    if not path.is_file():
        raise FileNotFoundError(f"SWE-bench dataset is missing: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SWE-bench dataset manifest is required but missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SWE-bench dataset manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SWE-bench dataset manifest must be a JSON object")
    missing = MANIFEST_REQUIRED_FIELDS - set(payload)
    if missing:
        raise RuntimeError(
            "SWE-bench dataset manifest is missing fields: " + ", ".join(sorted(missing))
        )
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported SWE-bench dataset manifest schema: {payload['schema_version']!r}"
        )
    for field in ("dataset", "config", "split", "requested_revision"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise RuntimeError(f"SWE-bench dataset manifest field {field} is invalid")
    revision = str(payload["resolved_revision"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("SWE-bench dataset manifest has no immutable revision")
    expected_hash = str(payload["dataset_sha256"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise RuntimeError("SWE-bench dataset manifest has an invalid dataset_sha256")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise RuntimeError("SWE-bench dataset bytes do not match its download manifest")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SWE-bench dataset is not valid JSON: {path}") from exc
    validate_records(records)
    declared_rows = payload["rows"]
    if isinstance(declared_rows, bool) or not isinstance(declared_rows, int):
        raise RuntimeError("SWE-bench dataset manifest rows must be an integer")
    if declared_rows != len(records):
        raise RuntimeError(
            f"Dataset row count mismatch: declared {declared_rows}, actual {len(records)}"
        )
    expectations = {
        "dataset": expected_dataset,
        "config": expected_config,
        "split": expected_split,
        "resolved_revision": expected_revision.lower() if expected_revision else None,
    }
    actual = {**payload, "resolved_revision": revision}
    for field, expected in expectations.items():
        if expected is not None and actual[field] != expected:
            raise RuntimeError(
                f"SWE-bench dataset manifest {field} mismatch: "
                f"expected {expected!r}, got {actual[field]!r}"
            )
    return actual


def write_dataset_bundle(
    path: Path,
    records: list[dict[str, Any]],
    *,
    dataset: str,
    config: str,
    split: str,
    requested_revision: str,
    resolved_revision: str,
) -> Path:
    """Write the dataset first and its sidecar last as the bundle commit marker."""

    validate_records(records)
    atomic_write(path, records)
    return write_manifest(
        path,
        dataset=dataset,
        config=config,
        split=split,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        rows=len(records),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "swebench_lite_test.json",
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()

    resolved_revision = resolve_revision(args.dataset, args.revision)
    records = download_rows(
        args.dataset,
        args.config,
        args.split,
        resolved_revision,
    )
    output = args.output.resolve()
    manifest = write_dataset_bundle(
        output,
        records,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        requested_revision=args.revision,
        resolved_revision=resolved_revision,
    )
    print(
        f"Saved {len(records)} rows at {resolved_revision} to {output}; "
        f"manifest: {manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
