"""Automatic preparation of the revision-pinned SWE-bench Lite test bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from evaluation.swebench.download import (
    CONFIG,
    DATASET,
    download_rows,
    resolve_revision,
    validate_bundle,
    write_dataset_bundle,
)


def ensure_test_dataset(
    path: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Validate the local Lite test bundle or download it automatically."""

    announce = log or (lambda _message: None)
    try:
        validate_bundle(
            path,
            expected_dataset=DATASET,
            expected_config=CONFIG,
            expected_split="test",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("SWE-bench Lite test dataset is not an array")
        announce(f"Using validated SWE-bench Lite test bundle ({len(payload)} rows)")
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        announce(f"Local test bundle unavailable ({exc}); downloading automatically")

    resolved = resolve_revision(DATASET, "main")
    rows = download_rows(DATASET, CONFIG, "test", resolved)
    write_dataset_bundle(
        path,
        rows,
        dataset=DATASET,
        config=CONFIG,
        split="test",
        requested_revision="main",
        resolved_revision=resolved,
    )
    announce(f"Downloaded and pinned {len(rows)} SWE-bench Lite test rows")
    return rows
