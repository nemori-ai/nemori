"""Helper script to bootstrap a Nemori workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemori import MemoryConfig
import json


def init_workspace(path: str) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "memories").mkdir(exist_ok=True)
    (root / "chroma_db").mkdir(exist_ok=True)

    config = MemoryConfig(storage_path=str(root / "memories"), chroma_persist_directory=str(root / "chroma_db"))
    (root / "nemori_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"Workspace initialized at {root}")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Initialize a Nemori workspace")
    parser.add_argument("path", nargs="?", default="./nemori_workspace")
    args = parser.parse_args()
    init_workspace(args.path)
