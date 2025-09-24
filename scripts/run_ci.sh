#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
pytest tests/test_facade.py tests/test_repositories_contract.py
# placeholder for lint/benchmark hooks
https://chatgpt.com/codex