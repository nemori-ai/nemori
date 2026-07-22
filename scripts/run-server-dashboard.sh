#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd "$project_root"

export PATH="$project_root/.venv/bin:$HOME/.local/bin:$PATH"
export NEMORI_DASHBOARD_HOST="${NEMORI_DASHBOARD_HOST:-127.0.0.1}"
export NEMORI_DASHBOARD_PORT="${NEMORI_DASHBOARD_PORT:-18765}"
printf '%s\n' "$$" > "$project_root/.nemori-dashboard.pid"

exec "$project_root/.venv/bin/python" -m dashboard
