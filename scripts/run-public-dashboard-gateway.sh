#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
runtime_dir="$project_root/.nemori-dashboard"
env_file="$runtime_dir/public-gateway.env"
cd "$project_root"

if [[ ! -f "$env_file" ]]; then
  printf 'Missing public gateway configuration: %s\n' "$env_file" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${NEMORI_PUBLIC_DASHBOARD_USERNAME:?Missing public dashboard username}"
: "${NEMORI_PUBLIC_DASHBOARD_PASSWORD_HASH:?Missing public dashboard password hash}"

export PATH="$project_root/.venv/bin:$HOME/.local/bin:$PATH"
export NEMORI_PUBLIC_DASHBOARD_HOST="${NEMORI_PUBLIC_DASHBOARD_HOST:-192.168.106.20}"
export NEMORI_PUBLIC_DASHBOARD_PORT="${NEMORI_PUBLIC_DASHBOARD_PORT:-18766}"
printf '%s\n' "$$" > "$runtime_dir/public-gateway.pid"

exec "$project_root/.venv/bin/python" -m uvicorn dashboard.public_gateway:app \
  --host "$NEMORI_PUBLIC_DASHBOARD_HOST" \
  --port "$NEMORI_PUBLIC_DASHBOARD_PORT" \
  --no-access-log
