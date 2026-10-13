#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/activate_env.sh
export ROBOT_PROFILE="${ROBOT_PROFILE:-yam007_config}"
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    exec uv run --frozen --extra deploy scripts/dry_run_dagger.py "$@"
fi
if [[ " $* " != *" --help "* && " $* " != *" -h "* ]]; then
    uv run --frozen --extra deploy scripts/check_dagger_hardware.py
fi
exec uv run --frozen --extra deploy deploy/dagger.py \
    --clip.cache-dir "$repo_root/cache/clip" "$@"
