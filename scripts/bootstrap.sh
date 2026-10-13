#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$repo_dir/.cache/pip}"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e '.[dev]'
.venv/bin/python -m pip check
