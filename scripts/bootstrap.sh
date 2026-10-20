#!/usr/bin/env bash
# Run with bash. This script installs only inside this checkout's .venv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:-}"
if [[ "$#" -gt 1 || ( -n "$mode" && "$mode" != "--gpu" ) ]]; then
    printf '%s\n' "Usage: bash scripts/bootstrap.sh [--gpu]" >&2
    exit 2
fi
python_bin="${LAB_PYTHON:-python3}"
if [[ ! -x ".venv/bin/python" ]]; then
    "$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12; set LAB_PYTHON to an existing interpreter."'
    if [[ -e ".venv" ]]; then
        printf '%s\n' ".venv exists but is not a Linux Python environment. Inspect it; nothing was overwritten." >&2
        exit 1
    fi
    "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Existing venv must use Python 3.10-3.12."'
.venv/bin/python scripts/cpu_checks.py
.venv/bin/python scripts/validate_plan.py

if [[ "$mode" == "--gpu" ]]; then
    .venv/bin/python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
    .venv/bin/python -m pip install -r requirements-training.txt
    .venv/bin/python -m pip check
    mkdir -p runs
    .venv/bin/python -m pip freeze --all > runs/requirements-resolved.txt
    .venv/bin/python scripts/doctor.py --require-gpu --output runs/preflight-gpu.json
fi
printf '%s\n' "Ready for the next documented step. Activate: source .venv/bin/activate"
