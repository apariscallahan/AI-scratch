#!/usr/bin/env bash
# NeuroBlocks — start the block editor (Linux / macOS).
# First run creates .venv and installs the packages (PyTorch is reused if already installed).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up NeuroBlocks for the first time…"
  "$PY" -m venv .venv --system-site-packages
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -c "import torch" 2>/dev/null || .venv/bin/python -m pip install torch
  .venv/bin/python -m pip install -e .
fi
exec .venv/bin/python -m neuroblocks gui "$@"
