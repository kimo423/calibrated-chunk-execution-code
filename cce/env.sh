#!/bin/bash
# Run a script inside the CalibrationToPrompt clean env (ctp-e0); repo root added to sys.path by cce/_bootstrap.py.
set -eu
export IDEA_V2_ROOT='/opt/cce'
export PYTHONUNBUFFERED=1
export PATH="/opt/runtime/miniconda3/bin:$PATH"
exec '/opt/CalibrationToPrompt/tools/run_clean_env.sh' conda run --no-capture-output -p /opt/CalibrationToPrompt/envs/ctp-e0 python -u "$@"
