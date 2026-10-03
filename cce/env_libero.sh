#!/bin/bash
# Run a cce script inside the cloned LIBERO env (idea_v2/data/envs/libero).
# Mirrors cce/env.sh, which targets the ctp-e0 env used by the policy server.
set -eu
export IDEA_V2_ROOT='/opt/cce'
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH='/opt/cce/data/libero_config'
cd "$IDEA_V2_ROOT"
exec /opt/cce/data/envs/libero/bin/python -u "$@"
