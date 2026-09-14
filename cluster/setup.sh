#!/usr/bin/env bash
# Set up Tina/ on macha: Python environment, tests, data, validation.
#
# macha has no tmux, so start tmux on dagda and log into macha from inside it.
# The dagda-to-macha connection stays inside FSU, so it survives your own
# connection dropping:
#     ssh <username>@dagda.hep.fsu.edu
#     tmux new -s setup
#     ssh macha                   (password; this also gives a Kerberos ticket)
#     cd ~/GSOC_2026_PINNDE/Tina && bash cluster/setup.sh 2>&1 | tee ~/setup.log
# Detach with Ctrl-b then d; reattach on dagda with: tmux attach -t setup
#
# On macha the home folder is only readable with a Kerberos ticket, which lasts
# 10 hours. For longer jobs renew it with `kinit -R` (no password needed, up to
# 2 days after login).
#
# Safe to rerun: finished steps are skipped. Download the data first on dagda
# with cluster/get_data.sh; step 4 then only checks it.
#
# Setting: PYTHON=python3.12 is the interpreter used to create .venv.

set -euo pipefail
cd "$(dirname "$0")/.."                 # the Tina/ folder
export OPENBLAS_NUM_THREADS=1           # without it sklearn busy-waits and looks hung

PYTHON=${PYTHON:-python3.12}

step() { printf '\n=== %s ===\n' "$1"; }

step "1. Python environment (.venv)"
# Checks for pip, not python: a failed venv creation leaves python behind.
if [ ! -x .venv/bin/pip ]; then
    "$PYTHON" -m venv --clear .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
fi
.venv/bin/python --version
# Install only when requirements.txt differs from what was last installed.
if ! cmp -s cluster/requirements.txt .venv/installed-requirements.txt; then
    # No working NVIDIA driver here (macha's only card, a GeForce GT 730, is too
    # old for current CUDA), so take the CPU build of torch and skip the CUDA
    # libraries. requirements.txt's torch==2.13.0 accepts 2.13.0+cpu.
    if ! command -v nvidia-smi >/dev/null; then
        .venv/bin/python -m pip install --no-cache-dir torch==2.13.0+cpu \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple
    fi
    .venv/bin/python -m pip install --no-cache-dir -r cluster/requirements.txt
    cp cluster/requirements.txt .venv/installed-requirements.txt
fi

step "2. Tests"
.venv/bin/python -m pytest pinnde_eval/tests flow_matching/tests -q

step "3. GPU"
.venv/bin/python - <<'EOF'
import torch
from flow_matching import sample, train_flow_matching

print("torch", torch.__version__)
if not torch.cuda.is_available():
    print("no usable GPU on this machine: training runs on CPU")
else:
    print("GPU:", torch.cuda.get_device_name(0))
    model, _ = train_flow_matching(torch.randn(2000, 7), dim=7, n_steps=200, device="cuda")
    sample(model, 1000, 7, steps=20, device="cuda")
    print("200 training steps and 1000 samples on the GPU: OK")
EOF

step "4. Data (CaloChallenge ds2, 2 files, 2.7 GB)"
bash cluster/get_data.sh

step "5. Classical-test validation on all 200,000 showers"
.venv/bin/python -m pinnde_eval.validate_classical
