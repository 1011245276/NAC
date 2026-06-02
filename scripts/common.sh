#!/bin/bash
# ============================================================
# NAC: Common setup for all reproduction scripts
# Source this at the start of each script:
#   source "$(dirname "$0")/common.sh"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Run dataset setup (idempotent — skips already-downloaded)
echo "[setup] Checking datasets..."
python setup_datasets.py 2>&1 | tail -5

# Auto-detect batch size based on GPU memory
if [ -z "$BATCH_SIZE" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        if [ "$GPU_MEM" -ge 15000 ]; then BATCH_SIZE=64
        elif [ "$GPU_MEM" -ge 7000 ]; then BATCH_SIZE=32
        elif [ "$GPU_MEM" -ge 5000 ]; then BATCH_SIZE=16
        else BATCH_SIZE=8
        fi
        echo "[auto] GPU memory: ${GPU_MEM}MB → BATCH_SIZE=$BATCH_SIZE"
    else
        BATCH_SIZE=32
        echo "[auto] No nvidia-smi → BATCH_SIZE=32"
    fi
fi
