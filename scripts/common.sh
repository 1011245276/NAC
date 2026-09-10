#!/bin/bash
# ============================================================
# NAC: Common setup for all reproduction scripts
# Source this at the start of each script:
#   source "$(dirname "$0")/common.sh"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============================================================
# Resolve a working Python interpreter. Priority:
#   1) $PYTHON_BIN (explicit override)
#   2) `python` on PATH that actually runs (skips the Windows Store alias stub)
#   3) known install locations — covers Git Bash (/f, /cygdrive/f, /c),
#      WSL (/mnt/f, /mnt/c) and native Windows paths
#   4) `python3` as a last resort
# ============================================================
resolve_python() {
  # 1) Explicit override from the environment
  if [ -n "${PYTHON_BIN:-}" ] && "${PYTHON_BIN}" -c "import sys" >/dev/null 2>&1; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  # 2) `python` on PATH that actually runs (skips the Windows Store alias stub)
  if command -v python >/dev/null 2>&1 && command python -c "import sys" >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  # 3) Known install locations (Git Bash / WSL / native Windows)
  for candidate in \
    "F:/anaconda3/python.exe" "/f/anaconda3/python.exe" "/cygdrive/f/anaconda3/python.exe" "/mnt/f/anaconda3/python.exe" \
    "F:/anaconda3/envs/base/python.exe" "/f/anaconda3/envs/base/python.exe" "/cygdrive/f/anaconda3/envs/base/python.exe" "/mnt/f/anaconda3/envs/base/python.exe" \
    "C:/Python312/python.exe" "/c/Python312/python.exe" "/mnt/c/Python312/python.exe" \
    "C:/Python311/python.exe" "/c/Python311/python.exe" "/mnt/c/Python311/python.exe" \
    "C:/Python310/python.exe" "/c/Python310/python.exe" "/mnt/c/Python310/python.exe"; do
    if "$candidate" -c "import sys" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  # 4) `python3` as a last resort
  if command -v python3 >/dev/null 2>&1 && command python3 -c "import sys" >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

if RESOLVED_PYTHON="$(resolve_python)"; then
  python() { "$RESOLVED_PYTHON" "$@"; }
  echo "[setup] Using Python: $RESOLVED_PYTHON"
else
  echo "[error] python not found. Activate a Python environment or set PYTHON_BIN." >&2
  exit 1
fi
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
