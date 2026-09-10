#!/bin/bash
# ============================================================
# NAC 一键运行全部论文复现实验
#
# 用法：
#   bash run_all.sh                 # 默认完整版：所有脚本，SEEDS="0 1 2"
#   SEEDS=0 bash run_all.sh         # 单 seed 快速版，适合先跑通流程
#   bash run_all.sh --skip-aft      # 跳过 AFT（需要 TeCoA/FARE 外部权重）
#   BATCH_SIZE=8 bash run_all.sh    # 显存小时全局降低 batch size
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_AFT=1
if [[ "${1:-}" == "--skip-aft" ]]; then
  RUN_AFT=0
  shift
fi

export SEEDS="${SEEDS:-0 1 2}"
export BATCH_SIZE="${BATCH_SIZE:-}"

LOG_DIR="results"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all.log"
: > "$LOG_FILE"

log() {
  local msg="[run_all $(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

log "START seeds=[$SEEDS] batch_size=[${BATCH_SIZE:-auto}] skip_aft=$([ "$RUN_AFT" = 1 ] && echo no || echo yes)"

STEPS=(
  "scripts/reproduce_main.sh|Table 1 主结果"
  "scripts/reproduce_multi_eps.sh|Table 2 多攻击强度"
  "scripts/reproduce_nstep.sh|Table 3 N-step"
  "scripts/reproduce_autoattack.sh|Table 4 AutoAttack"
  "scripts/reproduce_cross_arch.sh|Table 5 跨架构"
  "scripts/reproduce_ablation.sh|Table 7 消融"
  "scripts/reproduce_mu_scan.sh|μ 扫描"
  "scripts/reproduce_aft.sh|Table 6 AFT"
  "scripts/compare_ttc_nac_doc.sh|DOC 公平对比"
)

for item in "${STEPS[@]}"; do
  script="${item%%|*}"
  desc="${item#*|}"

  if [[ "$script" == "scripts/reproduce_aft.sh" && "$RUN_AFT" == "0" ]]; then
    log "SKIP $desc (--skip-aft)"
    continue
  fi

  if [[ "$script" == "scripts/compare_ttc_nac_doc.sh" ]]; then
    for seed in $SEEDS; do
      log "RUN $desc (seed=$seed) -> $script"
      bash "$script" "$seed" 2>&1 | tee -a "$LOG_FILE"
      log "DONE $desc (seed=$seed)"
    done
    continue
  fi

  log "RUN $desc -> $script"
  bash "$script" 2>&1 | tee -a "$LOG_FILE"
  log "DONE $desc"
done

log "ALL DONE"