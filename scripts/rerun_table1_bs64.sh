#!/bin/bash
# ============================================================
# NAC: Re-run Table 1 primary cells (seed 0) at the paper's
# stated batch sizes: 5 datasets bs=64, ImageNet-100 bs=32.
#
# Writes 5 datasets to results/eps1/ and ImageNet-100 to
# results/imagenet_full/ (matching main_results.json _source).
# 5 datasets are run in ONE invocation per method so the nested
# seed_0.log contains all of them (avoids filemode='w' overwrite).
#
# Usage: bash scripts/rerun_table1_bs64.sh
# ============================================================
set -e

ROOT=./data
EPS=1

FIVE=("cifar10" "cifar100" "STL10" "flowers102" "DTD")

run() {
  local method=$1
  local batch=$2
  local outdir=$3
  shift 3
  echo "[$(date '+%H:%M:%S')] method=$method batch=$batch datasets=$*"
  python nac_fair_experiment.py \
    --batch_size "$batch" --root "$ROOT" \
    --num_workers "${NUM_WORKERS:-0}" \
    --test_attack_type pgd --test_eps "$EPS" --test_numsteps 10 --test_stepsize 1 \
    --test_set "$@" \
    --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
    --counterattack "$method" --nac_momentum 0.9 --seed 0 \
    --outdir "$outdir" --num_seeds 1
}

# 5 datasets, bs=64 -> results/eps1
for method in ttc nac; do
  run "$method" 64 ./results/eps1 "${FIVE[@]}"
done

# ImageNet-100, bs=32 -> results/imagenet_full
for method in ttc nac; do
  run "$method" 32 ./results/imagenet_full ImageNet
done

echo "[$(date '+%H:%M:%S')] Done. Table 1 re-run complete."
