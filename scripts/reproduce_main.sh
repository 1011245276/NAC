#!/bin/bash
# ============================================================
# NAC: Reproduce Main Results (Table 1 in paper)
# PGD eps=1/255, 2-step defense, 6 datasets.
#
# 5 datasets (CIFAR-10/100, STL-10, Flowers-102, DTD) run in ONE
# invocation per method at bs=64 -> results/eps1/  so the nested
# seed_*.log contains all five (avoids filemode='w' overwrite).
# ImageNet-100 runs separately at bs=32 -> results/imagenet_full/.
#
# Usage: bash scripts/reproduce_main.sh
# Multi-seed: SEEDS="0 1 2" bash scripts/reproduce_main.sh
# ============================================================
set -e

source "$(dirname "$0")/common.sh"

METHODS=("ttc" "nac")
FIVE=("cifar10" "cifar100" "STL10" "flowers102" "DTD")
EPS=1
SEEDS=(${SEEDS:-0})
BATCH_FIVE=${BATCH_FIVE:-64}
BATCH_IN=${BATCH_IN:-32}

echo "============================================"
echo " NAC: Reproduce Main Results (Table 1)"
echo " PGD eps=$EPS/255, 2-step defense"
echo " Methods: ${METHODS[@]}   Seeds: ${SEEDS[@]}"
echo " 5 datasets bs=$BATCH_FIVE -> results/eps1/"
echo " ImageNet   bs=$BATCH_IN  -> results/imagenet_full/"
echo "============================================"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "[$(date '+%H:%M:%S')] 5-datasets method=$method seed=$seed bs=$BATCH_FIVE"
    python nac_fair_experiment.py \
      --batch_size "$BATCH_FIVE" --root ./data --num_workers "${NUM_WORKERS:-0}" \
      --test_attack_type pgd --test_eps "$EPS" --test_numsteps 10 --test_stepsize 1 \
      --test_set "${FIVE[@]}" \
      --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack "$method" --nac_momentum 0.9 --seed "$seed" \
      --outdir ./results/eps1 --num_seeds 1
  done
  for method in "${METHODS[@]}"; do
    echo "[$(date '+%H:%M:%S')] ImageNet method=$method seed=$seed bs=$BATCH_IN"
    python nac_fair_experiment.py \
      --batch_size "$BATCH_IN" --root ./data --num_workers "${NUM_WORKERS:-0}" \
      --test_attack_type pgd --test_eps "$EPS" --test_numsteps 10 --test_stepsize 1 \
      --test_set ImageNet \
      --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack "$method" --nac_momentum 0.9 --seed "$seed" \
      --outdir ./results/imagenet_full --num_seeds 1
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results in results/eps1/ and results/imagenet_full/"
echo "Aggregate with: python scripts/aggregate_all.py"
