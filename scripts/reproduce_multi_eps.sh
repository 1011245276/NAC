#!/bin/bash
# ============================================================
# NAC: Reproduce Multi-Epsilon Results (Table 2 in paper)
# PGD eps=1,2,4/255, 2-step defense, 6 datasets
# Usage: bash scripts/reproduce_multi_eps.sh
# ============================================================
set -e

# Source common setup (dataset check + GPU auto-detect)
source "$(dirname "$0")/common.sh"

METHODS=("ttc" "nac")
DATASETS=("cifar10" "cifar100" "STL10" "flowers102" "DTD" "imagenet100")
EPSILONS=(1 2 4)
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: Multi-Epsilon Results (Table 2)"
echo " 2 methods x 6 datasets x 3 epsilons x 3 seeds = 108 runs"
echo "============================================"

for eps in "${EPSILONS[@]}"; do
  for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] eps=$eps method=$method dataset=$dataset seed=$seed"
        python nac_fair_experiment.py \
          --batch_size ${BATCH_SIZE:-32} \
          --root ./data \
          --test_attack_type pgd \
          --test_eps $eps \
          --test_numsteps 10 \
          --test_stepsize 1 \
          --test_set $dataset \
          --ttc_eps 4 \
          --beta 2 \
          --tau_thres 0.2 \
          --ttc_numsteps 2 \
          --counterattack $method \
          --nac_momentum 0.9 \
          --seed $seed \
          --outdir ./results/multi_eps
      done
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/multi_eps/"
