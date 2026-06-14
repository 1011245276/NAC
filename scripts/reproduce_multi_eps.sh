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
DATASETS=("cifar10" "cifar100" "STL10" "flowers102" "DTD" "ImageNet")
EPSILONS=(1 2 4)
SEEDS=(0 1 2)
TOTAL_RUNS=$((2 * 6 * 3 * 2 + 2 * 5 * 3))  # 6 datasets (seeds 0-2) + 5 datasets for eps 2,4 (seed 0 only ImageNet)
echo "============================================"
echo " NAC: Multi-Epsilon Results (Table 2)"
echo " $TOTAL_RUNS runs (ImageNet-100: seed 0 only per paper)"
echo "============================================"

for eps in "${EPSILONS[@]}"; do
  for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        # ImageNet-100: primary seed only (per paper Reproducibility statement)
        if [ "$dataset" = "ImageNet" ] && [ "$seed" != "0" ]; then
          continue
        fi
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
