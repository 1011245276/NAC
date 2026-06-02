#!/bin/bash
# ============================================================
# NAC: Reproduce Main Results (Table 1 in paper)
# PGD eps=1/255, 2-step defense, 6 datasets
# Usage: bash scripts/reproduce_main.sh
# ============================================================
set -e

METHODS=("ttc" "nac")
DATASETS=("cifar10" "cifar100" "STL10" "flowers102" "DTD" "imagenet100")
EPS=1
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: Reproduce Main Results (Table 1)"
echo " PGD eps=$EPS/255, 2-step defense"
echo " 2 methods x 6 datasets x 3 seeds = 36 runs"
echo "============================================"

for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "[$(date '+%H:%M:%S')] method=$method dataset=$dataset seed=$seed"
      python nac_fair_experiment.py \
        --batch_size ${BATCH_SIZE:-32} \
        --root ./data \
        --test_attack_type pgd \
        --test_eps $EPS \
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
        --outdir ./results/main
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/main/"
