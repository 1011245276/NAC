#!/bin/bash
# ============================================================
# NAC: Reproduce Main Results (Table 1 in paper)
# PGD eps=1/255, 2-step defense, 6 datasets
# Usage: bash scripts/reproduce_main.sh
# Multi-seed mode: SEEDS="0 1 2" bash scripts/reproduce_main.sh
# ============================================================
set -e

# Source common setup (dataset check + GPU auto-detect)
source "$(dirname "$0")/common.sh"

METHODS=("ttc" "nac")
DATASETS=("cifar10" "cifar100" "STL10" "flowers102" "DTD" "ImageNet")
EPS=1
SEEDS=(${SEEDS:-0})  # default single seed for time efficiency

echo "============================================"
echo " NAC: Reproduce Main Results (Table 1)"
echo " PGD eps=$EPS/255, 2-step defense"
echo " Methods: ${METHODS[@]}"
echo " Datasets: ${DATASETS[@]}"
echo " Seeds: ${SEEDS[@]}"
echo " Total runs: $((${#METHODS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))"
echo "============================================"

mkdir -p ./results/eps1
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
        --outdir ./results/eps1 \
        --num_seeds 1 2>&1 | tee -a ./results/eps1/${method}_${dataset}_seed${seed}.log
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/eps1/"
echo "Aggregate with: python scripts/aggregate_results.py --root ./results/eps1"
