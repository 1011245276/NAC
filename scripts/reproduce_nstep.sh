#!/bin/bash
# ============================================================
# NAC: N-Step Scaling (Table 3 in paper)
# TTC vs NAC at 2-step and 4-step, PGD eps=4/255
# Usage: bash scripts/reproduce_nstep.sh
# ============================================================
set -e

# Source common setup (dataset check + GPU auto-detect)
source "$(dirname "$0")/common.sh"

DATASETS=("cifar10" "STL10")
STEPS=(2 4)
METHODS=("ttc" "nac")
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: N-Step Scaling (Table 3)"
echo " PGD eps=4/255, 2 datasets x 2 steps x 2 methods x 3 seeds"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
  for steps in "${STEPS[@]}"; do
    for method in "${METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] steps=$steps method=$method dataset=$dataset seed=$seed"
        python nac_fair_experiment.py \
          --batch_size ${BATCH_SIZE:-32} --root ./data \
          --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
          --test_set $dataset --ttc_eps 4 --beta 2 --tau_thres 0.2 \
          --ttc_numsteps $steps --counterattack $method --nac_momentum 0.9 \
          --seed $seed --outdir ./results/nstep
      done
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/nstep/"
