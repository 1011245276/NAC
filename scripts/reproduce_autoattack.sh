#!/bin/bash
# ============================================================
# NAC: AutoAttack Evaluation (Table 4 in paper)
# TTC vs NAC under AutoAttack, CIFAR-10, eps=4/255
# Usage: bash scripts/reproduce_autoattack.sh
# ============================================================
set -e

DATASET="cifar10"
METHODS=("ttc" "nac")
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: AutoAttack Evaluation (Table 4)"
echo " AutoAttack (APGD-CE + APGD-DLR), eps=4/255"
echo "============================================"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] method=$method dataset=$DATASET seed=$seed"
    python nac_fair_experiment.py \
      --batch_size ${BATCH_SIZE:-32} --root ./data \
      --test_attack_type autoattack --test_eps 4 \
      --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
      --ttc_numsteps 2 --counterattack $method --nac_momentum 0.9 \
      --seed $seed --outdir ./results/autoattack
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/autoattack/"
