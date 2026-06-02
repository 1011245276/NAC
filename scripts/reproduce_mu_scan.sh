#!/bin/bash
# ============================================================
# NAC: Momentum Coefficient Scan (Table in paper, Figure 3)
# NAC with mu from 0 to 0.99, CIFAR-10, eps=4/255
# Usage: bash scripts/reproduce_mu_scan.sh
# ============================================================
set -e

DATASET="cifar10"
MU_VALUES=(0 0.1 0.5 0.7 0.9 0.99)
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: Momentum Coefficient Scan"
echo " mu = 0 (TTC) to 0.99, PGD eps=4/255"
echo "============================================"

for mu in "${MU_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [ "$mu" = "0" ]; then
      # mu=0 → equivalent to TTC
      echo "[$(date '+%H:%M:%S')] mu=$mu (TTC baseline) seed=$seed"
      python nac_fair_experiment.py \
        --batch_size 32 --root ./data \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
        --ttc_numsteps 2 --counterattack ttc \
        --seed $seed --outdir ./results/mu_scan
    else
      echo "[$(date '+%H:%M:%S')] mu=$mu seed=$seed"
      python nac_fair_experiment.py \
        --batch_size 32 --root ./data \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
        --ttc_numsteps 2 --counterattack nac --nac_momentum $mu \
        --seed $seed --outdir ./results/mu_scan
    fi
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/mu_scan/"
