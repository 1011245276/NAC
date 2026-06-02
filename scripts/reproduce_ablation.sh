#!/bin/bash
# ============================================================
# NAC: Reproduce Ablation Study (Table 7 in paper)
# TTC vs Standard Momentum vs NAC, PGD eps=4/255
# Usage: bash scripts/reproduce_ablation.sh
# ============================================================
set -e

DATASETS=("cifar10" "STL10")
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: Ablation Study (Table 7)"
echo " TTC vs Standard Momentum vs NAC"
echo " PGD eps=4/255, 2-step defense"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
  # TTC (no momentum)
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] TTC dataset=$dataset seed=$seed"
    python nac_fair_experiment.py \
      --batch_size 32 --root ./data \
      --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
      --test_set $dataset --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack ttc --seed $seed --outdir ./results/ablation
  done

  # Standard momentum
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Momentum dataset=$dataset seed=$seed"
    python nac_fair_experiment.py \
      --batch_size 32 --root ./data \
      --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
      --test_set $dataset --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack momentum --nac_momentum 0.9 --seed $seed --outdir ./results/ablation
  done

  # NAC (Nesterov look-ahead)
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] NAC dataset=$dataset seed=$seed"
    python nac_fair_experiment.py \
      --batch_size 32 --root ./data \
      --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
      --test_set $dataset --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack nac --nac_momentum 0.9 --seed $seed --outdir ./results/ablation
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/ablation/"
