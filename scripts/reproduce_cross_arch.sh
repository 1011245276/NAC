#!/bin/bash
# ============================================================
# NAC: Cross-Architecture (Table 5 in paper)
# TTC vs NAC on ViT-B/32 and ViT-B/16, CIFAR-10, eps=4/255
# Usage: bash scripts/reproduce_cross_arch.sh
# ============================================================
set -e

DATASET="cifar10"
ARCHS=("vit_b32" "vit_b16")
METHODS=("ttc" "nac")
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: Cross-Architecture (Table 5)"
echo " ViT-B/32 + ViT-B/16, PGD eps=4/255"
echo "============================================"

for arch in "${ARCHS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "[$(date '+%H:%M:%S')] arch=$arch method=$method seed=$seed"
      python nac_fair_experiment.py \
        --batch_size ${BATCH_SIZE:-32} --root ./data --arch $arch \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
        --ttc_numsteps 2 --counterattack $method --nac_momentum 0.9 \
        --seed $seed --outdir ./results/cross_arch
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/cross_arch/"
