#!/bin/bash
# ============================================================
# Re-run Table 4 (AutoAttack) and Table 7 (ablation) with the
# current code for full consistency with re-run Tables 1/2/3/5.
#
#   Table 4 (AutoAttack):  cifar10, eps=4, K=2 -> results/aa
#   Table 7 (ablation):    cifar10+STL10, eps=4, K=2, bs=64
#                          ttc / momentum / pure_la / nac
#                          -> results/ablation
# ============================================================
set -e

ROOT=./data
NW=${NUM_WORKERS:-0}

echo "=== Table 4: AutoAttack (cifar10, eps=4, K=2) ==="
for m in ttc nac; do
  echo "[$(date '+%H:%M:%S')] autoattack method=$m"
  python nac_fair_experiment.py \
    --batch_size 32 --root "$ROOT" --num_workers "$NW" \
    --test_attack_type autoattack --test_eps 4 \
    --test_set cifar10 --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
    --counterattack "$m" --nac_momentum 0.9 --seed 0 \
    --outdir ./results/aa --num_seeds 1
done

echo "=== Table 7: ablation (cifar10+STL10, eps=4, K=2, bs=64) ==="
for m in ttc momentum pure_la nac; do
  echo "[$(date '+%H:%M:%S')] ablation method=$m"
  python nac_fair_experiment.py \
    --batch_size 64 --root "$ROOT" --num_workers "$NW" \
    --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --test_set cifar10 STL10 --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
    --counterattack "$m" --nac_momentum 0.9 --seed 0 \
    --outdir ./results/ablation --num_seeds 1
done

echo "[$(date '+%H:%M:%S')] DONE Tables 4/7 re-run."
