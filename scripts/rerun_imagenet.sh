#!/bin/bash
# ============================================================
# Re-run ImageNet-100 only (Table 1 + Table 2), after fixing the
# support-path bug in get_text_prompts_val (was using WordNet IDs
# instead of clean class names -> clean ~1% instead of ~67%).
#
# Usage: bash scripts/rerun_imagenet.sh
#        SEEDS="0 1 2" bash scripts/rerun_imagenet.sh   # multi-seed
# ============================================================
set -e

source "$(dirname "$0")/common.sh"

METHODS=("ttc" "nac")
SEEDS=(${SEEDS:-0})

echo "============================================"
echo " Re-run ImageNet-100 (Table 1 + Table 2)"
echo " Methods: ${METHODS[@]}  Seeds: ${SEEDS[@]}"
echo "============================================"

# ---- Table 1: main results, eps=1/255 ----
for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Table1 eps=1 method=$method seed=$seed"
    python nac_fair_experiment.py \
      --batch_size ${BATCH_SIZE:-32} --root ./data \
      --test_attack_type pgd --test_eps 1 --test_numsteps 10 --test_stepsize 1 \
      --test_set ImageNet --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
      --counterattack $method --nac_momentum 0.9 --seed $seed \
      --outdir ./results/imagenet_full --num_seeds 1
  done
done

# ---- Table 2: multi-eps, eps=1/2/4 (ImageNet seed 0 only per paper) ----
for eps in 1 2 4; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ "$seed" != "0" ]; then
        continue   # ImageNet-100: primary seed only (per paper)
      fi
      echo "[$(date '+%H:%M:%S')] Table2 eps=$eps method=$method seed=$seed"
      python nac_fair_experiment.py \
        --batch_size ${BATCH_SIZE:-32} --root ./data \
        --test_attack_type pgd --test_eps $eps --test_numsteps 10 --test_stepsize 1 \
        --test_set ImageNet --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps 2 \
        --counterattack $method --nac_momentum 0.9 --seed $seed \
        --outdir ./results/multi_eps --num_seeds 1
    done
  done
done

echo "[$(date '+%H:%M:%S')] Done. ImageNet-100 re-run complete."
echo "Aggregate with: python scripts/aggregate_all.py"
