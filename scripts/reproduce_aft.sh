#!/bin/bash
# ============================================================
# NAC: AFT Superposition (Table 6 in paper)
# NAC on top of TeCoA / FARE adversarially fine-tuned models
# Prerequisites: download TeCoA and FARE checkpoints first
# Usage: bash scripts/reproduce_aft.sh
# ============================================================
set -e

# Source common setup (dataset check + GPU auto-detect)
source "$(dirname "$0")/common.sh"

DATASET="cifar10"
METHODS=("ttc" "nac")
SEEDS=(0 1 2)

echo "============================================"
echo " NAC: AFT Superposition (Table 6)"
echo " TeCoA + FARE, PGD eps=4/255"
echo ""
echo " NOTE: Requires TeCoA and FARE model checkpoints."
echo " Download from their official repos and place in ./checkpoints/"
echo "   TeCoA eps=4: https://nc.mlcloud.uni-tuebingen.de/index.php/s/92req4Pak5i56tX/download/tecoa_eps_4.pt"
echo "   FARE  eps=4: https://nc.mlcloud.uni-tuebingen.de/index.php/s/jnQ2qmp9tst8kyQ/download/fare_eps_4.pt"
echo "   If unavailable, script skips AFT experiments gracefully."
echo "============================================"

# Check if AFT checkpoints exist
if [ ! -f "./checkpoints/tecoa_vit_b32.pth" ] && [ ! -f "./checkpoints/fare_vit_b32.pth" ]; then
  echo ""
  echo "WARNING: No AFT checkpoints found in ./checkpoints/"
  echo "Skipping AFT superposition experiment."
  echo "Download TeCoA/FARE weights and re-run this script."
  exit 0
fi

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # TeCoA
    if [ -f "./checkpoints/tecoa_vit_b32.pth" ]; then
      echo "[$(date '+%H:%M:%S')] TeCoA method=$method seed=$seed"
      python nac_fair_experiment.py \
        --batch_size ${BATCH_SIZE:-32} --root ./data \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
        --ttc_numsteps 2 --counterattack $method --nac_momentum 0.9 \
        --victim_resume ./checkpoints/tecoa_vit_b32.pth \
        --seed $seed --outdir ./results/aft
    fi

    # FARE
    if [ -f "./checkpoints/fare_vit_b32.pth" ]; then
      echo "[$(date '+%H:%M:%S')] FARE method=$method seed=$seed"
      python nac_fair_experiment.py \
        --batch_size ${BATCH_SIZE:-32} --root ./data \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --test_set $DATASET --ttc_eps 4 --beta 2 --tau_thres 0.2 \
        --ttc_numsteps 2 --counterattack $method --nac_momentum 0.9 \
        --victim_resume ./checkpoints/fare_vit_b32.pth \
        --seed $seed --outdir ./results/aft
    fi
  done
done

echo "[$(date '+%H:%M:%S')] Done! Results saved to ./results/aft/"
