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
echo " AFT checkpoints auto-download if missing."
echo "============================================"

# Auto-download TeCoA checkpoint if missing
TECOA_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/92req4Pak5i56tX/download/tecoa_eps_4.pt"
TECOA_PATH="./checkpoints/tecoa_vit_b32.pth"
if [ ! -f "$TECOA_PATH" ]; then
  echo "[*] Downloading TeCoA checkpoint (eps=4, ViT-B/32) ..."
  mkdir -p ./checkpoints
  if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$TECOA_PATH" "$TECOA_URL" || {
      echo "[!] wget failed, trying curl ..."
      curl -L -o "$TECOA_PATH" "$TECOA_URL"
    }
  elif command -v curl &> /dev/null; then
    curl -L -o "$TECOA_PATH" "$TECOA_URL"
  else
    echo "[!] Neither wget nor curl found. Download manually from:"
    echo "    $TECOA_URL"
    echo "    -> save as $TECOA_PATH"
  fi
  [ -f "$TECOA_PATH" ] && echo "[OK] TeCoA checkpoint downloaded." || echo "[!] TeCoA download failed."
fi

# Auto-download FARE checkpoint if missing
FARE_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/jnQ2qmp9tst8kyQ/download/fare_eps_4.pt"
FARE_PATH="./checkpoints/fare_vit_b32.pth"
if [ ! -f "$FARE_PATH" ]; then
  echo "[*] Downloading FARE checkpoint (eps=4, ViT-B/32) ..."
  mkdir -p ./checkpoints
  if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$FARE_PATH" "$FARE_URL" || {
      echo "[!] wget failed, trying curl ..."
      curl -L -o "$FARE_PATH" "$FARE_URL"
    }
  elif command -v curl &> /dev/null; then
    curl -L -o "$FARE_PATH" "$FARE_URL"
  else
    echo "[!] Neither wget nor curl found. Download manually from:"
    echo "    $FARE_URL"
    echo "    -> save as $FARE_PATH"
  fi
  [ -f "$FARE_PATH" ] && echo "[OK] FARE checkpoint downloaded." || echo "[!] FARE download failed."
fi

# Check if AFT checkpoints exist after download attempt
if [ ! -f "$TECOA_PATH" ] && [ ! -f "$FARE_PATH" ]; then
  echo ""
  echo "WARNING: No AFT checkpoints available in ./checkpoints/"
  echo "Skipping AFT superposition experiment."
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
