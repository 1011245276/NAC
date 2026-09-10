#!/bin/bash
# ============================================================
# Re-run Tables 2/3/5 (eps=2/4) with the current code so their
# clean values match the re-run Table 1 (clean=85.07 etc.).
#
#   Table 2 (multi-eps):    eps=2,4  ViT-B/32 K=2  5 datasets -> results/eps2, eps4
#   Table 2 (ImageNet):     eps=2,4  ViT-B/32 K=2  ImageNet   -> results/multi_eps
#   Table 3 (n-step K=4):   eps=4    ViT-B/32 K=4  cifar10+STL10 -> results/final
#   Table 5 (ViT-B/16):     eps=4    ViT-B/16 K=2  cifar10+STL10 -> results/cross_model
#   Table 5 (RN50):         eps=4    RN50    K=2  cifar10      -> results/rn50
#
# num_workers=0 to avoid DataLoader worker-spawn memory hangs on
# low-RAM machines. Run in one invocation per (eps, method) so the
# nested seed_0.log holds all datasets (filemode='w' overwrite).
# ============================================================
set -e

ROOT=./data
NW=${NUM_WORKERS:-0}

run() {
  local method=$1; local batch=$2; local outdir=$3; local eps=$4; local steps=$5; local arch=$6; shift 6
  echo "[$(date '+%H:%M:%S')] eps=$eps steps=$steps arch=$arch method=$method batch=$batch ds=$*"
  python nac_fair_experiment.py \
    --batch_size "$batch" --root "$ROOT" --num_workers "$NW" \
    --arch "$arch" \
    --test_attack_type pgd --test_eps "$eps" --test_numsteps 10 --test_stepsize 1 \
    --test_set "$@" \
    --ttc_eps 4 --beta 2 --tau_thres 0.2 --ttc_numsteps "$steps" \
    --counterattack "$method" --nac_momentum 0.9 --seed 0 \
    --outdir "$outdir" --num_seeds 1
}

FIVE=(cifar10 cifar100 STL10 flowers102 DTD)
TWO=(cifar10 STL10)

echo "=== Table 2: eps=2 (5 datasets, ViT-B/32, K=2) ==="
for m in ttc nac; do run "$m" 64 ./results/eps2 2 2 vit_b32 "${FIVE[@]}"; done

echo "=== Table 2: eps=4 (5 datasets, ViT-B/32, K=2) ==="
for m in ttc nac; do run "$m" 64 ./results/eps4 4 2 vit_b32 "${FIVE[@]}"; done

echo "=== Table 2: ImageNet eps=2,4 (bs=32) ==="
for e in 2 4; do for m in ttc nac; do run "$m" 32 ./results/multi_eps "$e" 2 vit_b32 ImageNet; done; done

echo "=== Table 3: K=4 (cifar10+STL10, ViT-B/32, eps=4) ==="
for m in ttc nac; do run "$m" 32 ./results/final 4 4 vit_b32 "${TWO[@]}"; done

echo "=== Table 5: ViT-B/16 (cifar10+STL10, eps=4, K=2) ==="
for m in ttc nac; do run "$m" 32 ./results/cross_model 4 2 vit_b16 "${TWO[@]}"; done

echo "=== Table 5: RN50 (cifar10, eps=4, K=2) ==="
for m in ttc nac; do run "$m" 16 ./results/rn50 4 2 RN50 cifar10; done

echo "[$(date '+%H:%M:%S')] DONE Tables 2/3/5 re-run."
