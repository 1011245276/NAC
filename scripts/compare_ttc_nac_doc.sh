#!/bin/bash
# ============================================================
# NAC vs TTC vs DOC — Fair comparison
# All three methods use identical evaluation pipeline.
# Run from nac_project/ directory.
# ============================================================
set -e

OUTDIR="./results/compare"
SEED=0
BATCH=32

echo "========== Fair Comparison: TTC vs NAC vs DOC =========="
echo "Settings: PGD eps=4/255 attack, 2-step defense, CIFAR-10"
echo "Output: $OUTDIR"
echo ""

for METHOD in ttc nac doc; do
    echo "--- Running $METHOD ---"
    python nac_fair_experiment.py \
        --counterattack "$METHOD" \
        --test_set cifar10 \
        --test_eps 4 \
        --test_numsteps 10 \
        --ttc_numsteps 2 \
        --ttc_eps 4 \
        --ttc_stepsize 1 \
        --tau_thres 0.2 \
        --beta 2.0 \
        --nac_momentum 0.9 \
        --batch_size "$BATCH" \
        --seed "$SEED" \
        --outdir "$OUTDIR"
    echo ""
done

echo "========== Done =========="
echo "Results saved to $OUTDIR/"
echo ""
echo "For STL-10 comparison, run:"
echo "  bash scripts/compare_all.sh stl10"
