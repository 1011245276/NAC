#!/bin/bash
# ============================================================
# NAC vs TTC vs DOC — Fair comparison (Config A: K=2, step=1.0)
# All three methods use identical evaluation pipeline.
# Run from nac_project/ directory.
#
# NOTE: DOC uses its own internal parameters (learnable_tau=0.155,
# temperature=75.0) via nac_fair_experiment.py defaults. The
# --tau_thres flag is passed for TTC/NAC but ignored by DOC.
#
# For Config B (K=4, step=3.0, DOC paper defaults), run:
#   python nac_fair_experiment.py --counterattack doc --ttc_numsteps 4
#       --ttc_stepsize 3.0 --learnable_tau 0.155 --temperature 75.0 ...
# ============================================================
set -e

OUTDIR="./results/compare"
SEED=${1:-0}
BATCH=32

echo "========== Fair Comparison: TTC vs NAC vs DOC =========="
echo "Config A: PGD eps=4/255 attack, 2-step defense, K=2, step=1/255"
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
echo "For Config B (DOC paper defaults: K=4, step=3.0), edit --ttc_numsteps 4 --ttc_stepsize 3.0"
