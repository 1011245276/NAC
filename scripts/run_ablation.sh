#!/bin/bash
# Ablation study: momentum coefficient sweep
# Reproduces Figure X from the paper.

echo "============================================"
echo " NAC Momentum Ablation"
echo " PGD eps=4/255, 2-step counterattack"
echo "============================================"

for MOM in 0.0 0.1 0.3 0.5 0.7 0.9 0.99; do
    echo ""
    echo "--- m = $MOM ---"
    python evaluate.py \
        --datasets cifar10 \
        --method nac --nac_momentum $MOM \
        --attack_eps 4 --attack_steps 5 \
        --defense_eps 4 --defense_steps 2 \
        --seed 0
done

echo ""
echo "Note: m=0.0 is equivalent to TTC baseline"
echo "Done."
