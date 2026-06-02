#!/bin/bash
# Main experiment: TTC vs NAC on multiple datasets
# Reproduces Table 1 from the paper.

DATASETS="cifar10 cifar100 STL10"
STEPS=2
EPS=4

echo "============================================"
echo " NAC vs TTC: Main Comparison"
echo " PGD eps=${EPS}/255, ${STEPS}-step counterattack"
echo "============================================"

echo ""
echo "--- TTC (Baseline) ---"
python evaluate.py \
    --datasets $DATASETS \
    --method ttc \
    --attack_eps $EPS --attack_steps 5 \
    --defense_eps $EPS --defense_steps $STEPS \
    --seed 0

echo ""
echo "--- NAC (Ours, m=0.9) ---"
python evaluate.py \
    --datasets $DATASETS \
    --method nac --nac_momentum 0.9 \
    --attack_eps $EPS --attack_steps 5 \
    --defense_eps $EPS --defense_steps $STEPS \
    --seed 0

echo ""
echo "============================================"
echo "Done."
echo "============================================"
