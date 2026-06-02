#!/bin/bash
# Full experiments for NAC paper
# Uses TTC code infrastructure for dataset loading

TTC_DIR="/f/code/CLIP/03_测试时防御_Test-time_Defense/TTC/TTC_source/CLIP-Test-time-Counterattacks-main/code"
RESULT_DIR="/f/code/nac_project/results"
DATA_ROOT="/f/code/data"
BATCH_SIZE=64
SEED=0

mkdir -p "$RESULT_DIR"

echo "============================================"
echo " NAC Paper: Full Experiments"
echo " Datasets: cifar10 cifar100 STL10 flowers102 dtd eurosat sun397"
echo "============================================"

DATASETS="cifar10 cifar100 STL10 flowers102 dtd eurosat sun397"

# ==========================================
# Experiment 1: eps=1/255, 10-step PGD
# ==========================================
echo ""
echo "========== Exp 1: PGD eps=1/255, 10 steps =========="

for METHOD in ttc nac; do
    EXTRA=""
    TAG="$METHOD"
    if [ "$METHOD" = "nac" ]; then
        EXTRA="--nac_momentum 0.9"
        TAG="nac_m0.9"
    fi

    echo "--- $TAG ---"
    cd "$TTC_DIR" && /f/anaconda3/python nac_fair_experiment.py \
        --batch_size $BATCH_SIZE --num_workers 4 --root "$DATA_ROOT" \
        --test_set $DATASETS \
        --test_attack_type pgd --test_eps 1 --test_numsteps 10 --test_stepsize 1 \
        --ttc_eps 4 --ttc_numsteps 2 --ttc_stepsize 1 \
        --tau_thres 0.2 --beta 2 --seed $SEED \
        --counterattack $METHOD $EXTRA \
        --outdir "$RESULT_DIR/eps1" 2>&1 | grep -E "cifar|STL|dtd|eurosat|flowers|sun397|SUMMARY"
done

# ==========================================
# Experiment 2: eps=2/255, 10-step PGD
# ==========================================
echo ""
echo "========== Exp 2: PGD eps=2/255, 10 steps =========="

for METHOD in ttc nac; do
    EXTRA=""
    TAG="$METHOD"
    if [ "$METHOD" = "nac" ]; then
        EXTRA="--nac_momentum 0.9"
        TAG="nac_m0.9"
    fi

    echo "--- $TAG ---"
    cd "$TTC_DIR" && /f/anaconda3/python nac_fair_experiment.py \
        --batch_size $BATCH_SIZE --num_workers 4 --root "$DATA_ROOT" \
        --test_set $DATASETS \
        --test_attack_type pgd --test_eps 2 --test_numsteps 10 --test_stepsize 1 \
        --ttc_eps 4 --ttc_numsteps 2 --ttc_stepsize 1 \
        --tau_thres 0.2 --beta 2 --seed $SEED \
        --counterattack $METHOD $EXTRA \
        --outdir "$RESULT_DIR/eps2" 2>&1 | grep -E "cifar|STL|dtd|eurosat|flowers|sun397|SUMMARY"
done

# ==========================================
# Experiment 3: eps=4/255, 10-step PGD
# ==========================================
echo ""
echo "========== Exp 3: PGD eps=4/255, 10 steps =========="

for METHOD in ttc nac; do
    EXTRA=""
    TAG="$METHOD"
    if [ "$METHOD" = "nac" ]; then
        EXTRA="--nac_momentum 0.9"
        TAG="nac_m0.9"
    fi

    echo "--- $TAG ---"
    cd "$TTC_DIR" && /f/anaconda3/python nac_fair_experiment.py \
        --batch_size $BATCH_SIZE --num_workers 4 --root "$DATA_ROOT" \
        --test_set $DATASETS \
        --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
        --ttc_eps 4 --ttc_numsteps 2 --ttc_stepsize 1 \
        --tau_thres 0.2 --beta 2 --seed $SEED \
        --counterattack $METHOD $EXTRA \
        --outdir "$RESULT_DIR/eps4" 2>&1 | grep -E "cifar|STL|dtd|eurosat|flowers|sun397|SUMMARY"
done

echo ""
echo "========== ALL DONE =========="
