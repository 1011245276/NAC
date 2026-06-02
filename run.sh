#!/bin/bash
# NAC: Nesterov Accelerated Counterattack
# Usage: bash run.sh [method] [dataset] [eps]

METHOD=${1:-nac}
DATASET=${2:-cifar10}
EPS=${3:-4}

echo "NAC: method=$METHOD dataset=$DATASET eps=$EPS/255"

python nac_fair_experiment.py \
    --batch_size ${BATCH_SIZE:-32} \
    --root ./data \
    --test_attack_type pgd \
    --test_eps $EPS \
    --test_numsteps 10 \
    --test_stepsize 1 \
    --test_set $DATASET \
    --ttc_eps 4 \
    --beta 2 \
    --tau_thres 0.2 \
    --ttc_numsteps 2 \
    --counterattack $METHOD \
    --nac_momentum 0.9 \
    --seed 0 \
    --outdir ./results
