#!/bin/bash
# Kill zombies, run all experiments, AutoAttack last
LOGDIR="/f/code/nac_project/results/final"
mkdir -p "$LOGDIR"
DOC_DIR="/f/code/CLIP/03_测试时防御_Test-time_Defense/DOC/DOC_source/DOC-main/Code"
TTC_DIR="/f/code/CLIP/03_测试时防御_Test-time_Defense/TTC/TTC_source/CLIP-Test-time-Counterattacks-main/code"

cleanup() {
    for pid in $(ps aux 2>/dev/null | grep python | grep -v grep | awk '{print $1}'); do
        kill -9 $pid 2>/dev/null
    done
    sleep 3
}

# Wait for current DOC-2 to finish, then continue
echo "Waiting for DOC-2..."
wait
echo "DOC-2 done, starting remaining experiments..."

cleanup

# DOC-4
cd "$DOC_DIR"
echo "=== DOC-4 $(date) ==="
/f/anaconda3/python DOC.py --batch_size 64 --num_workers 4 --root /f/code/data \
    --test_set cifar10 STL10 --test_attack_type pgd --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --DOC_eps 4 --DOC_numsteps 4 --DOC_stepsize 1 --tau_thres 0.2 --beta 2 --seed 0 \
    --outdir "$LOGDIR" > "$LOGDIR/doc4.txt" 2>&1
echo "DOC-4 done"
cleanup

# TTC-4
cd "$TTC_DIR"
echo "=== TTC-4 $(date) ==="
/f/anaconda3/python nac_fair_experiment.py --batch_size 64 --num_workers 4 --root /f/code/data \
    --test_set cifar10 STL10 --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --ttc_eps 4 --ttc_numsteps 4 --ttc_stepsize 1 --tau_thres 0.2 --beta 2 --seed 0 \
    --counterattack ttc --outdir "$LOGDIR" > "$LOGDIR/ttc4.txt" 2>&1
echo "TTC-4 done"
cleanup

# NAC-4
echo "=== NAC-4 $(date) ==="
/f/anaconda3/python nac_fair_experiment.py --batch_size 64 --num_workers 4 --root /f/code/data \
    --test_set cifar10 STL10 --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --ttc_eps 4 --ttc_numsteps 4 --ttc_stepsize 1 --tau_thres 0.2 --beta 2 --seed 0 \
    --counterattack nac --nac_momentum 0.9 --outdir "$LOGDIR" > "$LOGDIR/nac4.txt" 2>&1
echo "NAC-4 done"
cleanup

# AutoAttack TTC
echo "=== AutoAttack TTC $(date) ==="
/f/anaconda3/python nac_fair_experiment.py --batch_size 16 --num_workers 2 --root /f/code/data \
    --test_set cifar10 --test_attack_type autoattack --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --ttc_eps 4 --ttc_numsteps 2 --ttc_stepsize 1 --tau_thres 0.2 --beta 2 --seed 0 \
    --counterattack ttc --outdir /f/code/nac_project/results/aa > /f/code/nac_project/results/aa/ttc_final.log 2>&1
echo "AutoAttack TTC done"
cleanup

# AutoAttack NAC
echo "=== AutoAttack NAC $(date) ==="
/f/anaconda3/python nac_fair_experiment.py --batch_size 16 --num_workers 2 --root /f/code/data \
    --test_set cifar10 --test_attack_type autoattack --test_eps 4 --test_numsteps 10 --test_stepsize 1 \
    --ttc_eps 4 --ttc_numsteps 2 --ttc_stepsize 1 --tau_thres 0.2 --beta 2 --seed 0 \
    --counterattack nac --nac_momentum 0.9 --outdir /f/code/nac_project/results/aa > /f/code/nac_project/results/aa/nac_final.log 2>&1
echo "AutoAttack NAC done"

echo "=== ALL EXPERIMENTS COMPLETE $(date) ==="
