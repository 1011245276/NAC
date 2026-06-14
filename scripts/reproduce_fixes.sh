#!/bin/bash
# ============================================================
# NAC 必要重跑实验 �?修复消融数据完整�?# 预计耗时: ~4h (RTX 4060)
# ============================================================
set -e
cd F:/codex/NAC/nac_project

# ---- 1. Standard momentum ablation (当前无有效数�? ----
echo "=== [1/4] Standard momentum ablation ==="
python nac_fair_experiment.py \
    --counterattack momentum \
    --test_set cifar10 STL10 \
    --test_eps 4 --test_numsteps 10 \
    --ttc_numsteps 2 --ttc_eps 4 \
    --outdir ./results/ablation_fixed \
    --seed 0 --batch_size 32 \
    --nac_momentum 0.9

# ---- 2. DOC with original paper defaults (tau=0.3, T=20) ----
echo "=== [2/4] DOC with original paper defaults ==="
python nac_fair_experiment.py \
    --counterattack doc \
    --test_set cifar10 \
    --test_eps 4 --test_numsteps 10 \
    --ttc_numsteps 4 --ttc_stepsize 3 --ttc_eps 4 \
    --learnable_tau 0.3 --temperature 20.0 \
    --outdir ./results/compare_doc_orig \
    --seed 0 --batch_size 32

# ---- 3. Multi-seed for remaining 4 datasets (eps=1) ----
echo "=== [3/4] Multi-seed NAC eps=1 ==="
for seed in 0 1 2; do
    echo "--- Seed $seed ---"
    python nac_fair_experiment.py \
        --counterattack nac \
        --test_set cifar100 STL10 flowers102 ImageNet \
        --test_eps 1 --test_numsteps 10 \
        --ttc_numsteps 2 --ttc_eps 4 \
        --outdir ./results/multiseed_all \
        --seed $seed --batch_size 64
done

# ---- 4. L-BFGS baseline ----
echo "=== [4/4] L-BFGS baseline ==="
python nac_fair_experiment.py \
    --counterattack lbfgs \
    --test_set cifar10 \
    --test_eps 4 --test_numsteps 10 \
    --ttc_numsteps 4 --ttc_eps 4 \
    --outdir ./results/optimizers \
    --seed 0 --batch_size 16

echo "=== ALL DONE ==="
echo "Check results in:"
echo "  results/ablation_fixed/pgd_eps_4.0/momentum/seed_0.log"
echo "  results/compare_doc_orig/pgd_eps_4.0/doc/seed_0.log"
echo "  results/multiseed_all/pgd_eps_1.0/nac_m0.9/seed_{0,1,2}.log"
echo "  results/optimizers/pgd_eps_4.0/lbfgs/seed_0.log"