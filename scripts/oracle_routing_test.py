"""Oracle Solver Routing feasibility test — from existing final_verify.npz
(10000 samples, per-sample defended correctness for TTC/Momentum/PureLA/NAC
at K in {1,2,4,6,8}, seed 0 deterministic). NO new compute.

Defines Best Solver per sample (any solver that defends it; tie if >1;
none if all fail) at a FIXED budget K -> Oracle. Measures Oracle - NAC.
Also Best-Solver share, consistency of preference, and correlations with
available cheap features (adv-clean dist = dclean at K=1, uncertainty=margin,
grad-consistency=gc, perturbation=mean_abs).
"""
from __future__ import print_function
import os, numpy as np
from scipy import stats as sps
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
R = np.load(os.path.join(_R, 'final_verify.npz'))
MS = ['ttc', 'momentum', 'purela', 'nac']
KS_DEC = [4, 8]   # decision point budgets
KS_FEAT = [1, 2]


def corr(s, m, k):
    return R[f's{s}_{m}_K{k}__correct'].astype(np.int8)


for K in KS_DEC:
    print('=' * 64)
    print('Oracle test @ budget K = %d  (10000 samples, seed 0)' % K)
    C = {m: corr(0, m, K) for m in MS}
    fixed = {m: C[m].mean() * 100 for m in MS}
    print('A. fixed TTC      = %.2f%%' % fixed['ttc'])
    print('B. fixed NAC      = %.2f%%' % fixed['nac'])
    any_ok = np.array([C[m] for m in MS]).any(axis=0)          # Oracle (any solver defends)
    all_ok = np.array([C[m] for m in MS]).all(axis=0)
    oracle = any_ok.mean() * 100
    print('C. Oracle routing = %.2f%%' % oracle)
    gain = oracle - fixed['nac']
    print('D. Oracle - NAC   = %+.2f pp   (std/abs) %s' % (gain, '...'))
    print('   [Oracle is UPPER-BOUND oracle with zero routing cost]')
    # note: also oracle via each single-correct + ties meeting criterion (>=1 solver)

    # Best-solver attribution among success samples (that any solver defends)
    n_saved = int(any_ok.sum())
    print('\nE. Best-solver share (among %d samples +defended, "tie"=>=2 solvers, "only-X"=exactly one):' % n_saved)
    share = {}
    for m in MS:
        onlym = (C[m] == 1) & (np.array([C[o] for o in MS if o != m]).sum(axis=0) == 0)
        share[m] = onlym.sum()
    tie = (np.array([C[m] for m in MS]).sum(axis=0) >= 2) & any_ok
    none = (~any_ok).sum()
    tot = any_ok.sum() + none
    for m in MS:
        print('   only-%-9s %5d (%.1f%% of %d samples)' % (m, share[m], 100.0 * share[m] / max(tot, 1), tot))
    print('   tie(>=2)       %5d (%.1f%%)' % (tie.sum(), 100.0 * tie.sum() / max(tot, 1)))
    print('   none(all fail) %5d (%.1f%%)' % (none, 100.0 * none / max(tot, 1)))

    # heterogeneity metric: number of samples where the preferred solver differs
    # from a single fixed favourite (NAC). i.e. samples NAC fails but another passes
    nac_only_gain = ((~C['nac'].astype(bool)) & any_ok).sum()
    print('G. heterogeneity (NAC fails but >=1 other solves): %d (%.1f%% of saved)' %
          (nac_only_gain, 100.0 * nac_only_gain / max(n_saved, 1)))

# ---- feature correlation (F) : per-sample instrumented features (pilot_data,
# seed0, eps=4/255, SAME adversarial inputs/order as final_verify) -> Best-Solver
print('\n' + '=' * 64)
print('F. cheap routing-features (per-sample, from pilot_data.npz) vs Best-Solver')
P = np.load(os.path.join(_R, 'pilot_data.npz'))
K = 8
C = {m: corr(0, m, K) for m in MS}
any_ok = np.array([C[m] for m in MS]).any(axis=0)
nac_ok = C['nac'].astype(bool)
# routing target: sample is saved ONLY by a non-NAC solver -> router must NOT pick NAC
tgt_best_nac = (nac_ok).astype(np.float64)                      # best=broadly "NAC works"
tgt_leave_nac = ((~nac_ok) & any_ok).astype(np.float64)         # best=leave NAC (hardest, most valuable route signal)
feats = {
    'feature_drift(diff_ratio)': P['nac_K2_eps2__sig_diff_ratio'],
    'gradient_norm(mean)': P['nac_K2_eps2__sig_grad_norm_mean'],
    'attack-cos(g0)': P['nac_K2_eps2__sig_cos_atkdelta_g0'],
    'one-step_obj_gain': P['nac_K2_eps2__sig_gain_rate'],
    'feature_displacement': P['nac_K2_eps2__sig_emb_jump'],
    'pred_uncertainty(-margin)': -R['s0_ttc_K1__margin'],
}
res = {}
for fn, x in feats.items():
    x = np.asarray(x, dtype=np.float64).reshape(-1)[:10000]
    row = {}
    for tn, y in (('best_nac', tgt_best_nac), ('leave_nac', tgt_leave_nac)):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            row[tn] = (np.nan, np.nan); continue
        row[tn] = (sps.pearsonr(x[m], y[m])[0], sps.spearmanr(x[m], y[m])[0])
    res[fn] = row
    print('  %-30s' % fn + ''.join('  %-9s p=%+.3f sp=%+.3f' % (tn, *r) for tn, r in row.items()))
top = sorted(res.items(), key=lambda kv: -abs(kv[1]['leave_nac'][0] or 0))[:3]
print('\n  Top-3 most predictive features (vs leave_nac):', '; '.join(t[0] for t in top))
print('\nNOTE: Oracle = perfect zero-cost upper bound. Real features capture only part of it.')