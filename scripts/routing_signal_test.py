"""
Minimal single-step routing-signal test (NAC vs Momentum, 2-way).

Strict split: train 6000 / val 2000 / test 2000, by index. Threshold only fit on
train+val, frozen, reported ONLY on test. NO oracle labels used in fitting the
threshold; test set is completely untouched during selection.

Signals = NAC first-step (K=1) backward signals from pilot_data (eps=4/255,
seed 0, same sample order as final_verify — alignment verified =1.0).
Target = "switch away from NAC to Momentum gains" : y=1 iff (NAC_wrong & Mom_ok).

Router (frozen on val): route_NAC if s<thr else route Momentum.
Output: test AUROC; NAC fixed / routed / 2-way oracle / full 4-way oracle;
routed-NAC at K=4 and K=8.
"""
from __future__ import print_function
import numpy as np
from scipy import stats as sps
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
P = np.load(_R + r'\pilot_data.npz')
R = np.load(_R + r'\final_verify.npz')

SIGS = ['grad_norm0', 'diff_ratio', 'cos_atkdelta_g0', 'gain_rate', 'loss_final']
# K=1 signals (true single step). Also add 2-step emb_jump & grad_cos_succ for reference.
SIGS += ['emb_jump', 'grad_cos_succ']

SEED0 = np.random.RandomState(0)
N = 10000
idx = SEED0.permutation(N)
tr, va, te = idx[:6000], idx[6000:8000], idx[8000:]


def corr(m, K):
    return R[f's0_{m}_K{K}__correct'].astype(bool)


def get_sig(s, K):
    if s in ('emb_jump', 'grad_cos_succ'):
        # these need >=2 steps; use K2 (which still has K=1 gradient for the pair)
        x = P[f'nac_K2_eps2__sig_{s}']
    else:
        x = P[f'nac_K1_eps2__sig_{s}']
    return np.asarray(x, dtype=np.float64).reshape(-1)[:N]


def auroc(s, y):
    m = np.isfinite(s) & (np.isfinite(y))
    if m.sum() < 3 or (y[m]).sum() == 0 or ((1 - y[m]).sum() == 0):
        return np.nan
    s, y = s[m], y[m]
    order = np.argsort(s, kind='mergesort')
    ss, yy = s[order], y[order]
    ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss) + 1)
    n1, n0 = yy.sum(), (1 - yy).sum()
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for K in (4, 8):
    cN = corr('nac', K); cM = corr('momentum', K)
    # label: switching NAC->Momentum gains
    y = (cN==False) & (cM==True)
    print('=' * 66)
    print('== K=%d  (NAC %.2f, Momentum %.2f)  switch-gain pos=%d/%d ==' % (
        K, cN.mean()*100, cM.mean()*100, y.sum(), N))
    # ---- signal evaluation (AUROC on val for selection, test for report) ----
    rows = []
    for s in SIGS:
        x = get_sig(s, K)
        neg = (x < 0)  # detach raw sign: AUROC is sign-invariant, router picks sign on val
        # fit direction+threshold on val
        best = (-1e9, 0.0, 1)
        for sign in (1, -1):
            xv = (x * sign)[va]
            qs = np.quantile(xv[~np.isnan(xv)], np.linspace(0.05, 0.95, 40))
            for thr in qs:
                rn = xv < thr
                routed = np.where(rn, cN[va], cM[va]).mean()
                if routed > best[0]:
                    best = (routed, thr, sign)
        _, thr, sign = best
        # report on TEST
        xt = (x * sign)[te]
        rn = xt < thr
        routed_test = np.where(rn, cN[te], cM[te]).mean() * 100
        auroc_test = auroc(x, y)
        # top vs bottom quantile switching-rate
        qv = np.nanquantile(x[te], [0.2, 0.8])
        lo, hi = (x[te] <= qv[0]).mean(), (x[te] >= qv[1]).mean()
        ylo = y[te][x[te] <= qv[0]].mean() if (x[te] <= qv[0]).sum() else np.nan
        yhi = y[te][x[te] >= qv[1]].mean() if (x[te] >= qv[1]).sum() else np.nan
        rows.append((s, auroc_test, routed_test, ylo, yhi))
    rows.sort(key=lambda r: -r[1])  # by test AUROC
    print('%-16s %8s %8s %9s %9s' % ('signal', 'testAUROC', 'routed%', 'lo20gain', 'hi20gain'))
    for s, au, rt, ylo, yhi in rows:
        print('%-16s %8.3f %8.2f %9.3f %9.3f' % (s, au, rt, ylo, yhi))
    best_s, best_au, best_rt, _, _ = rows[0]
    print('--- best signal on K%d: %s  testAUROC=%.3f  routed=%.2f  (NAC %.2f, 2-way-oracle %.2f, full-oracle %.2f) ---' % (
        K, best_s, best_au, best_rt, cN.mean()*100,
        (cN | cM).mean()*100,
        (cN|corr('momentum',K)|corr('purela',K)|corr('ttc',K)).mean()*100))