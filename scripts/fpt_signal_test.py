"""
FPT two-level active-probe signal evaluation vs existing static signals.

Reads fpt_signals.npz (fpt_tau from adversarial samples, seed0, exact index
alignment with final_verify) and evaluates it as a routing signal for
  y = switch away from NAC to Momentum  (y=1 iff NAC_wrong & Momentum_ok)
using the SAME threshold-on-val / frozen / report-on-test protocol as
routing_signal_test.py. Also reports AUROC for the same target for comparison
with the static signals already measured (~0.50).

Signal is sign-invariant: router picks sign+threshold on VAL only; AUROC
reported on TEST.
"""
from __future__ import print_function
import numpy as np
from scipy import stats as sps
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
S = np.load(_R + r'\fpt_signals.npz')
R = np.load(_R + r'\final_verify.npz')
G = np.load(_R + r'\geometry.npz')

SEED0 = np.random.RandomState(0)
N = 10000
idx = SEED0.permutation(N)
tr, va, te = idx[:6000], idx[6000:8000], idx[8000:]

tau = np.asarray(S['fpt_tau'], dtype=np.float64).reshape(-1)[:N]
# alignment guard
al = np.linalg.norm(np.asarray(S['f_adv']) - np.asarray(G['f_adv']), axis=1)
print('alignment: f_adv max_L2=%.2e (must be tiny)' % al.max())


def corr(m, K):
    return R[f's0_{m}_K{K}__correct'].astype(bool)


def auroc(s, y):
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() < 3 or (y[m]).sum() == 0 or ((1 - y[m]).sum() == 0):
        return np.nan
    s, y = s[m], y[m]
    order = np.argsort(s, kind='mergesort')
    ss, yy = s[order], y[order]
    ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss) + 1)
    n1, n0 = yy.sum(), (1 - yy).sum()
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# descriptive: dist of tau for clean vs adv (clean from geometry labels not stored
# here; f_adv only. Report signal vs the routing target and its raw stats.)
print('\nFPT two-level probe on ADV samples (seed0):')
print('  fpt_tau: mean=%.4f std=%.4f nan=%d' % (
    np.nanmean(tau), np.nanstd(tau), int(np.isnan(tau).sum())))

for K in (4, 8):
    cN = corr('nac', K); cM = corr('momentum', K)
    y = (cN == False) & (cM == True)
    print('=' * 60)
    print('== K=%d  (NAC %.2f, Momentum %.2f)  switch-gain pos=%d/%d ==' % (
        K, cN.mean()*100, cM.mean()*100, y.sum(), N))
    # router: fit sign+threshold on val, report test
    best = (-1e9, 0.0, 1)
    for sign in (1, -1):
        xv = (tau * sign)[va]
        qs = np.quantile(xv[~np.isnan(xv)], np.linspace(0.05, 0.95, 40))
        for thr in qs:
            rn = xv < thr
            routed = np.where(rn, cN[va], cM[va]).mean()
            if routed > best[0]:
                best = (routed, thr, sign)
    _, thr, sign = best
    xt = (tau * sign)[te]
    rn = xt < thr
    routed_test = np.where(rn, cN[te], cM[te]).mean() * 100
    auroc_test = auroc(tau, y)
    # correlation signal<->y (descriptive)
    rho = sps.spearmanr(tau[te], y[te]).statistic if len(np.unique(y[te])) > 1 else np.nan
    print('  FPT fpt_tau  testAUROC=%.3f  routed=%.2f  |rho|=%.3f' % (
        auroc_test, routed_test, abs(rho) if np.isfinite(rho) else np.nan))
    print('  ref: NAC %.2f  2-way-oracle %.2f  full-4way-oracle %.2f' % (
        cN.mean()*100, (cN | cM).mean()*100,
        (cN | corr('momentum', K) | corr('purela', K) | corr('ttc', K)).mean()*100))