"""
Control experiment analysis: does FPT-Noise's clean/adv discrimination reproduce
on our setup? Compares the two-level FPT τ on clean vs adversarial samples that
used POINTWISE-IDENTICAL noise draws.

Report:
  1. clean vs adv τ distributions (mean/std, overlap via Wasserstein/quantiles)
  2. clean/adv discrimination AUROC (this is FPT-Noise's actual goal)
  3. contrast with the routing AUROC (0.501/0.498) — the whole point:
     discriminating clean/adv  (FPT's purpose) is easy/possible,
     routing which solver to use is not (both are on adv-only samples).
"""
from __future__ import print_function
import numpy as np
from scipy import stats as sps
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
A = np.load(_R + r'\fpt_signals.npz')   # adv side
C = np.load(_R + r'\fpt_clean.npz')     # clean side
G = np.load(_R + r'\geometry.npz')

t_adv = np.asarray(A['fpt_tau'], dtype=np.float64).reshape(-1)
t_cln = np.asarray(C['fpt_tau_clean'], dtype=np.float64).reshape(-1)
n = min(len(t_adv), len(t_cln)); t_adv = t_adv[:n]; t_cln = t_cln[:n]

# alignment sanity on the shared feature stream
al = np.linalg.norm(np.asarray(A['f_adv']) - np.asarray(G['f_adv']), axis=1)
print('adv f_adv max_L2=%.2e (must be 0)' % al.max())


def auroc(s0, s1):
    s = np.concatenate([s0, s1]); y = np.concatenate([np.zeros(len(s0)), np.ones(len(s1))])
    order = np.argsort(s, kind='mergesort'); ss, yy = s[order], y[order]
    ranks = np.empty(len(ss)); ranks[order] = np.arange(1, len(ss) + 1)
    n0, n1 = (yy == 0).sum(), (yy == 1).sum()
    return (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


d = sps.wasserstein_distance(t_cln, t_adv)
print('\n=== FPT two-level τ on CLEAN vs ADV (n=%d, pointwise-identical noise) ===' % n)
print('  clean: mean=%.4f std=%.4f    adv: mean=%.4f std=%.4f' % (
    t_cln.mean(), t_cln.std(), t_adv.mean(), t_adv.std()))
print('  clean/adv discrimination AUROC = %.3f   (FPT-Noise origin purpose)' % auroc(t_cln, t_adv))
print('  Wasserstein distance = %.4f  (feature-norm-normalised)' % d)
q = np.nanquantile(t_cln, [0.25, 0.75]); qa = np.nanquantile(t_adv, [0.25, 0.75])
print('  clean IQR=[%.3f,%.3f]  adv IQR=[%.3f,%.3f]  overlap=%s' % (
    q[0], q[1], qa[0], qa[1], 'YES-wide' if (q[1] > qa[0]) else 'SEPARATED'))
print('\nCONTRAST: clean/adv discrimination AUROC=%.3f  vs  solver-routing AUROC=0.501/0.498 (K=4/8)' % (
    auroc(t_cln, t_adv)))