"""Analyze final_verify.npz (3 seeds x 4 methods x K{1,2,4,6,8}, 10000 samples).
Reports mean+-std over seeds for the 8 metrics, marginal gain, feature
separation, perturbation, gradient consistency, + McNemar/pairwise stats.
Honest: no selective reporting; computes A-E verdicts.
"""
from __future__ import print_function
import os, numpy as np, json
from scipy import stats as sps
_ROOT = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
R = np.load(os.path.join(_ROOT, 'final_verify.npz'))
KS = [1, 2, 4, 6, 8]; MS = ['ttc', 'momentum', 'purela', 'nac']; SEEDS = [0, 1, 2]

def arr(seed, m, k, key):
    return R[f's{seed}_{m}_K{k}__{key}']

def per_seed(m, k, key):
    return np.array([arr(s, m, k, key).mean() for s in SEEDS])

# ---------------- 1. defended accuracy ----------------
print('=' * 60)
print('DEFENDED ACCURACY  mean+-std over 3 seeds (%)')
print('%-9s' % 'method' + ''.join('K%-4d' % k for k in KS))
acc = {m: {k: per_seed(m, k, 'correct') * 100 for k in KS} for m in MS}
for m in MS:
    print('%-9s' % m + ''.join('%6.1f' % acc[m][k].mean() for k in KS) +
          '   | +/- ' + ''.join('%4.1f' % acc[m][k].std() for k in KS))
print()
for k in KS:
    base = acc['ttc'][k]
    print('K=%d TTC %.1f | Mom %.1f | P_LA %.1f | NAC %.1f' % (
        k, base.mean(), acc['momentum'][k].mean(), acc['purela'][k].mean(), acc['nac'][k].mean()))

# ---------------- 2. plateau / ceiling (A) ----------------
print('\n' + '=' * 60)
print('A) CEILING/PLATEAU: K6->K8 delta (pp) per seed')
for m in MS:
    d = [(acc[m][8][i] - acc[m][6][i]) for i in range(3)]
    print('  %-9s K8-K6 = %+.1f +-%0.1f pp' % (m, np.mean(d), np.std(d)))

# ---------------- 3. perturbation budget (B) ----------------
print('\n' + '=' * 60)
print('B) PERTURBATION: frac@boundary (mean+-std)  +  mean|d|')
for k in KS:
    print('K=%d' % k)
    for m in MS:
        fb = per_seed(m, k, 'frac'); ma = per_seed(m, k, 'mean_abs')
        print('   %-9s frac@bound %.3f+%.3f  mean|d| %.4f+%.4f' % (
            m, fb.mean(), fb.std(), ma.mean(), ma.std()))
# stat: momentum vs ttc frac@bound at K8 (3 seeds, per-sample via pooled)
def pooled_ttest(mA, mB, k, key):
    a = np.concatenate([arr(s, mA, k, key) for s in SEEDS])
    b = np.concatenate([arr(s, mB, k, key) for s in SEEDS])
    return sps.ttest_ind(a, b).pvalue
print('  McNemar/pairwise (per-sample, 3 seeds pooled): momentum > ttc frac@K8',
      't-test p=%.2e' % pooled_ttest('momentum', 'ttc', 8, 'frac'))
print('  momentum > ttc frac@K4 p=%.2e' % pooled_ttest('momentum', 'ttc', 4, 'frac'))

# ---------------- 4. feature separation (C) ----------------
print('\n' + '=' * 60)
print('C) FEATURE-SPACE: separation (d2adv - d2clean), mean+-std over seeds')
for k in KS:
    print('K=%d' % k + ''.join('  %s %+.2f' % (m, (per_seed(m, k, 'dadv') - per_seed(m, k, 'dclean')).mean())
                               for m in MS))
# equality of budget: nac vs momentum
for k in KS:
    a = per_seed('nac', k, 'mean_abs').mean(); b = per_seed('momentum', k, 'mean_abs').mean()
    print('K=%d  nac/mom mean|d| ratio = %.4f  (frac %+.3f)' % (
        k, a / b if b else 0,
        per_seed('nac', k, 'frac').mean() - per_seed('momentum', k, 'frac').mean()))
# NAC vs momentum accuracy gap + feature-separation gap at same budget
print('\n  NAC-momentum acc gap (pp)   &   separation gap (adv-clean delta)')
for k in KS:
    agap = acc['nac'][k].mean() - acc['momentum'][k].mean()
    sep_n = (per_seed('nac', k, 'dadv') - per_seed('nac', k, 'dclean')).mean()
    sep_m = (per_seed('momentum', k, 'dadv') - per_seed('momentum', k, 'dclean')).mean()
    print('   K=%d acc_gap %+.2f pp | NAC_sep %+.3f vs Mom_sep %+.3f' % (k, agap, sep_n, sep_m))

# ---------------- 5. gradient consistency (D) ----------------
print('\n' + '=' * 60)
print('D) GRADIENT-DIRECTION CONSISTENCY  (mean cos(g_t,g_{t-1}), higher=stronger alignment)')
for k in KS:
    print('K=%d' % k + ''.join('  %s %.3f' % (m, per_seed(m, k, 'gc').mean()) for m in MS))

# ---------------- 6. marginal gain (metric 8) ----------------
print('\n' + '=' * 60)
print('MARGINAL GAIN (pp added at each K step), mean over seeds: K->K+')
KORD = [2, 4, 6, 8]
prev = {m: acc[m][1] for m in MS}
for k in KORD:
    print('K=%d' % k + ''.join('  %s %+.1f' % (m, (acc[m][k].mean() - prev[m].mean())) for m in MS))
    prev = {m: acc[m][k] for m in MS}

# ---------------- E: K2 relative advantage ----------------
print('\n' + '=' * 60)
print('E) K2 relative advantage NAC-vs-TTC: %.2fx  (%+d samples/seed mean)' % (
    acc['nac'][2].mean() / max(acc['ttc'][2].mean(), 1e-9),
    int(np.round((acc['nac'][2].mean() - acc['ttc'][2].mean()) / 100 * 10000))))