"""
Geometry analysis of CLIP feature-space mechanism (consumes geometry.npz).

Questions (from the research directive):
 A. Is adversarial feature displacement (d = f_adv - f_clean) structurally
    aligned across samples / classes? i.e. is there a common "attack axis"?
 B. Does d have a stable relation to the correct-class text prototype?
 C. Does counterattack displacement aim back toward the correct prototype?
 D. Do NAC-success vs NAC-fail samples differ in feature geometry?
 E. Is there a simple feature-space signal that could enter a new objective?

Only reports measured geometry + directional consistency. No manufactured claims.
"""
from __future__ import print_function
import os, numpy as np
from scipy import stats as sps
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
Z = np.load(_R + r'\geometry.npz')
fc, fa, fd = Z['f_clean'], Z['f_adv'], Z['f_def']
lab = Z['label']; proto = Z['protos']; (
    clean_ok, adv_ok, nac_ok) = Z['clean_ok'], Z['adv_ok'], Z['nac_ok']
N = len(lab); C = proto.shape[0]
L = lab
norm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
fn, fan, fdn = norm(fc), norm(fa), norm(fd)
pn = norm(proto)

def cosmat(A, B):  # A[N,d], B[C,d] -> N,C cos
    return A @ B.T

def cdot(A, P):  # A[N,d] . P[N,d] elementwise -> N
    return (A * P).sum(1)

# distance to correct-class prototype (cosine)
d_clean_c = cdot(fn, pn[L])          # cos(f_clean, proto_correct)
d_adv_c = cdot(fan, pn[L])
d_def_c = cdot(fdn, pn[L])
# distance to wrong-class (nearest wrong) and mean wrong
def wrong_cos(A_n):  # max cos over wrong classes
    cA = A_n @ pn.T
    cA[np.arange(N), L] = -2
    wmax = cA.max(1)
    return wmax
wc_clean = wrong_cos(fn); wc_adv = wrong_cos(fan); wc_def = wrong_cos(fdn)
margin_clean = d_clean_c - wc_clean; margin_adv = d_adv_c - wc_adv; margin_def = d_def_c - wc_def

print('='*72)
print('Geometry pilot   N=%d   clean=%.2f  adv=%.2f  NAC@K4=%.2f' %
      (N, clean_ok.mean()*100, adv_ok.mean()*100, nac_ok.mean()*100))

# ---------------- A. adversarial displacement direction ----------------
da = fa - fc                      # adversarial feature displacement
dan = norm(da)
mean_d = da.mean(0); mean_dn = mean_d / np.linalg.norm(mean_d)
align_mean = np.abs(dan @ mean_dn).mean()   # |cos| with global mean direction
align_mean_signed = (dan @ mean_dn).mean()
# coherence: mean pairwise cos of normalized displacements (bounded estimate)
sample = np.random.RandomState(0).choice(N, 2000, replace=False)
pair = dan[sample] @ dan[sample].T
coh = (pair[np.triu_indices(len(sample), 1)]).mean()
print('\nA. adversarial displacement axis consistency:')
print('   mean |cos(d_adv, mean_d)| = %.3f  (cos(d_adv, mean_d) = %.3f)' % (align_mean, align_mean_signed))
print('   mean pairwise cos (2000-samp) = %.3f  [0=random, >0=shared dir]' % coh)

# per-class alignment of mean displacement with mean wrong-prototype-walk direction
print('\n   per-class: displacement magnitude |d_adv| and alignment with class-mean dir')
for c in range(C):
    m = np.where(L == c)[0]
    mag = np.linalg.norm(da[m], axis=1).mean()
    cm_dir = da[m].mean(0); cm_dir /= np.linalg.norm(cm_dir)
    a = np.abs(dan[m] @ cm_dir).mean()
    print('   class %d: |d|=%.2f  self-align=%.3f' % (c, mag, a))

# ---------------- B. relation to correct-class prototype ----------------
print('\nB. attack vs correct-prototype geometry (cos change):')
print('   cos clean->proto_c %+.3f -> adv %+.3f  (delta %+.4f, p)' % (
    d_clean_c.mean(), d_adv_c.mean(), (d_adv_c-d_clean_c).mean()))
print('   clean margin %.3f -> adv margin %.3f (delta %+.4f)' % (
    margin_clean.mean(), margin_adv.mean(), (margin_adv-margin_clean).mean()))
print('   NAC-fail: adv pulls AWAY from proto_c? frac=(d_adv_c<d_clean_c): %.3f' %
      ((d_adv_c < d_clean_c).mean()))
# is displacement direction aligned with leaving-correct-proto (i.e. -proto_c dir)?
# is displacement direction aligned with leaving-correct-proto? (informative: cos(d_adv, proto_correct))
dorp_c = cdot(dan, pn[L])
# better: does attack correlate with moving toward *a* wrong proto (the winning wrong class)?
cos_wrong_adv = da @ pn.T
cos_wrong_adv[np.arange(N), L] = -2
win_wrong = cos_wrong_adv.argmax(1)
d_to_win = cdot(dan, pn[win_wrong])  # cos(d_adv, proto_of_winning wrong class)
print('   cos(d_adv, winning-wrong-class proto) = %+.3f  (does attack point AT the adversarial class?)' % d_to_win.mean())

# ---------------- C. counterattack displacement vs prototype ----------------
dd = fd - fa ; ddn = norm(dd)
dd_c = cdot(ddn, pn[L])     # cos(counterattack shift, proto_correct)
dorp = cdot(ddn, pn[L])
print('\nC. counterattack displacement direction:')
print('   cos(d_def, proto_correct)       = %+.3f  (>0 => NAC pulls feature TOWARD correct proto)' % dorp.mean())
print('   distance-to-proto: adv %.3f -> def %.3f (cos delta %+.4f)' % (
    d_adv_c.mean(), d_def_c.mean(), (d_def_c-d_adv_c).mean()))
print('   margin: adv %.3f -> def %.3f (delta %+.4f)' % (margin_adv.mean(), margin_def.mean(), (margin_def-margin_adv).mean()))

# ---------------- D. NAC success vs fail geometry ----------------
ok, bad = nac_ok.astype(bool), ~nac_ok.astype(bool)
print('\nD. NAC success vs fail (before counterattack geometry):')
def stat(name, v):
    print('   %-26s success=%.3f   fail=%.3f   (ttest p=%.1e)' % (
        name, v[ok].mean(), v[bad].mean(), sps.ttest_ind(v[ok], v[bad]).pvalue))
stat('|d_adv| displacement mag', np.linalg.norm(da, axis=1))
stat('coholder adv-alignment |cos(d_adv,xmean)|', np.abs(dan @ mean_dn))
stat('cos adv->proto_c', d_adv_c)
stat('adv margin', margin_adv)
stat('clean margin', margin_clean)
stat('d_adv . proto_win-wrong', d_to_win)
stat('clean->proto distance', d_clean_c)

# ---------------- E. simple predictive signal(s) pre-attack -------------------
# can pre-attack geometry (clean margin / distance) predict NAC outcome?
print('\nE. clean(no-def) geometry -> NAC success prediction (AUROC):')
def auroc(s, y):
    m = np.isfinite(s)
    s, y = s[m], y.astype(bool)[m]
    if y.sum()==0 or (1-y).sum()==0: return np.nan
    o=np.argsort(s,kind='mergesort'); y2=y[o]; r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    return (r[y2==1].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(1-y).sum())
for nm, s, yv in [
        ('clean margin', margin_clean, nac_ok),
        ('clean cos proto_c', d_clean_c, nac_ok),
        ('clean |f_clean|', fc@fc if False else np.linalg.norm(fc,axis=1), nac_ok),
        ('adv margin', margin_adv, nac_ok)]:
    yv1 = yv.astype(float)
    print('   %-22s auroc=%+.3f' % (nm, auroc(s, yv1)))
# does a counterattack displacement parallel a prototype axis? overall option + which proto
ddcx = ddn @ pn.T
vari = ddcx.var(1).mean()
print('\n   variance of cos(d_def,*) over classes (low => feat drops onto a single proto axis): %.3f' % vari)
print('\n   NAC success pulls TOWARD correct proto?? cos(d_def, proto_c) success=%.3f fail=%.3f' % (
    dd_c[ok].mean(), dd_c[bad].mean()))
# compute and store key arrays for potential objective design
np.save(_R + r'/geometry_signals.npy', dict(margin_adv=margin_adv, margin_clean=margin_clean,
      d_adv_c=d_adv_c, dd_c=dd_c, d_to_win=d_to_win,
      align=np.abs(dan @ mean_dn)))
print('\n[geometry] DONE')