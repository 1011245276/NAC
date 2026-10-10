"""
Objective-level MoE oracle analysis (consumes objective_experts.npz + final_verify).
Reports:
 - NAC fixed (final_verify), per-expert acc, best single objective
 - objective oracle over ALL experts (clean/label-anchored B/C are oracle-informative)
 - objective oracle over TEST-TIME-realistic experts only (A repel-own, D confidence)
 - Oracle - NAC (both realistic and full)
 - heterogeneity (only-X / multiple / none)
"""
from __future__ import print_function
import os, numpy as np
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
Z = np.load(_R + r'\objective_experts.npz')
O = {v: Z[f'obj_{v}__correct'].astype(bool) for v in 'ABCD'}
F = np.load(_R + r'\final_verify.npz')
NAC = F['s0_nac_K4__correct'].astype(bool)
TTC = F['s0_ttc_K4__correct'].astype(bool)
N = len(NAC)

print('='*66)
print('Objective-level oracle   N=%d   eps=4/255 K=4 (plain-PGD optimizer)' % N)
nac = NAC.mean()*100; ttc = TTC.mean()*100
print('TTC (fixed)            = %.2f%%' % ttc)
print('NAC (fixed)            = %.2f%%' % nac)
EXP = 'ABCD'
for v in EXP:
    print('  expert %s acc = %.2f%%' % (v, O[v].mean()*100))
best_single = max(O[v].mean()*100 for v in EXP)
print('best single objective  = %.2f%%' % best_single)

# full objective oracle (all 4 experts)
full = np.array([O[v] for v in EXP]).any(axis=0)
oracle_all = full.mean()*100
# test-time-realistic oracle (A + D only)
tt_real = (O['A']|O['D'])
oracle_real = tt_real.mean()*100
# include NAC as an expert too
with_nac = (full|NAC).mean()*100

print('\nObjective Oracle (A/B/C/D, any)   = %.2f%%' % oracle_all)
print('Objective Oracle (test-time A/D)   = %.2f%%   [deployable]' % oracle_real)
print('Objective Oracle (incl. NAC expert)= %.2f%%' % with_nac)
print('\nOracle - NAC:')
print('  full (A/B/C/D)     %+.2f pp' % (oracle_all - nac))
print('  test-time (A/D)    %+.2f pp' % (oracle_real - nac))

# --- heterogeneity among objective experts only (which objective is unique best) ---
print('\n=== Objective heterogeneity (4 experts) ===')
any_ok = full
nok = int(any_ok.sum())
only = {}
for v in EXP:
    others = np.array([O[o] for o in EXP if o != v]).any(axis=0)
    only[v] = ((O[v]) & (~others)).sum()
multi = (np.array([O[v] for v in EXP]).sum(axis=0) >= 2).sum()  # count samples with >=2
none = (~any_ok).sum()
print('samples +defended by >=1 objective: %d' % nok)
for v in EXP:
    print('  only-%s: %d (%.1f%% of corpus)' % (v, only[v], 100.0*only[v]/N))
print('  multiple(>=2): %d (%.1f%%)' % (multi, 100.0*multi/N))
print('  none(all fail): %d (%.1f%%)' % (none, 100.0*none/N))

# test-time value: can A route to D (or vice versa) help? overlap
print('\n=== test-time experts (A,D) overlap ===')
print('  A correct: %.1f%%  D correct: %.1f%%  A|D oracle: %.2f%%  A&D: %.1f%%' % (
    O['A'].mean()*100, O['D'].mean()*100, (O['A']|O['D']).mean()*100, (O['A']&O['D']).mean()*100))
print('  D-only gain over A: %+.2f pp (A fails, D ok)' % ((~O['A']&O['D']).mean()*100))
print('  A-only gain over D: %+.2f pp (D fails, A ok)' % ((~O['D']&O['A']).mean()*100))

# NAC as extra expert vs oracle
print('\n=== with NAC as an expert ===')
print('  NAC-only gain vs objective-oracle: %+.2f pp (samples only NAC saves)' % ((NAC&~full).mean()*100))
print('  full-oracle (objs+NAC) = %.2f%%' % with_nac)