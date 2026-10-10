"""
Learned router feasibility pilot: can a NONLINEAR router on test-time features
(f_adv + adversarial class-logits) capture any of the +19pp solver-routing
oracle, vs the hand-made single-step signals which were all AUROC~0.5?

Experts: TTC / Momentum / PureLA / NAC at K=8 (all test-time deployable).
Target per sample = the best solver (argmax over obtainables; 'none' if none).
Strict train6000 / val2000 / test2000. Routed acc = correctness of predicted solver.
"""
from __future__ import print_function
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
_R = r'F:\codex\NAC\nac_project\results\adaptive_pilot'
G = np.load(_R + r'\geometry.npz'); F = np.load(_R + r'\final_verify.npz')
N = len(G['label']); K = 8
fa = G['f_adv']; proto = G['protos']; pn = proto/np.linalg.norm(proto, axis=1, keepdims=True)
fa_n = fa/np.linalg.norm(fa, axis=1, keepdims=True)
sim = fa_n @ pn.T                                   # [N,10] adversarial class cos
probs = np.exp(sim*10.0); probs/=probs.sum(1, keepdims=True)
ent = -(probs*np.log(probs+1e-12)).sum(1)
T = np.argsort(-sim, axis=1); margin = sim[np.arange(N),T[:,0]]-sim[np.arange(N),T[:,1]]
# test-time feature vector (all from attacked image + fixed class protos):
feat = np.concatenate([sim, probs, margin[:,None], ent[:,None]], axis=1)  # [N, 12]

# solver correctness at K8
corr = {m: F[f's0_{m}_K8__correct'].astype(bool) for m in ['ttc','momentum','purela','nac']}
MS = ['ttc','momentum','purela','nac']
any_ok = np.array([corr[m] for m in MS]).any(axis=0)
# 5-class target: index of a correct solver, else 4 (=none)
y = np.full(N, 4, dtype=int)
for i, m in enumerate(MS):
    y[corr[m]] = i
y[~any_ok] = 4
y = y.astype(int)
# among overlap (multiple correct) we keep first; fine for oracle-argmax proxy

rng = np.random.RandomState(0); idx = rng.permutation(N)
tr, va, te = idx[:6000], idx[6000:8000], idx[8000:]
exact = {m: corr[m].astype(int) for m in MS}

def eval_router(pred, test_idx, name):
    te_i = test_idx; ntest = len(te_i)
    routed = np.zeros(ntest)
    for k, s in enumerate(te_i):
        if pred[k] < 4 and exact[MS[pred[k]]][s]:
            routed[k] = 1
    oracle = any_ok[te_i].mean()*100
    nac = corr['nac'][te_i].mean()*100
    mom = corr['momentum'][te_i].mean()*100
    print('%-6s routed=%.2f  | vs NAC %.2f (%+.2f) | vs Oracle %.2f | capture-of-oracle-gap %.0f%%' % (
        name, routed.mean()*100, nac, routed.mean()*100-nac, oracle,
        100.0*(routed.mean()-nac/100)/(oracle/100-nac/100) if oracle>nac else float('nan')))

scaler = StandardScaler().fit(feat[tr])   # fit scaler on train only
Xtr, Xva, Xte = scaler.transform(feat[tr]), scaler.transform(feat[va]), scaler.transform(feat[te])

# baselines
print('=== BASELINES (test set) ===')
eval_router(np.full(len(te),3), te, 'alwaysNAC')
eval_router(np.full(len(te),1), te, 'alwaysMom')
eval_router(np.full(len(te),0), te, 'alwaysTTC')
print('oracle(any)=%.2f  NAC fixed=%.2f  => gap=%.2fpp' % (
    any_ok[te].mean()*100, corr['nac'][te].mean()*100,
    (any_ok[te].mean()-corr['nac'][te].mean())*100))

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=0,
                    early_stopping=True, validation_fraction=0.2)
mlp.fit(Xtr, y[tr])
eval_router(mlp.predict(Xva), va, 'MLP(val)')
eval_router(mlp.predict(Xte), te, 'MLP(test)')
print('  MLP 5-class val acc=%.3f test acc=%.3f' % (mlp.score(Xva,y[va]), mlp.score(Xte,y[te])))

# RandomForest
rf = RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=5)
rf.fit(Xtr, y[tr])
eval_router(rf.predict(Xva), va, 'RF(val)')
eval_router(rf.predict(Xte), te, 'RF(test)')
print('  RF 5-class val acc=%.3f test acc=%.3f' % (rf.score(Xva,y[va]), rf.score(Xte,y[te])))

# also: can we even predict 'none' vs 'restorable'? (binary oracle-ish)
print('\n=== can router tell restorable vs not? (2-class) ===')
y2 = any_ok.astype(int)
print('restorable fraction=%.3f' % any_ok.mean())
for name, m in [('MLP',MLPClassifier(hidden_layer_sizes=(64,),max_iter=800,random_state=0).fit(Xtr,y2[tr])),
                ('RF',RandomForestClassifier(300,random_state=0).fit(Xtr,y2[tr]))]:
    from sklearn.metrics import roc_auc_score
    pv=m.predict_proba(Xte)[:,1]
    print('  %s restorable-AUROC(test)=%.3f' % (name, roc_auc_score(y2[te],pv)))