# Adaptive-signal pilot analysis (auto-generated)

- samples: 10000  clean: 85.07%  adv(PGD-10,4/255): 0.00%

## Grid accuracy (NAC, %%)

| eps \ K | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1/255 | 0.06 | 0.23 | 0.67 | 0.99 |
| 2/255 | 0.06 | 5.18 | 14.39 | 20.22 |
| 4/255 | 0.26 | 17.57 | 33.73 | 37.51 |

TTC(K=2,4/255): 7.16%   NAC(K=2,4/255): 17.57%

## Benefit structure (eps=4/255)

- K2 17.57% -> K3 33.73% -> K4 37.51%
- recoverable K2->K4: 2423   regression K2->K4: 429

## Signal ranking (top 15 by |Spearman| vs benefit)

| signal | spearman | spearman_mg | pearson | AUROC_recover | AUROC_regress |
|---|---|---|---|---|---|
| K2_eps2:emb_jump | -0.114 | 0.150 | -0.110 | 0.427 | 0.569 |
| K2_eps2:gain_rate | -0.112 | 0.151 | -0.107 | 0.428 | 0.565 |
| K2_eps2:loss_final | -0.112 | 0.152 | -0.107 | 0.428 | 0.563 |
| K2_eps2:grad_norm_mean | -0.072 | 0.097 | -0.065 | 0.446 | 0.496 |
| K2_eps2:cos_atkdelta_g0 | -0.028 | -0.038 | -0.032 | 0.477 | 0.492 |
| K1_eps2:cos_atkdelta_g0 | -0.028 | -0.038 | -0.032 | 0.477 | 0.492 |
| K2_eps2:velocity_cos | 0.027 | 0.031 | 0.019 | 0.546 | 0.644 |
| K2_eps2:diff_ratio | 0.018 | 0.041 | 0.010 | 0.505 | 0.451 |
| K1_eps2:diff_ratio | 0.018 | 0.041 | 0.010 | 0.505 | 0.451 |
| K1_eps2:loss_final | 0.016 | 0.056 | 0.000 | 0.504 | 0.456 |
| K2_eps2:grad_norm0 | 0.014 | 0.024 | 0.004 | 0.503 | 0.462 |
| K1_eps2:grad_norm0 | 0.014 | 0.024 | 0.004 | 0.503 | 0.462 |
| K1_eps2:grad_norm_mean | 0.014 | 0.024 | 0.004 | 0.503 | 0.462 |
| K2_eps2:step_align | -0.009 | 0.035 | -0.021 | 0.522 | 0.664 |
| K2_eps2:grad_cos_succ | -0.004 | 0.034 | -0.018 | 0.520 | 0.629 |

## Best adaptive rules

- best overall: velocity_cos (from K2 prefix) thr=0.6760 route=95% -> acc 36.85% @ avgK 3.90
- best budget(avgK<=3): diff_ratio thr=0.1030 route=50% -> acc 27.86% @ avgK 3.00
- anchors: fixed K2 17.57% @2fwd, fixed K4 37.51% @4fwd
