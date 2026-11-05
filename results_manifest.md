# Results Manifest: Table-to-Log Path Mapping

This file maps every numerical result in the manuscript to the exact experiment
log files backing it, under the repository's `results/` directory. Notation:
`{a,b}` denotes alternatives, `seed_{0,1,2}` denotes three seeds, `...` elides
the unchanged middle of a path. All accuracies are CIFAR-10 unless a dataset is
stated. Multi-seed uncertainties are sample standard deviations over per-seed
accuracies. Aggregated per-entry sources are additionally recorded in
`results/main_results.json`.

## Main tables

| Item | Log path(s) |
|---|---|
| Table 1 (main, TTC-protocol, seed 0, bs 64) | `results/eps1/pgd_eps_1.0/{ttc,nac_m0.9}/seed_0.log` |
| Table 1, ImageNet-100 (seed 0, bs 32) | `results/imagenet_full/pgd_eps_1.0/{ttc,nac_m0.9}/seed_0.log` |
| Table 1, seeds 1--2 (2026-10-10 and 2026-11-01 runs, bit-exact) | `results/eps1_seeds12/pgd_eps_1.0/{ttc,nac_m0.9}/seed_{1,2}.log`; `results/imagenet_full_seeds12/pgd_eps_1.0/{ttc,nac_m0.9}/seed_{1,2}.log` |
| Table 1, standard-step columns (alpha=eps/4, seed 0, 2026-11-04) | `results/dense_eps1/pgd_eps_1.0/{ttc,nac_m0.9}/seed_0.log`; `results/dense_eps1_imagenet/pgd_eps_1.0/{ttc,nac_m0.9}/seed_0.log` |
| Table 2 (multi-attack, seed 0) | `results/eps{1,2,4}/pgd_eps_*/{ttc,nac_m0.9}/seed_0.log`; ImageNet-100: `results/imagenet_full/pgd_eps_{1.0,2.0,4.0}/{ttc,nac_m0.9}/seed_0.log` (eps=2.0/4.0 cells from the 2026-06-04 generation, as disclosed in the caption) |
| Table 2, standard-step rows | `results/dense_eps1/...`, `results/dense_eps1_imagenet/...` (eps=1/255); `results/dense_eps2/pgd_eps_2.0/{ttc,nac_m0.9}/seed_0.log` (eps=2/255, five small datasets) |
| Table 2, DTD cells at eps=2/4 (2026-10-13/14 runs) | `results/dtd_eps2_fair/`, `results/dtd_eps4_fair/20261013_174757/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log` |
| Gate diagnostics (Section "Gate Sensitivity") | `results/_diag_grid/grid_e12.json` (step-size sweep, 256 images), `grid_e4.json`, `grid_e4r.json`, `structure_e1.json`, `structure_e2.json` (matched-budget controls, 128 images each); generator: `scripts/diag_grid_structure.py` |
| Matched-budget control (eps_def = eps_att; Supp. S2) | `results/eps_symmetric/atk{1,2,4}_def{1,2,4}_{ttc,nac}_bs64/seed_0.json` |
| Mechanism ablation, K=2 (three seeds) | `results/ablation/pgd_eps_4.0/{ttc,momentum,pure_la,nac_m0.9}/seed_{0,1,2}.log` |
| Mechanism ablation, K=4 reference rows (seed 0) | `results/final/pgd_eps_4.0/{ttc,nac_m0.9}/seed_0.log`; pure look-ahead K=4: CIFAR-10 `results/ablation_la_k4/.../_archive/seed_0_20260910_211326.log`, STL-10 `results/ablation_la_k4/pgd_eps_4.0/pure_la/seed_0.log` |
| Momentum-coefficient scan (Supp. S3) | `results/mu_scan/pgd_eps_4.0/{ttc,nac_m0.1,nac_m0.5,nac_m0.7,nac_m0.9,nac_m0.99}/seed_0.log` |
| DOC comparison, Config A (seed 0) | `results/compare/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log` |
| DOC comparison, Config B (three seeds) | `results/rerun_docB/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_{0,1,2}.log` |
| DOC default-parameter verification (40.41%) | `results/compare_doc_orig/pgd_eps_4.0/doc/seed_0.log` |
| DOC library-default gating, three seeds (40.04+-0.33%) | `results/doc_libdefault_configB/20261014_183047/pgd_eps_4.0/doc/seed_{1,2}.log` |
| AutoAttack (subset; three seeds) | `results/autoattack/autoattack_eps_4.0/{ttc,nac_m0.9}/seed_{0,1,2}.log` |
| AutoAttack full standard suite (three seeds) | `results/autoattack_full/autoattack_eps_4.0/{ttc,nac_m0.9}/seed_{0,1,2}.log`; 1000-image sanity runs: `results/autoattack_full_1k/` |
| Optimizer family, K=2: Adam / AdaGrad / RMSProp (uncorrected) | `results/optimizers/pgd_eps_4.0/{adam,adagrad,rmsprop}/seed_{0,1,2}.log` |
| Optimizer family, K=2: RMSProp bias-corrected control | `results/optimizers_stepsize/pgd_eps_4.0/rmsprop_bc/seed_{0,1,2}.log` |
| Optimizer family, K=2: TTC / NAC reference rows | `results/mu_scan/pgd_eps_4.0/{ttc,nac_m0.9}/seed_{0,1,2}.log` (the mu=0 and mu=0.9 arms) |
| K=4 optimizer family (three seeds) | `results/k4_optimizer_grid/20261013_193541/pgd_eps_4.0/{ttc,nac_m0.9,adam,adagrad}/seed_{0,1,2}.log` (TTC/NAC seed 0 from `results/final/pgd_eps_4.0/`); RMSProp: `results/optimizers_stepsize/pgd_eps_4.0/rmsprop/seed_{0,1,2}.log`; family completion (2026-10-14): `results/k4_optimizer_grid/20261014_183047/pgd_eps_4.0/{momentum,rmsprop_bc}/seed_{0,1,2}.log` and `.../pure_la/seed_{1,2}.log` (pure look-ahead seed 0 from `results/ablation_la_k4/`) |
| Cross-architecture (ViT-B/16, three seeds) | `results/cross_arch/pgd_eps_4.0/{ttc,nac_m0.9}/seed_{0,1,2}.log`; STL-10: `results/cross_model/pgd_eps_4.0/*/seed_0.log`; RN50: `results/rn50/` |
| AFT superposition (FARE, three seeds) | `results/aft/pgd_eps_4.0/{ttc,nac_m0.9}/seed_{0,1,2}.log`; TeCoA: `results/run_all.log` (checkpoints `tecoa_vit_b32.pth`, `fare_vit_b32.pth`) |
| Latency benchmark / clean-image evaluation | `results/benchmark_latency/` (incl. `latency_k2_opt_family_*.json`, `latency_k4_opt_family_*.json`, `latency_doc_published_*.json`, `latency_mac_published_*.json`, `latency_mac_matched_*.json`), `results/clean_defense/`; scripts: `scripts/benchmark_latency.py`, `scripts/clean_defense_eval.py` |
| MAC comparison (DTD agreement check) | `results/mac_official_dtd/` + `results/mac_mirror_fixed_dtd/20261013_153216/` |
| MAC published config, CIFAR-10 | `results/mac_port_cifar10/20261013_155859/` |
| MAC budget-matched (diagnostic row) | `results/mac_budget_matched_cifar10/20261013_180352/` |
| MAC two-view matched-budget row (11.84%) | `results/mac_2view_matched_cifar10/20261014_183047/pgd_eps_4.0/mac/seed_0.log` |
| Displacement-matched TTC control (alpha=1.45/255, 14.40%) | `results/ea_ttc_displacement/20261014_183047/pgd_eps_4.0/ttc/seed_0.log` |
| Adaptive attack, first-order pilot (standard-PGD reference column) | `results/adaptive_pilot/` |
| Adaptive attack, EOT over defence initialisation (M=5, three seeds) | `results/adaptive_attack_eot/20261014_202836/eot_results.json`; script: `scripts/adaptive_attack_eot.py` |
| Mechanism controls, look-ahead containment (Table: clamped arms) | `results/clamp_lookahead/` (K=4: `seed_0.log`; K=2: `_archive/seed_0_20261012_184517.log`), 2026-10-12; reference arms: `results/clamp_rerun_20261028/`, 2026-10-28 |
| Mechanism controls, panel B (step-size / displacement) | `results/mu_scan/`, `results/ea_ttc_displacement/`, `results/optimizers_stepsize/` |
| Gradient-alignment diagnostic (K=4, 1024 samples) | `results/mechanism_diagnostic/` |
| N-step scaling | `results/nstep/`, `results/nstep_K4/` |

## Notes

- Seed 0 of the main table is the archived 2026-09-09 run set. Seeds 1--2 exist
  in two independent runs (2026-10-10 and 2026-11-01) under an identical
  protocol; bit-exact reproducibility between the two generations was verified
  twice.
- Batch size affects only the random-initialisation stream of the counterattack
  (the attack and defense are per-sample computations); runs at different batch
  sizes are separate runs and are labeled as such in the manuscript.
- An earlier log-overwriting defect has been fixed: existing logs are archived
  (in `_archive/` subdirectories) rather than replaced.
- An independent fresh-clone replay of this repository (commit `093b8b9`)
  reproduced the archived seed-0 CIFAR-10 TTC-protocol cell of the main table
  to the precision recorded in the archived logs (a fixed-cell spot check with
  locally provided data and cached model weights).
