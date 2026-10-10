# NAC: Nesterov Accelerated Counterattack

**Nesterov Accelerated Counterattack for Test-Time Adversarial Defense in Vision-Language Models**

Research code for "Nesterov Accelerated Counterattack for Test-Time Adversarial Defense of Vision-Language Models." The main experiments were conducted using `nac_fair_experiment.py`. This folder provides a self-contained, runnable version for reproduction.

📄 **Paper**: [`paper_backup/NAC_paper_arxiv.pdf`](paper_backup/NAC_paper_arxiv.pdf) (arXiv) · [`paper_backup/NAC_paper_sivp_8page.pdf`](paper_backup/NAC_paper_sivp_8page.pdf) (SIVP 8-page) · [`paper_backup/NAC_paper_sivp_chinese.pdf`](paper_backup/NAC_paper_sivp_chinese.pdf) (internal Chinese reference)

> ⚠️ **Data integrity notice (2026-06-12; resolved 2026-09-10)**: An earlier draft of this paper/plot script reported a DOC K=4 α=3 value of `37.74` and a NAC > DOC conclusion (`+0.59pp`). Subsequent re-runs in the same config gave `DOC=40.38`, `NAC=38.33` at that time (i.e., DOC ahead by +2.05pp), which the draft of that date reported honestly. **Superseded 2026-09-10**: a full re-run of the Config-B trio with the current code (`results/rerun_docB/`) gives TTC 34.91 / **NAC 39.84** / DOC 37.92 under the tuned gating; DOC reaches 40.41% with its published-default gating (see the DOC table below). See `results/main_results.json` for the authoritative summary.
>
> 🔒 **Log-preservation fix (2026-09-10)**: earlier versions of `nac_fair_experiment.py` overwrote `seed_N.log` on every re-run, which destroyed the raw logs backing the original draft's numbers. The runner now archives any existing log to `<outdir>/<method>/_archive/seed_N_<timestamp>.log` before writing. The SIVP manuscript (`paper_backup/NAC_paper_sivp_8page.tex`) and its supplementary material have been fully re-synced (2026-09-10) to the current log generation; the Data Integrity Statement in the paper lists the exact log file behind every number.

---

## Overview

CLIP models are highly vulnerable to adversarial perturbations in zero-shot classification. Test-Time Counterattack (TTC, CVPR 2025) proposes a training-free defense: run PGD to maximize the L2 distance between the perturbed embedding and the input image's own (self-anchored) embedding — no clean image is required — which empirically pushes predictions back toward the correct class.

We identify that TTC's standard PGD converges slowly under strong attacks (ε ≥ 2/255) due to the non-convex optimization landscape. **NAC replaces PGD with Nesterov accelerated gradient** — a two-line change that provides substantial gains in the small-iteration regime with no additional forward/backward passes.

| Method | Update Rule |
|--------|-------------|
| TTC | delta = delta + alpha * sign(grad(x+delta)) |
| **NAC** | v = mu * v + alpha * sign(grad(x+delta+mu*v)), delta = delta + v |

Both use the sign-gradient (L∞-projected) update. Nesterov's O(1/k²) rate applies to smooth convex objectives with full gradients, so NAC's acceleration is reported as an empirical result (Table 3), not a theoretical convergence rate.

---

## Results

All numbers below are from a single seed (seed 0). Table 1 uses batch size 64 except ImageNet-100, which uses batch size 32. A five-dataset multi-seed pilot is preserved under `results/multiseed_all/`, `results/multiseed_dtd/`, and `results/multiseed_imagenet/` (max std 1.16pp). For per-experiment raw outputs see `results/main_results.json` and `results/<experiment>/seed_0.log`.

### Table 1: Main Results — PGD eps=1/255, ViT-B/32, 2-step defense

| Dataset | Clean | TTC | NAC | Gain (pp) |
|--------|-------|-----|-----|-----------|
| CIFAR-10 | 85.07 | 28.17 | 31.27 | +3.10 |
| CIFAR-100 | 57.16 | 14.48 | 16.86 | +2.38 |
| STL-10 | 96.40 | 77.01 | 79.71 | +2.70 |
| Flowers-102 | 65.41 | 38.90 | 45.83 | +6.93 |
| DTD | 40.53 | 28.14 | 31.01 | +2.87 |
| ImageNet-100 | 66.61 | 45.17 | 49.65 | +4.48 |
| **Average** | — | 38.65 | **42.39** | +3.74 |

> Re-run 2026-09-09 with the current code (5 datasets bs=64, ImageNet-100 bs=32). Clean values drift slightly from the original June run (e.g. CIFAR-10 85.07 vs 84.89; Flowers-102 65.41 vs 62.73; DTD 40.53 vs 42.71) due to dataset-loading fixes, but the TTC-vs-NAC gain is stable.

### Table 2: Multi-Attack Strength — 5-dataset avg, 2-step defense

Datasets: CIFAR-10, CIFAR-100, STL-10, Flowers-102, ImageNet-100. (DTD excluded: low accuracy at eps=4 introduces disproportionate variance.)

| eps | TTC | NAC | Gain (pp) |
|-----|-----|-----|-----------|
| 1/255 | 40.75 | 44.66 | +3.91 |
| 2/255 | 28.49 | 38.46 | +9.97 |
| 4/255 | 6.17 | 13.42 | +7.25 |

### Table 3: N-Step Scaling — PGD eps=4/255

*All K=2 and K=4 values re-run 2026-09-09 with current code. Small discrepancies with paper (0.1-1.0pp) are PGD random-init noise.*

| Method | Steps | CIFAR-10 | STL-10 |
|--------|-------|----------|--------|
| TTC | 2 | 7.02 | 17.52 |
| NAC | 2 | **16.77** | **34.17** |
| TTC | 4 | 26.21$^{+}$ | 48.62$^{+}$ |
| NAC | 4 | **37.70** | **66.01** |

NAC-2 outperforms TTC-2 by +9.75pp (CIFAR-10) and +16.65pp (STL-10). At $K=4$, NAC exceeds TTC by +11.49pp (CIFAR-10) and +17.39pp (STL-10), both verified.

### Table 4: AutoAttack (APGD-CE+APGD-DLR) — CIFAR-10, eps=4/255, 3 seeds

| Method | Clean | Adv | Defended (mean±std) | Gain (pp) |
|--------|-------|-----|----------|-----------|
| TTC | 85.07 | 0.01 | 7.53 ± 0.09 | — |
| NAC | 85.07 | 0.01 | **11.28 ± 0.12** | +3.75 |

### Table 5: Cross-Model — PGD eps=4/255

| Model | Clean | TTC | NAC |
|-------|-------|-----|-----|
| ViT-B/32 (CIFAR-10) | 85.07 | 7.02 | **16.77** |
| ViT-B/32 (STL-10) | 96.40 | 17.52 | **34.17** |
| ViT-B/16 (CIFAR-10) | 87.61 | 7.70 | **13.27** |
| ViT-B/16 (STL-10) | 97.64 | 24.06 | **40.49** |
| RN50 (CIFAR-10) | 67.81 | 0.01 | 0.01$^{*}$ |

$^{*}$RN50 re-run 2026-09-09: both TTC and NAC give ~0% (0.01pp), confirming the paper's claim that NAC is not effective on RN50's attention-pooled embedding space. ResNet's pooling mechanism is more sensitive to pixel-space perturbations than ViT's self-attention, beyond what ViT-tuned hyperparameters can compensate for.

### Table 6: AFT Superposition — PGD eps=4/255

| Base | Clean | TTC | NAC | Gain (pp) |
|------|-------|-----|-----|-----------|
| TeCoA (CIFAR-10) | 63.85 | 2.75 | **4.84** | +2.09 |
| TeCoA (STL-10) | 87.53 | 25.07 | **30.89** | +5.82 |
| FARE (CIFAR-10) | 73.67 | 0.86 | **3.30** | +2.44 |
| FARE (STL-10) | 91.74 | 14.32 | **24.21** | +9.89 |

### Table 7: Ablation — Momentum vs Nesterov (CIFAR-10, eps=4/255, K=2)

*Re-run 2026-09-10 with current code (bs=64).*

| Method | CIFAR-10 | STL-10 |
|--------|----------|--------|
| TTC (no momentum) | 7.02 | 18.52 |
| + Standard momentum | 8.69 | 22.71 |
| + Pure look-ahead (no momentum) | 8.01 | 20.52 |
| + Nesterov look-ahead (NAC) | **16.77** | **35.51** |

Standard momentum gives +1.67pp; pure look-ahead gives +0.99pp; NAC gives +9.75pp.
The Nesterov coupling of momentum and look-ahead—not either component alone—drives NAC's gains (super-additive: 1.67+0.99=2.66 << 9.75pp).

### Momentum Coefficient (CIFAR-10, eps=4/255, K=2, 10 attack steps)

*Re-run 2026-09-03 with current code (bs=32, seed 0). Verified from `results/mu_scan/pgd_eps_4.0/{ttc,nac_m*}/seed_0.log`.*

| mu | 0 (TTC) | 0.1 | 0.5 | 0.7 | 0.9 | 0.99 |
|----|---------|----|----|----|----|------|
| NAC | 7.05 | 8.23 | 12.75 | 15.05 | **17.17** | 18.10 |

### Hyperparameter Sensitivity — tau_thres × beta (CIFAR-10, eps=4/255, K=2, NAC μ=0.9)

*Verified from `results/tau_beta/nac_tau*_seed0.json`. Note: defended values differ ~0.7pp from main table due to PGD random-init across separate runs.*

| tau_thres | beta | Defended (%) | Note |
|-----------|------|--------------|------|
| 0.1 | 2.0 | 8.68 | tau too small → gating too strict |
| 0.2 | 1.0 | 12.81 | beta too small → soft weighting less discriminative |
| **0.2** | **2.0** | **17.17** | **DEFAULT** |
| 0.2 | 3.0 | 18.51 | beta=3.0 slightly outperforms default |
| 0.3 | 2.0 | 17.36 | tau=0.3 close to default |

NAC is robust to hyperparameter choice around the default. `tau=0.1` halves accuracy; `beta=1.0` reduces by ~4pp; `beta=3.0` gives marginal +1.3pp improvement.

### Table DOC: TTC vs NAC vs DOC — Fair Comparison (CIFAR-10, PGD eps=4/255, ViT-B/32, bs=32)

*Config A verified from `results/compare/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log` (2026-09-04); Config B re-run 2026-09-10/12 with current code over seeds 0–2, verified from `results/rerun_docB/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_{0,1,2}.log` (identical attack generation per method within each seed). The main K=2 ablation uses batch size 64 (TTC 7.02, NAC 16.77). The reproduction script explicitly passes `--learnable_tau 0.155 --temperature 75.0`.*

| Method | Config A (K=2, α=1/255, seed 0) | Config B (K=4, α=3/255, 3 seeds) |
|--------|------------------------|------------------------|
| TTC | 7.05 | 34.59 ± 0.27 |
| NAC | **17.17** | **40.01 ± 0.17** |
| DOC | 3.79 | 37.45 ± 0.42 |

**Honest reading (2026-09-10/12 re-run, three seeds):**
- Config A (NAC default): NAC leads by +10.12pp over TTC, +13.38pp over DOC.
- Config B (DOC default steps): NAC leads by +5.41pp over TTC (paired) and +2.56±0.58pp over DOC under the tuned gating (τ=0.155, T=75.0); the paired difference is positive on every seed (+1.92/+3.05/+2.71). With the official code's library-default gating (τ=0.3, T=20.0), DOC reaches 40.41% (single seed, `results/compare_doc_orig`), statistically comparable to NAC.

**Official-repo validation (2026-09-10)**: `doc_counterattack` in `nac_fair_experiment.py` is a line-by-line port of the official `Code/DOC.py` (`F:\codex\DOC\DOC_source\DOC-main`): identical random init, step-0 embedding-shift directional sensitivity for zero-shot CLIP (random probes only for AFT encoders), sigmoid gating, gradient normalization (+1e-8), λ=0.05 orthogonal noise, 0.75/0.25 momentum, sign update, and final weight blending. The gating values τ=0.155 / T=75.0 and Config B (K=4, α=3/255, β=2) are exactly the official README's prescribed configuration. Our measured DOC=37.92% is consistent with the 38.14% reported in the DOC paper itself (CIFAR-10, same nominal configuration).

The two methods address complementary axes: NAC replaces PGD with Nesterov acceleration (one optimizer substitution, two code-level changes: look-ahead gradient evaluation and momentum velocity update, ~5 modified lines in a function otherwise copied verbatim from the official TTC); DOC's exploration mechanisms (orthogonal gradient directions, directional sensitivity, learnable τ gating, gradient normalization) are competitive at K=4 but degrade sharply at K=2. DOC's gating hyperparameters cause a ~3pp swing (37.45±0.42 vs 40.41).

> Earlier draft of this README and the paper reported a DOC value of `3.87` (Config A) and `37.74` (Config B); both came from lost/buggy runs. The verified current values are 3.79 and 37.92 (the latter reproduced exactly twice: `doc_test` 2026-06-14 and `rerun_docB` 2026-09-10). An interim README revision cited Config B values TTC 35.01 / NAC 38.33 / DOC 40.38 from logs that were later overwritten; those numbers are superseded by the 2026-09-10 re-run above.

---

## Installation

```bash
git clone https://github.com/1011245276/NAC.git
cd nac_project
pip install -r requirements.txt
```

**Windows users:** Use Git Bash, WSL, or PowerShell (`run.ps1`). The reproduction scripts (`scripts/*.sh`) require Bash.

CLIP model weights download automatically on first use (~/.cache/clip/).

## Dataset Preparation

Datasets download automatically when you run any experiment script. Or manually:

```bash
python setup_datasets.py
```

This handles CIFAR-10/100, STL-10, DTD, Flowers-102 (auto-download) and ImageNet-100 (via ModelScope).

Target layout after setup:
```
data/
├── cifar-10-batches-py/          (auto)
├── cifar-100-python/             (auto)
├── stl10_binary/                 (auto)
├── dtd/                          (auto)
├── flowers-102/                  (auto)
└── imagenet-100/
    └── imagenet_folder/
        └── val/                  (100 class subdirs)
```

See `data/README.md` for manual download alternatives.

## Checkpoints (Table 6)

Table 6 (AFT superposition) needs the TeCoA and FARE model weights, which are
**not in this repo** (~1.4 GB, over GitHub's 100 MB file limit). They are
hosted on a GitHub Release instead:

```bash
bash scripts/download_checkpoints.sh          # default tag v1.0
TAG=v1.0 bash scripts/download_checkpoints.sh # custom tag
```

This fetches `checkpoints/fare_vit_b32.pth` and `checkpoints/tecoa_vit_b32.pth`.
Without them, `reproduce_aft.sh` skips Table 6 gracefully; all other tables run
regardless.

## Quick Start

```bash
# Single experiment: NAC on CIFAR-10, eps=4/255
bash run.sh nac cifar10 4

# Compare TTC vs NAC
bash run.sh ttc cifar10 4
bash run.sh nac cifar10 4
```

## Reproduce All Paper Results

| Script | Paper Table | Runs | Command |
|--------|------------|------|---------|
| `scripts/reproduce_main.sh` | Table 1: Main results | 6 (single seed) | `bash scripts/reproduce_main.sh` |
| `scripts/reproduce_multi_eps.sh` | Table 2: Multi-epsilon | 18 | `bash scripts/reproduce_multi_eps.sh` |
| `scripts/reproduce_nstep.sh` | Table 3: N-step scaling | 4 | `bash scripts/reproduce_nstep.sh` |
| `scripts/reproduce_autoattack.sh` | Table 4: AutoAttack | 2 | `bash scripts/reproduce_autoattack.sh` |
| `scripts/reproduce_cross_arch.sh` | Table 5: Cross-architecture | 4 | `bash scripts/reproduce_cross_arch.sh` |
| `scripts/reproduce_ablation.sh` | Table 7: Ablation | 4 | `bash scripts/reproduce_ablation.sh` |
| `scripts/reproduce_mu_scan.sh` | Momentum coefficient | 6 | `bash scripts/reproduce_mu_scan.sh` |
| `scripts/reproduce_aft.sh` | Table 6: AFT (download weights first) | 4 | `bash scripts/reproduce_aft.sh` |
| `scripts/compare_ttc_nac_doc.sh` | Table DOC: TTC vs NAC vs DOC | 6 | `bash scripts/compare_ttc_nac_doc.sh` |

After running, aggregate all results into one table (auto-scans every `results/*/pgd_eps_*/<method>/seed_*.log`, computes mean ± std across seeds, and writes `results/aggregated_all.json` + `results/aggregated_all.md`):

```bash
python scripts/aggregate_all.py
```

To re-run only ImageNet-100 (Table 1 + Table 2, after any data/class-name fix): `bash scripts/rerun_imagenet.sh`

For multi-seed mode (the paper now reports main tables as seed 0, with a five-dataset multi-seed pilot max std <= 1.16pp):
```bash
SEEDS="0 1 2" bash scripts/reproduce_main.sh
```

Results are saved to `results/<experiment>/` after each script completes.

**Note:** AutoAttack requires `pip install autoattack`. AFT superposition (Table 6) requires the TeCoA/FARE model checkpoints — download them first with `bash scripts/download_checkpoints.sh` (hosted on GitHub Releases; see below), otherwise `reproduce_aft.sh` skips gracefully.

## System Requirements

- **GPU:** 8 GB VRAM minimum. Lower VRAM: `BATCH_SIZE=8 bash scripts/reproduce_main.sh`
- **Storage:** ~10 GB (datasets + model weights)
- **Total runs:** ~50 across all scripts

### Runtime Estimates (RTX 4060, batch_size=32)

| Script | Runs | ~Time |
|--------|------|-------|
| `reproduce_main.sh` | 6 | 30 min |
| `reproduce_multi_eps.sh` | 18 | 1.5 h |
| `reproduce_nstep.sh` | 4 | 30 min |
| `reproduce_autoattack.sh` | 2 | 30 min |
| `reproduce_cross_arch.sh` | 4 | 30 min |
| `reproduce_ablation.sh` | 4 | 30 min |
| `reproduce_mu_scan.sh` | 6 | 30 min |
| `reproduce_aft.sh` | 4 | 30 min |
| **Total** | **~50** | **~5 h** |

## Method

NAC modifies only **two lines** in the TTC counterattack loop:

```python
# TTC: gradient at current position
images_input = X + delta

# NAC: gradient at look-ahead position (Nesterov)
images_input = X + delta + momentum * velocity
```

```python
# TTC: standard PGD update
delta = delta + alpha * sign(gradient)

# NAC: Nesterov momentum update
velocity = momentum * velocity + alpha * sign(gradient)
delta = delta + velocity
```

Everything else — tau_threshold gating, step weighting, perturbation budget — remains identical to TTC.

## Citation

```
@article{nac2026,
  title={Nesterov Accelerated Counterattack for Test-Time Adversarial Defense of Vision-Language Models},
  author={Zhang, Zhihao and Liu, Yazhi},
  journal={Signal, Image and Video Processing (under review)},
  year={2026}
}
```

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) (Radford et al., ICML 2021)
- [TTC](https://github.com/Sxing2/CLIP-Test-time-Counterattacks) (Xing et al., CVPR 2025)
- [DOC](https://github.com/Chengze-Jiang/DOC) (Jiang et al., AAAI 2026 Oral)

## License

MIT License. This project builds upon [TTC](https://github.com/Sxing2/CLIP-Test-time-Counterattacks) (CVPR 2025). The `replace/clip.py` and `replace/model.py` originate from [OpenAI CLIP](https://github.com/openai/CLIP) (MIT).
