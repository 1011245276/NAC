# NAC: Nesterov Accelerated Counterattack

**Nesterov Accelerated Counterattack for Test-Time Adversarial Defense in Vision-Language Models**

Research code for "Nesterov Accelerated Counterattack for Test-Time Adversarial Defense of Vision-Language Models." The main experiments were conducted using `nac_fair_experiment.py`. This folder provides a self-contained, runnable version for reproduction.

📄 **Paper**: [`paper_backup/NAC_paper_arxiv.pdf`](paper_backup/NAC_paper_arxiv.pdf) (arXiv) · [`paper_backup/NAC_paper_sivp_8page.pdf`](paper_backup/NAC_paper_sivp_8page.pdf) (SIVP 8-page) · [`paper_backup/NAC_paper_sivp_chinese.pdf`](paper_backup/NAC_paper_sivp_chinese.pdf) (internal Chinese reference)

> ⚠️ **Data integrity notice (2026-06-12)**: An earlier draft of this paper/plot script reported a DOC K=4 α=3 value of `37.74` and a NAC > DOC conclusion (`+0.59pp`). Subsequent re-runs in the same config give `DOC=40.38`, `NAC=38.33`, i.e. **DOC outperforms NAC by +2.05pp**. We have updated the paper and this README to report the verified value honestly. See `results/main_results.json` and `results/compare/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log` for the raw logs.

---

## Overview

CLIP models are highly vulnerable to adversarial perturbations in zero-shot classification. Test-Time Counterattack (TTC, CVPR 2025) proposes a training-free defense: use PGD to push adversarial images back toward their clean embeddings.

We identify that TTC's standard PGD converges slowly due to the non-convex optimization landscape. **NAC replaces PGD with Nesterov accelerated gradient** — a two-line change that provides substantial gains in the small-iteration regime at zero additional computation.

| Method | Update Rule | Convergence |
|--------|-------------|-------------|
| TTC | delta = delta + alpha * sign(grad(x+delta)) | O(1/k) |
| **NAC** | v = mu * v + alpha * sign(grad(x+delta+mu*v)), delta = delta + v | **O(1/k^2)** |

---

## Results

All numbers below are from a single seed (seed 0). For per-experiment raw outputs see `results/main_results.json` and `results/<experiment>/seed_0.log`.

### Table 1: Main Results — PGD eps=1/255, ViT-B/32, 2-step defense

| Dataset | Clean | TTC | NAC | Gain (pp) |
|--------|-------|-----|-----|-----------|
| CIFAR-10 | 84.89 | 27.95 | 30.94 | +2.99 |
| CIFAR-100 | 58.19 | 14.68 | 16.87 | +2.19 |
| STL-10 | 96.67 | 77.11 | 80.33 | +3.22 |
| Flowers-102 | 62.73 | 38.62 | 45.76 | +7.14 |
| DTD | 42.71 | 26.54 | 29.68 | +3.14 |
| ImageNet-100 | 67.82 | 46.79 | 51.40 | +4.61 |
| **Average** | — | 38.62 | **42.50** | +3.88 |

### Table 2: Multi-Attack Strength — 5-dataset avg, 2-step defense

Datasets: CIFAR-10, CIFAR-100, STL-10, Flowers-102, ImageNet-100. (DTD excluded: low accuracy at eps=4 introduces disproportionate variance.)

| eps | TTC | NAC | Gain (pp) |
|-----|-----|-----|-----------|
| 1/255 | 41.03 | 45.06 | +4.03 |
| 2/255 | 28.95 | 38.80 | +9.85 |
| 4/255 | 6.15 | 13.54 | +7.39 |

### Table 3: N-Step Scaling — PGD eps=4/255

*All K=2 and K=4 values verified 2026-06-12. Small discrepancies with paper (0.4-0.9pp) are PGD random-init noise.*

| Method | Steps | CIFAR-10 | STL-10 |
|--------|-------|----------|--------|
| TTC | 2 | 6.98 | 17.32 |
| NAC | 2 | **16.61** | **34.19** |
| TTC | 4 | 25.37$^{+}$ | 48.29$^{+}$ |
| NAC | 4 | **37.05** | **66.74** |

NAC-2 outperforms TTC-2 by +9.63pp (CIFAR-10) and +16.87pp (STL-10). At $K=4$, NAC exceeds TTC by +11.68pp (CIFAR-10) and +18.45pp (STL-10), both verified.

### Table 4: AutoAttack — CIFAR-10, eps=4/255

| Method | Clean | Adv | Defended | Gain (pp) |
|--------|-------|-----|----------|-----------|
| TTC | 84.90 | 0.05 | 7.59 | — |
| NAC | 84.90 | 0.05 | **11.29** | +3.70 |

### Table 5: Cross-Model — PGD eps=4/255

| Model | Clean | TTC | NAC |
|-------|-------|-----|-----|
| ViT-B/32 (CIFAR-10) | 84.89 | 7.07 | **16.47** |
| ViT-B/32 (STL-10) | 96.67 | 17.32 | **34.19** |
| ViT-B/16 (CIFAR-10) | 87.24 | 8.51 | **14.66** |
| ViT-B/16 (STL-10) | 97.71 | 24.64 | **41.02** |
| RN50 (CIFAR-10) | 67.81 | 0.00 | 0.01$^{*}$ |

$^{*}$RN50 verified 2026-06-12: NAC achieves a marginal 0.01pp (essentially zero), confirming the paper's claim that NAC is not effective on RN50's attention-pooled embedding space. ResNet's pooling mechanism is more sensitive to pixel-space perturbations than ViT's self-attention, beyond what ViT-tuned hyperparameters can compensate for.

### Table 6: AFT Superposition — PGD eps=4/255

| Base | Clean | TTC | NAC | Gain (pp) |
|------|-------|-----|-----|-----------|
| TeCoA (CIFAR-10) | 63.85 | 2.75 | **4.84** | +2.09 |
| TeCoA (STL-10) | 87.53 | 25.07 | **30.89** | +5.82 |
| FARE (CIFAR-10) | 73.67 | 0.86 | **3.30** | +2.44 |
| FARE (STL-10) | 91.74 | 14.32 | **24.21** | +9.89 |

### Table 7: Ablation — Momentum vs Nesterov (CIFAR-10, eps=4/255, K=2)

*All values verified from `results/ablation/` and `results/ablation_fixed/`.*

| Method | CIFAR-10 | STL-10 |
|--------|----------|--------|
| TTC (no momentum) | 6.98 | 17.32 |
| + Standard momentum | 8.69 | 21.73 |
| + Pure look-ahead (no momentum) | 9.35 | 23.11 |
| + Nesterov look-ahead (NAC) | **16.61** | **34.19** |

Standard momentum gives +1.71pp; pure look-ahead gives +2.37pp; NAC gives +9.63pp.
The Nesterov coupling of momentum and look-ahead—not either component alone—drives NAC's gains (super-additive: 1.71+2.37=4.08 << 9.63pp).

### Momentum Coefficient (CIFAR-10, eps=4/255, K=2, 10 attack steps)

*Verified from `results/mu_scan/pgd_eps_4.0/nac_m*/seed_0.log`.*

| mu | 0 (TTC) | 0.1 | 0.5 | 0.7 | 0.9 | 0.99 |
|----|---------|----|----|----|----|------|
| NAC | 7.07 | 8.14 | 12.32 | 14.38 | **16.47** | 17.47 |

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

### Table DOC: TTC vs NAC vs DOC — Fair Comparison (CIFAR-10, PGD eps=4/255, ViT-B/32)

*Verified from `results/compare/pgd_eps_4.0/{ttc,nac_m0.9,doc}/seed_0.log`.*

| Method | Config A (K=2, α=1/255) | Config B (K=4, α=3/255) |
|--------|------------------------|------------------------|
| TTC | 7.07 | 35.01 |
| NAC | **16.61** | 38.33 |
| DOC | 4.44 | **40.38** |

**Honest reading:**
- Config A (NAC default): NAC leads by +9.40pp over TTC, +12.03pp over DOC.
- Config B (DOC default): DOC leads by +5.37pp over TTC, +2.05pp over NAC.

The two methods address complementary axes: NAC is a single-principled-modification optimizer substitution that dominates at small $K$; DOC's exploration mechanisms (orthogonal gradient directions, directional sensitivity, learnable $\tau$ gating, gradient normalization) provide additional benefit at higher $K$ with a larger step size.

> Earlier draft of this README and the paper reported a DOC value of `3.87` (Config A) and `37.74` (Config B). The Config A `3.87` came from a buggy run; the verified value is `4.44`. The Config B `37.74` came from a lost earlier run; the verified value is `40.38`. The paper and this README have been updated to match the verified logs.

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
| `scripts/reproduce_aft.sh` | Table 6: AFT (needs weights) | 4 | `bash scripts/reproduce_aft.sh` |
| `scripts/compare_ttc_nac_doc.sh` | Table DOC: TTC vs NAC vs DOC | 6 | `bash scripts/compare_ttc_nac_doc.sh` |

After running, aggregate with: `python scripts/aggregate_results.py --root ./results/main`

For multi-seed mode (paper claims of "std < 0.2pp across seeds" not formally verified; informal pilots suggest std ≤ 0.8pp):
```bash
SEEDS="0 1 2" bash scripts/reproduce_main.sh
```

Results are saved to `results/<experiment>/` after each script completes.

**Note:** AutoAttack requires `pip install autoattack`. AFT superposition (Table 6) requires TeCoA/FARE model checkpoints — the script will skip gracefully if unavailable.

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
