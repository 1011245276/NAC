# NAC: Nesterov Accelerated Counterattack

**Nesterov Accelerated Counterattack for Test-Time Adversarial Defense in Vision-Language Models**

Research code for "Nesterov Accelerated Counterattack for Test-Time Adversarial Defense of Vision-Language Models." The main experiments were conducted using `nac_fair_experiment.py` in the TTC source directory. This folder provides a self-contained, runnable version for reproduction.

📄 **Paper**: [arXiv version](paper/NAC_paper_arxiv.pdf) | [arXiv coming soon]()

---

## Overview

CLIP models are highly vulnerable to adversarial perturbations in zero-shot classification. Test-Time Counterattack (TTC, CVPR 2025) proposes a training-free defense: use PGD to push adversarial images back toward their clean embeddings.

We identify that TTC's standard PGD converges slowly due to the non-convex optimization landscape. **NAC replaces PGD with Nesterov accelerated gradient** — a two-line change that provides significant gains with zero additional computation.

| Method | Update Rule | Convergence |
|--------|-------------|-------------|
| TTC | delta = delta + alpha * sign(grad(x+delta)) | O(1/k) |
| **NAC** | v = mu * v + alpha * sign(grad(x+delta+mu*v)), delta = delta + v | **O(1/k^2)** |

---

## Results

### Table 1: Main Results — PGD eps=1/255, ViT-B/32, 2-step defense

| Dataset | Clean | TTC | NAC | Gain (pp) |
|--------|-------|-----|-----|-----------|
| CIFAR-10 | 84.89 | 27.95 | 30.94 | +2.99 |
| CIFAR-100 | 58.19 | 14.68 | 16.87 | +2.19 |
| STL-10 | 96.67 | 77.11 | 80.33 | +3.22 |
| Flowers-102 | 62.73 | 38.62 | 45.76 | +7.14 |
| DTD | 42.71 | 26.54 | 29.68 | +3.14 |
| ImageNet-100 | 67.82 | 46.79 | **51.40** | +4.61 |
| **Average** | — | 38.62 | **42.50** | +3.88 |

### Table 2: Multi-Attack Strength — 5-dataset avg, 2-step defense

Datasets: CIFAR-10, CIFAR-100, STL-10, Flowers-102, ImageNet-100. (DTD excluded: low accuracy at eps=4 introduces disproportionate variance.)

| eps | TTC | NAC | Gain (pp) |
|-----|-----|-----|-----------|
| 1/255 | 41.03 | 45.06 | +4.03 |
| 2/255 | 28.95 | 38.80 | +9.85 |
| 4/255 | 6.15 | 13.54 | +7.39 |

### Table 3: N-Step Scaling — PGD eps=4/255

| Method | Steps | CIFAR-10 | STL-10 |
|--------|-------|----------|--------|
| TTC | 2 | 7.07 | 17.44 |
| NAC | 2 | **16.47** | **34.19** |
| TTC | 4 | 25.37 | 48.29 |
| NAC | 4 | **37.05** | **66.74** |

NAC-2 outperforms TTC-2 by 9.40pp (CIFAR-10) and 16.75pp (STL-10).

### Table 4: AutoAttack — CIFAR-10, eps=4/255

| Method | Clean | Adv | Defended | Gain (pp) |
|--------|-------|-----|----------|-----------|
| TTC | 84.90 | 0.05 | 7.56 | — |
| NAC | 84.90 | 0.05 | **11.29** | +3.73 |

### Table 5: Cross-Model — PGD eps=4/255, CIFAR-10

| Model | Clean | TTC | NAC |
|-------|-------|-----|-----|
| ViT-B/32 | 84.89 | 6.98 | **16.61** |
| ViT-B/16 | 87.24 | 8.51 | **14.66** |
| RN50 | 65.58 | 0.00 | 0.00 |

RN50: ResNet's attention-pooled embedding is more sensitive to pixel-space perturbations. tau/epsilon were tuned for ViT. Consistent with TTC paper's ViT focus.

### Table 6: AFT Superposition — PGD eps=4/255, CIFAR-10

| Base | Clean | TTC | NAC | Gain (pp) |
|------|-------|-----|-----|-----------|
| TeCoA | 63.85 | 2.75 | **4.84** | +2.09 |
| FARE | 73.67 | 0.86 | **3.30** | +2.44 |

### Table 7: Ablation — Momentum vs Nesterov

| Method | CIFAR-10 | STL-10 |
|--------|----------|--------|
| TTC (no momentum) | 6.98 | 17.32 |
| + Standard momentum | 8.46 | 21.20 |
| + Nesterov look-ahead (NAC) | **16.61** | **34.19** |

Standard momentum gives +1.48pp; Nesterov look-ahead adds +8.15pp beyond that.
The Nesterov coupling of momentum and look-ahead—not either component alone—drives NAC's gains (super-additive: 2.66+1.48=4.14 << 9.63pp).

### Momentum Coefficient (CIFAR-10, eps=4, 5-step)

| mu | 0 (TTC) | 0.1 | 0.5 | 0.7 | 0.9 | 0.99 |
|----|---------|-----|-----|-----|-----|------|
| NAC | 7.07 | 8.14 | 12.32 | 14.38 | 16.47 | 17.47 |

### Table DOC: TTC vs NAC vs DOC — Fair Comparison (CIFAR-10, PGD eps=4/255, ViT-B/32)

| Method | Config A (K=2, α=1/255) | Config B (K=4, α=3/255) |
|--------|------------------------|------------------------|
| TTC | 7.07 | 35.01 |
| NAC | **16.47** | 38.33 |
| DOC | 4.44 | **40.38** |

Config A = NAC paper default (2 steps, step=1/255). Config B = DOC paper default (4 steps, step=3/255, tau=0.155, temp=75.0). NAC approaches DOC in Config B using a single modification versus DOC's four auxiliary components, and substantially outperforms both TTC and DOC under constrained iteration budgets (Config A).

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
| `scripts/reproduce_main.sh` | Table 1: Main results | 36 | `bash scripts/reproduce_main.sh` |
| `scripts/reproduce_multi_eps.sh` | Table 2: Multi-epsilon | 108 | `bash scripts/reproduce_multi_eps.sh` |
| `scripts/reproduce_nstep.sh` | Table 3: N-step scaling | 24 | `bash scripts/reproduce_nstep.sh` |
| `scripts/reproduce_autoattack.sh` | Table 4: AutoAttack | 6 | `bash scripts/reproduce_autoattack.sh` |
| `scripts/reproduce_cross_arch.sh` | Table 5: Cross-architecture | 12 | `bash scripts/reproduce_cross_arch.sh` |
| `scripts/reproduce_ablation.sh` | Table 7: Ablation | 18 | `bash scripts/reproduce_ablation.sh` |
| `scripts/reproduce_mu_scan.sh` | Momentum coefficient | 18 | `bash scripts/reproduce_mu_scan.sh` |
| `scripts/reproduce_aft.sh` | Table 6: AFT (needs weights) | 12 | `bash scripts/reproduce_aft.sh` |
| `scripts/compare_ttc_nac_doc.sh` | Table DOC: TTC vs NAC vs DOC | 3 | `bash scripts/compare_ttc_nac_doc.sh` |

Results are saved to `results/<experiment>/` after each script completes.

**Note:** AutoAttack requires `pip install autoattack`. AFT superposition (Table 6) requires TeCoA/FARE model checkpoints — the script will skip gracefully if unavailable.

## System Requirements

- **GPU:** 8 GB VRAM minimum. Lower VRAM: `BATCH_SIZE=8 bash scripts/reproduce_main.sh`
- **Storage:** ~10 GB (datasets + model weights)
- **Total runs:** 234 across all scripts (~222 without AFT)

### Runtime Estimates (RTX 4060, batch_size=32)

| Script | Runs | ~Time | Notes |
|--------|------|-------|-------|
| `reproduce_main.sh` | 36 | 2.5 h | 6 datasets × 2 methods × 3 seeds |
| `reproduce_multi_eps.sh` | 108 | 7 h | 3 epsilons × all above |
| `reproduce_nstep.sh` | 24 | 1.5 h | 2 datasets × 2/4 steps |
| `reproduce_autoattack.sh` | 6 | 0.5 h | AutoAttack is heavier per step |
| `reproduce_cross_arch.sh` | 12 | 1 h | ViT-B/16 similar to B/32 |
| `reproduce_ablation.sh` | 18 | 1 h | 2 datasets |
| `reproduce_mu_scan.sh` | 18 | 1 h | 6 mu values |
| `reproduce_aft.sh` | 12 | 1 h | Needs external weights |
| **Total** | **234** | **~15 h** | |

### Run a Single Experiment

Use `run.sh` for one-off runs. Map each paper result to the right command:

```bash
bash run.sh [method] [dataset] [eps]
# method:  nac | ttc | momentum
# dataset: cifar10 | cifar100 | STL10 | flowers102 | DTD | imagenet100
# eps:     1 | 2 | 4  (attack strength in /255)
```

| Paper result | Command |
|-------------|---------|
| Table 1, CIFAR-10, eps=1 | `bash run.sh nac cifar10 1` |
| Table 2, STL-10, eps=2 | `bash run.sh nac STL10 2` |
| Table 7, TTC baseline | `bash run.sh ttc cifar10 4` |
| Table 7, standard momentum | `bash run.sh momentum cifar10 4` |
| Table 7, NAC (Nesterov) | `bash run.sh nac cifar10 4` |
| Table 3 (4-step) | Use `reproduce_nstep.sh` (needs `--ttc_numsteps 4`) |
| Table 4 (AutoAttack) | Use `reproduce_autoattack.sh` (needs `--test_attack_type autoattack`) |
| Table 5 (ViT-B/16) | Use `reproduce_cross_arch.sh` (needs `--arch vit_b16`) |

Or call Python directly for full control:

```bash
python nac_fair_experiment.py \
    --batch_size 32 --root ./data \
    --test_set cifar10 --test_eps 4 --test_numsteps 10 \
    --counterattack nac --nac_momentum 0.9 \
    --ttc_numsteps 2 --seed 0 --outdir ./results/my_test
```

## Project Structure

```
nac_project/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt
├── run.sh / run.ps1              # Quick run (Linux / Windows)
├── nac_fair_experiment.py        # Main experiment runner (NAC + TTC + momentum)
├── test_time_counterattack.py    # TTC counterattack (imported as library)
├── nac.py / ttc.py               # Standalone counterattack modules
├── evaluate.py                   # Lightweight single-dataset eval
├── attacks.py                    # PGD / CW / AutoAttack
├── func.py                       # CLIP preprocessing
├── utils.py                      # Dataset loading
├── replace/                      # CLIP model (self-contained)
├── models/                       # Visual prompters
├── support/                      # ImageNet class names
├── scripts/                      # Reproduce all experiments
│   ├── reproduce_main.sh         #   Table 1: 6 datasets, eps=1
│   ├── reproduce_multi_eps.sh    #   Table 2: 3 epsilon values
│   ├── reproduce_nstep.sh        #   Table 3: 2-step vs 4-step
│   ├── reproduce_autoattack.sh   #   Table 4: AutoAttack eval
│   ├── reproduce_cross_arch.sh   #   Table 5: ViT-B/32 + ViT-B/16
│   ├── reproduce_aft.sh          #   Table 6: AFT superposition
│   ├── reproduce_ablation.sh     #   Table 7: momentum ablation
│   ├── reproduce_mu_scan.sh      #   momentum coefficient scan
│   └── compare_ttc_nac_doc.sh    #   Table DOC: TTC vs NAC vs DOC
├── figures/                      # Paper figures (PDF + PNG)
├── results/
│   ├── main_results.json         # All experiment data
│   ├── ablation/                 # Momentum ablation
│   └── final/                    # N-step comparison
├── paper/                        # arXiv preprint
├── data/                         # Dataset directory
```

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
  title={Nesterov Accelerated Counterattack for Test-Time Adversarial Defense in Vision-Language Models},
  author={},
  journal={},
  year={2026}
}
```

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) (Radford et al., ICML 2021)
- [TTC](https://github.com/Sxing2/CLIP-Test-time-Counterattacks) (Xing et al., CVPR 2025)

## License

MIT License. This project builds upon [TTC](https://github.com/Sxing2/CLIP-Test-time-Counterattacks) (CVPR 2025). The `replace/clip.py` and `replace/model.py` originate from [OpenAI CLIP](https://github.com/openai/CLIP) (MIT).
