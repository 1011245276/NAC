# Aggregated Results


## ablation

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| momentum | K=2 atk=10 bs=32 momentum | STL10 | 3 | 96.40 | 0.05 | 21.54 ± 0.37 | +21.49 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | STL10 | 3 | 96.40 | 0.05 | 34.38 ± 0.43 | +34.33 |
| pure_la | K=2 atk=10 bs=32 pure_la | STL10 | 1 | 96.67 | 0.00 | 22.68 | +22.68 |
| pure_la | K=2 atk=10 bs=32 pure_la | cifar10 | 1 | 84.89 | 0.00 | 9.64 | +9.64 |
| pure_la | K=2 atk=10 bs=64 pure_la | STL10 | 1 | 96.67 | 0.00 | 23.11 | +23.11 |
| pure_la | K=2 atk=10 bs=64 pure_la | cifar10 | 1 | 84.89 | 0.00 | 9.35 | +9.35 |
| pure_la | K=4 atk=10 bs=32 pure_la | cifar10 | 1 | 84.89 | 0.00 | 14.54 | +14.54 |
| pure_la | K=4 atk=10 bs=64 pure_la | STL10 | 1 | 96.67 | 0.00 | 32.89 | +32.89 |
| pure_la | K=4 atk=10 bs=64 pure_la | cifar10 | 1 | 84.89 | 0.00 | 14.53 | +14.53 |
| ttc | K=2 atk=10 bs=32 ttc | STL10 | 3 | 96.40 | 0.05 | 17.35 ± 0.03 | +17.30 |

## ablation_fixed

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| momentum | K=2 atk=10 bs=32 momentum | STL10 | 1 | 96.40 | 0.01 | 21.73 | +21.72 |
| momentum | K=2 atk=10 bs=32 momentum | cifar10 | 1 | 85.07 | 0.00 | 8.69 | +8.69 |

## aft

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 3 | 74.42 | 0.01 | 3.41 ± 0.04 | +3.40 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 3 | 74.42 | 0.01 | 0.99 ± 0.06 | +0.98 |

## compare

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| doc | K=2 atk=10 bs=32 doc | cifar10 | 1 | 85.07 | 0.00 | 3.79 | +3.79 |
| doc | K=4 atk=10 bs=16 doc | cifar10 | 1 | 84.90 | 0.00 | 38.81 | +38.81 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 1 | 85.07 | 0.00 | 17.17 | +17.17 |
| nac_m0.9 | K=4 atk=10 bs=16 nac m=0.9 | cifar10 | 1 | 84.90 | 0.00 | 38.72 | +38.72 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 1 | 85.07 | 0.00 | 7.05 | +7.05 |
| ttc | K=4 atk=10 bs=16 ttc | cifar10 | 1 | 84.90 | 0.00 | 34.64 | +34.64 |

## compare_doc_orig

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| doc | K=4 atk=10 bs=32 doc | cifar10 | 1 | 85.07 | 0.00 | 40.41 | +40.41 |

## cross_arch

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 3 | 87.61 | 0.00 | 13.65 ± 0.33 | +13.65 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 3 | 87.61 | 0.00 | 7.71 ± 0.11 | +7.71 |

## cross_model

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | STL10 | 1 | 97.71 | 0.00 | 41.02 | +41.02 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 1 | 87.24 | 0.00 | 14.66 | +14.66 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 92.79 | 0.01 | 0.29 | +0.28 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 65.58 | 0.00 | 0.00 | +0.00 |
| ttc | K=2 atk=10 bs=32 ttc | STL10 | 1 | 97.71 | 0.00 | 24.64 | +24.64 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 1 | 87.24 | 0.00 | 8.51 | +8.51 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 92.79 | 0.01 | 0.23 | +0.22 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 65.58 | 0.00 | 0.00 | +0.00 |

## doc_test

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| doc | K=4 atk=10 bs=32 doc | cifar10 | 1 | 85.07 | 0.00 | 37.92 | +37.92 |

## eps1

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 66.61 | 2.34 | 49.65 | +47.31 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 2 | 1.06 | 0.01 | 1.00 ± 0.03 | +0.99 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 66.61 | 2.34 | 45.17 | +42.83 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 2 | 1.06 | 0.01 | 0.80 ± 0.02 | +0.79 |

## eps2

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 0.42 | 73.76 | +73.34 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 39.16 | +39.16 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 1 | 58.19 | 0.00 | 18.14 | +18.14 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 1 | 42.71 | 0.05 | 18.88 | +18.83 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 1 | 62.73 | 0.00 | 26.88 | +26.88 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.42 | 61.64 | +61.22 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 29.17 | +29.17 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 1 | 58.19 | 0.00 | 12.42 | +12.42 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 1 | 42.71 | 0.05 | 13.24 | +13.19 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 1 | 62.73 | 0.00 | 15.89 | +15.89 |

## eps4

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 0.01 | 34.58 | +34.57 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 16.61 | +16.61 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 1 | 58.19 | 0.00 | 5.66 | +5.66 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 1 | 42.71 | 0.05 | 4.79 | +4.74 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 1 | 62.73 | 0.00 | 2.76 | +2.76 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.01 | 17.84 | +17.83 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 6.98 | +6.98 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 1 | 58.19 | 0.00 | 2.02 | +2.02 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 1 | 42.71 | 0.05 | 2.82 | +2.77 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 1 | 62.73 | 0.00 | 0.67 | +0.67 |

## final

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=4 atk=10 bs=32 nac m=0.9 | STL10 | 1 | 96.67 | 11.12 | 73.70 | +62.58 |
| nac_m0.9 | K=4 atk=10 bs=32 nac m=0.9 | cifar10 | 1 | 84.89 | 0.82 | 27.39 | +26.57 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | STL10 | 1 | 96.67 | 0.00 | 34.30 | +34.30 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 16.47 | +16.47 |
| nac_m0.9 | K=4 atk=10 bs=32 nac m=0.9 | STL10 | 1 | 96.67 | 0.00 | 66.74 | +66.74 |
| nac_m0.9 | K=4 atk=10 bs=32 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 37.05 | +37.05 |
| ttc | K=4 atk=10 bs=32 ttc | STL10 | 1 | 96.67 | 11.12 | 73.66 | +62.54 |
| ttc | K=4 atk=10 bs=32 ttc | cifar10 | 1 | 84.89 | 0.82 | 27.20 | +26.38 |
| ttc | K=4 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.00 | 48.29 | +48.29 |
| ttc | K=4 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 25.37 | +25.37 |

## imagenet

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | ImageNet | 1 | 67.82 | 0.01 | 7.99 | +7.98 |
| ttc | K=2 atk=10 bs=64 ttc | ImageNet | 1 | 67.82 | 0.01 | 3.68 | +3.67 |

## imagenet_full

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 67.82 | 2.32 | 51.40 | +49.08 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 67.82 | 0.09 | 36.07 | +35.98 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 67.82 | 0.02 | 8.09 | +8.07 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 67.82 | 2.32 | 46.79 | +44.47 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 67.82 | 0.09 | 25.64 | +25.55 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 67.82 | 0.02 | 3.25 | +3.23 |

## momentum_eps1

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| momentum | K=2 atk=10 bs=64 momentum | STL10 | 1 | 96.40 | 11.01 | 77.64 | +66.63 |
| momentum | K=2 atk=10 bs=64 momentum | cifar10 | 1 | 85.07 | 0.77 | 29.09 | +28.32 |

## mu_scan

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.0 | K=2 atk=10 bs=32 nac m=0.0 | cifar10 | 1 | 84.89 | 0.00 | 7.07 | +7.07 |
| nac_m0.0 | K=5 atk=10 bs=32 nac m=0.0 | cifar10 | 1 | 84.89 | 0.00 | 28.19 | +28.19 |
| nac_m0.0 | K=2 atk=5 bs=32 nac m=0.0 | cifar10 | 1 | 84.89 | 0.02 | 9.95 | +9.93 |
| nac_m0.1 | K=2 atk=10 bs=32 nac m=0.1 | cifar10 | 3 | 85.07 | 0.00 | 8.10 ± 0.15 | +8.10 |
| nac_m0.5 | K=2 atk=10 bs=32 nac m=0.5 | cifar10 | 3 | 85.07 | 0.00 | 12.78 ± 0.22 | +12.78 |
| nac_m0.7 | K=2 atk=10 bs=32 nac m=0.7 | cifar10 | 3 | 85.07 | 0.00 | 15.05 ± 0.07 | +15.05 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 3 | 85.07 | 0.00 | 17.15 ± 0.21 | +17.15 |
| nac_m0.99 | K=2 atk=10 bs=32 nac m=0.99 | cifar10 | 3 | 85.07 | 0.00 | 18.08 ± 0.24 | +18.08 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 3 | 85.07 | 0.00 | 6.88 ± 0.15 | +6.88 |

## multi_eps

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 66.61 | 2.34 | 49.65 | +47.31 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | DTD | 2 | 40.53 | 2.98 | 30.02 ± 0.94 | +27.04 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 66.61 | 0.10 | 35.25 | +35.15 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | DTD | 2 | 40.53 | 0.27 | 19.62 ± 0.30 | +19.36 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | ImageNet | 1 | 66.61 | 0.02 | 7.94 | +7.92 |
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | DTD | 2 | 40.53 | 0.11 | 4.52 ± 0.07 | +4.41 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 66.61 | 2.34 | 45.17 | +42.83 |
| ttc | K=2 atk=10 bs=32 ttc | DTD | 2 | 40.53 | 2.98 | 27.61 ± 0.08 | +24.62 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 66.61 | 0.10 | 25.40 | +25.30 |
| ttc | K=2 atk=10 bs=32 ttc | DTD | 2 | 40.53 | 0.27 | 14.41 ± 0.45 | +14.14 |
| ttc | K=2 atk=10 bs=32 ttc | ImageNet | 1 | 66.61 | 0.02 | 3.53 | +3.51 |
| ttc | K=2 atk=10 bs=32 ttc | DTD | 2 | 40.53 | 0.11 | 2.40 ± 0.08 | +2.29 |

## multiseed

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | cifar10 | 2 | 85.07 | 0.85 | 31.20 ± 0.23 | +30.35 |
| ttc | K=2 atk=10 bs=32 ttc | cifar10 | 2 | 85.07 | 0.85 | 28.51 ± 0.42 | +27.66 |

## multiseed_all

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 3 | 96.40 | 11.01 | 79.42 ± 0.19 | +68.41 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 3 | 57.16 | 0.23 | 16.43 ± 0.19 | +16.20 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 3 | 65.41 | 1.06 | 45.73 ± 0.43 | +44.67 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 2 | 96.40 | 11.01 | 76.75 ± 0.11 | +65.74 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 2 | 57.16 | 0.23 | 14.26 ± 0.16 | +14.03 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 2 | 65.41 | 1.06 | 38.91 ± 0.22 | +37.84 |

## multiseed_dtd

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 3 | 40.53 | 2.82 | 30.10 ± 0.80 | +27.28 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 3 | 40.53 | 2.82 | 26.67 ± 1.16 | +23.85 |

## multiseed_imagenet

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | ImageNet | 3 | 67.51 | 2.37 | 50.77 ± 0.63 | +48.40 |
| ttc | K=2 atk=10 bs=64 ttc | ImageNet | 3 | 67.51 | 2.37 | 46.38 ± 0.62 | +44.01 |

## nstep

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=4 atk=10 bs=32 nac m=0.9 | STL10 | 3 | 96.40 | 0.05 | 66.57 ± 0.21 | +66.52 |
| ttc | K=4 atk=10 bs=32 ttc | STL10 | 3 | 96.40 | 0.05 | 48.35 ± 0.31 | +48.30 |

## nstep_K4

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=4 atk=10 bs=16 nac m=0.9 | STL10 | 1 | 96.41 | 0.05 | 66.34 | +66.29 |
| ttc | K=4 atk=10 bs=16 ttc | STL10 | 1 | 96.41 | 0.05 | 49.16 | +49.11 |

## optimizers

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| adam | K=2 atk=10 bs=32 adam | cifar10 | 1 | 85.07 | 0.00 | 5.38 | +5.38 |

## rn50

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=16 nac m=0.9 | cifar10 | 1 | 67.81 | 0.00 | 0.01 | +0.01 |

## seed1

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 10.70 | 79.66 | +68.96 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.84 | 31.40 | +30.56 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 1 | 58.19 | 0.24 | 17.18 | +16.94 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 1 | 42.71 | 3.30 | 30.37 | +27.07 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 1 | 62.73 | 1.09 | 45.97 | +44.88 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 0.39 | 73.99 | +73.60 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 39.54 | +39.54 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 1 | 58.19 | 0.00 | 17.91 | +17.91 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 1 | 42.71 | 0.11 | 19.57 | +19.46 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 1 | 62.73 | 0.00 | 27.34 | +27.34 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 0.01 | 34.49 | +34.48 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 16.87 | +16.87 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar100 | 1 | 58.19 | 0.00 | 6.12 | +6.12 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | dtd | 1 | 42.71 | 0.00 | 4.68 | +4.68 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | flowers102 | 1 | 62.73 | 0.00 | 2.60 | +2.60 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 10.70 | 76.62 | +65.92 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.84 | 28.46 | +27.62 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 1 | 58.19 | 0.24 | 15.23 | +14.99 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 1 | 42.71 | 3.30 | 28.09 | +24.79 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 1 | 62.73 | 1.09 | 38.17 | +37.08 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.39 | 61.39 | +61.00 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 28.67 | +28.67 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 1 | 58.19 | 0.00 | 12.26 | +12.26 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 1 | 42.71 | 0.11 | 14.52 | +14.41 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 1 | 62.73 | 0.00 | 15.84 | +15.84 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.01 | 17.59 | +17.58 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 7.01 | +7.01 |
| ttc | K=2 atk=10 bs=64 ttc | cifar100 | 1 | 58.19 | 0.00 | 2.40 | +2.40 |
| ttc | K=2 atk=10 bs=64 ttc | dtd | 1 | 42.71 | 0.00 | 2.55 | +2.55 |
| ttc | K=2 atk=10 bs=64 ttc | flowers102 | 1 | 62.73 | 0.00 | 0.67 | +0.67 |

## seed2

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | STL10 | 1 | 96.67 | 0.00 | 34.06 | +34.06 |
| nac_m0.9 | K=2 atk=10 bs=64 nac m=0.9 | cifar10 | 1 | 84.89 | 0.00 | 16.54 | +16.54 |
| ttc | K=2 atk=10 bs=64 ttc | STL10 | 1 | 96.67 | 0.00 | 17.23 | +17.23 |
| ttc | K=2 atk=10 bs=64 ttc | cifar10 | 1 | 84.89 | 0.00 | 7.13 | +7.13 |

## sun397

| Method | Config | Dataset | Seeds | Clean | Adv | Defended | Gain |
|--------|--------|---------|-------|-------|-----|----------|------|
| nac_m0.9 | K=2 atk=10 bs=32 nac m=0.9 | SUN397 | 1 | 70.80 | 0.00 | 5.40 | +5.40 |
| ttc | K=2 atk=10 bs=32 ttc | SUN397 | 1 | 70.80 | 0.00 | 2.20 | +2.20 |
