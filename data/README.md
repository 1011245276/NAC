# Dataset Preparation

Place datasets under this directory. The following datasets are required:

## Auto-download (torchvision)
These download automatically on first use:

| Dataset | Classes | Test Images |
|---------|---------|-------------|
| CIFAR-10 | 10 | 10,000 |
| CIFAR-100 | 100 | 10,000 |
| STL-10 | 10 | 8,000 |

## Manual Download

### Flowers-102
```
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf 102flowers.tgz -C ./data/
```

### DTD (Describable Textures Dataset)
```
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz -C ./data/
```

### ImageNet-100
From ModelScope:
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('tany0699/mini_imagenet100', subset_name='default', split='validation')
```

### EuroSAT
The TTC code's custom loader expects `./data/eurosat/2750/` structure.
Download from: https://github.com/phelber/EuroSAT

### SUN397
The TTC code's custom loader expects `./data/SUN397/` with class subdirectories.
Download from: https://vision.princeton.edu/projects/2010/SUN/

## Directory structure after setup

```
data/
├── cifar-10-batches-py/        (auto)
├── cifar-100-python/           (auto)
├── stl10_binary/               (auto)
├── dtd/
│   └── dtd/
│       └── images/
├── eurosat/
│   └── 2750/
├── flowers-102/
├── imagenet-100/
│   └── imagenet_folder/
│       └── val/
├── SUN397/
└── tiny-imagenet-200/
```
