#!/usr/bin/env python3
"""
NAC: One-step dataset setup.
Usage: python setup_datasets.py

Downloads all required datasets:
  - CIFAR-10, CIFAR-100, STL-10, DTD, Flowers-102  (auto, via torchvision)
  - ImageNet-100                                     (via ModelScope → organized)
"""

import os, sys, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

def setup_imagenet100():
    """Download ImageNet-100 from ModelScope and organize into expected layout.

    Expected layout: data/imagenet-100/imagenet_folder/val/<class_folders>/
    """
    target = os.path.join(DATA_DIR, "imagenet-100", "imagenet_folder", "val")
    if os.path.exists(target) and len(os.listdir(target)) >= 50:
        print(f"[OK] ImageNet-100 already at: {target}")
        return

    print("[*] Downloading ImageNet-100 from ModelScope ...")
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        print("[!] modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    ds = MsDataset.load('tany0699/mini_imagenet100', subset_name='default', split='validation')

    # ModelScope caches to ~/.cache/modelscope/; find the actual files
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/datasets/tany0699/mini_imagenet100")
    if not os.path.exists(cache_dir):
        # Fallback: try to locate via the dataset object
        print("[!] Cannot locate ModelScope cache. Trying alternate method...")
        # Download directly using the dataset's raw data
        os.makedirs(target, exist_ok=True)
        from torchvision.datasets import ImageFolder
        print("[*] Extracting to:", target)
        # Iterate and save
        import torch
        from PIL import Image
        from tqdm import tqdm

        # Build class index
        class_names = ds.features['label'].names if hasattr(ds, 'features') else None
        for i, sample in enumerate(tqdm(ds, desc="ImageNet-100")):
            img = sample['image']
            label = sample['label']
            if class_names:
                label_name = class_names[label]
            else:
                label_name = str(label)
            class_dir = os.path.join(target, label_name)
            os.makedirs(class_dir, exist_ok=True)
            if isinstance(img, str):
                # It's a path
                shutil.copy(img, os.path.join(class_dir, os.path.basename(img)))
            else:
                img.save(os.path.join(class_dir, f"{i:05d}.JPEG"))
        print(f"[OK] ImageNet-100 saved to: {target}")
        return

    # Copy from ModelScope cache to data/
    os.makedirs(target, exist_ok=True)
    # The cache structure varies; try common patterns
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                # Determine class from parent folder name
                rel = os.path.relpath(root, cache_dir)
                parts = rel.replace('\\', '/').split('/')
                # Last meaningful part is the class name
                class_name = parts[-1] if parts[-1] else parts[-2]
                class_dir = os.path.join(target, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(root, f)
                if not os.path.exists(os.path.join(class_dir, f)):
                    shutil.copy2(src, class_dir)

    count = sum(1 for _ in os.listdir(target) if os.path.isdir(os.path.join(target, _)))
    if count >= 50:
        print(f"[OK] ImageNet-100 ready: {count} classes at {target}")
    else:
        print(f"[!] Only {count} class folders found. Expected 100. Check manually.")
        print(f"    Target: {target}")

def setup_auto_datasets():
    """Trigger download of all auto-downloadable datasets."""
    print("[*] Pre-downloading datasets ...")
    import torchvision.datasets as tvdata

    # CIFAR-10/100, STL-10: torchvision auto-download
    for name, fn in [
        ("CIFAR-10", lambda: tvdata.CIFAR10(DATA_DIR, download=True)),
        ("CIFAR-100", lambda: tvdata.CIFAR100(DATA_DIR, download=True)),
        ("STL-10", lambda: tvdata.STL10(DATA_DIR, split='test', download=True)),
    ]:
        print(f"  Downloading {name} ...")
        fn()
        print(f"  [OK] {name}")

    # DTD: auto-download (download=True in utils.py)
    from replace.datasets import dtd
    print("  Downloading DTD ...")
    dtd.DTD(DATA_DIR, split='test', download=True)
    print("  [OK] DTD")

    # Flowers-102: needs explicit download (utils.py uses download=False)
    from replace.datasets import flowers102
    print("  Downloading Flowers-102 ...")
    flowers102.Flowers102(DATA_DIR, split='test', download=True)
    print("  [OK] Flowers-102")

    print("[OK] All datasets ready.")

if __name__ == "__main__":
    print("=" * 55)
    print(" NAC Dataset Setup")
    print("=" * 55)
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_auto_datasets()
    setup_imagenet100()
    print("=" * 55)
    print(" Done! Run: bash scripts/reproduce_main.sh")
    print("=" * 55)
