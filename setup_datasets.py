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
    """Download ImageNet-100 via ModelScope (fast in China) or HuggingFace (global fallback).

    Expected layout: data/imagenet-100/imagenet_folder/val/<class_folders>/
    """
    target = os.path.join(DATA_DIR, "imagenet-100", "imagenet_folder", "val")
    if os.path.exists(target) and len(os.listdir(target)) >= 50:
        print(f"[OK] ImageNet-100 already at: {target}")
        return

    # --- Method 1: ModelScope (fast for China users) ---
    try:
        from modelscope.msdatasets import MsDataset
        print("[*] Downloading ImageNet-100 from ModelScope ...")
        ds = MsDataset.load('tany0699/mini_imagenet100', subset_name='default', split='validation')
        os.makedirs(target, exist_ok=True)
        from PIL import Image
        from tqdm import tqdm

        class_names = ds.features['label'].names if hasattr(ds, 'features') and ds.features else None
        for i, sample in enumerate(tqdm(ds, desc="ImageNet-100 (ModelScope)")):
            img = sample['image']
            label = sample['label']
            label_name = class_names[label] if class_names else str(label)
            class_dir = os.path.join(target, label_name)
            os.makedirs(class_dir, exist_ok=True)
            if isinstance(img, str):
                shutil.copy(img, os.path.join(class_dir, os.path.basename(img)))
            else:
                img.save(os.path.join(class_dir, f"{i:05d}.JPEG"))

        count = sum(1 for d in os.listdir(target) if os.path.isdir(os.path.join(target, d)))
        if count >= 50:
            print(f"[OK] ImageNet-100 ready (ModelScope): {count} classes")
            return
    except Exception as e:
        print(f"[!] ModelScope failed: {e}")

    # --- Method 2: HuggingFace datasets (global) ---
    try:
        from datasets import load_dataset
        print("[*] Downloading ImageNet-100 from HuggingFace (clane9/imagenet-100) ...")
        ds = load_dataset("clane9/imagenet-100", split="validation")
        os.makedirs(target, exist_ok=True)
        from tqdm import tqdm

        # Build class name mapping from dataset features
        label_names = ds.features['label'].names if hasattr(ds.features['label'], 'names') else None
        for sample in tqdm(ds, desc="ImageNet-100 (HF)"):
            img = sample['image']
            label = sample['label']
            label_name = label_names[label] if label_names else str(label)
            class_dir = os.path.join(target, label_name)
            os.makedirs(class_dir, exist_ok=True)
            # Use index from enumeration as unique filename
            idx = sample.get('__index__', 0) if isinstance(sample, dict) else 0
            img.save(os.path.join(class_dir, f"{idx:05d}.JPEG"))

        count = sum(1 for d in os.listdir(target) if os.path.isdir(os.path.join(target, d)))
        if count >= 50:
            print(f"[OK] ImageNet-100 ready (HuggingFace): {count} classes")
            return
    except Exception as e:
        print(f"[!] HuggingFace failed: {e}")

    # --- Both failed: give manual instructions ---
    print("[!] Automatic ImageNet-100 download failed.")
    print("    Manual setup: download 100-class ImageNet subset to:")
    print(f"    {target}")
    print("    Expected: 100 class subdirectories, each containing JPEG images.")

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
