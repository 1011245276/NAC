"""Latency benchmark: TTC vs NAC counterattack, per-image wall-clock time.

Measures the defense-only cost (counterattack on clean images, no attack
generation) on CIFAR-10 with CLIP ViT-B/32. Supports the paper's claim that
NAC adds no measurable cost over TTC at identical K.

Usage:
  python scripts/benchmark_latency.py --batch_size 32 --warmup 10 --iters 60
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import load_val_dataset, get_text_prompts_val, accuracy, convert_models_to_fp32
from func import clip_img_preprocessing, multiGPU_CLIP
from nac_fair_experiment import nac_counterattack
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=10, help='warmup batches (not timed)')
    p.add_argument('--iters', type=int, default=60, help='timed batches')
    p.add_argument('--K', type=int, default=2, help='counterattack steps')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random_seed = args.seed
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.root = './data'
    args.num_workers = 0
    args.evaluate = True
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p_ in model.parameters():
        p_.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).eval()

    ds, loader = load_val_dataset(args, args.dataset)
    texts = get_text_prompts_val([ds], [args.dataset])[0]
    text_tokens = clip.tokenize(texts).to(device)

    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    eps = 4 / 255.
    alpha = 1 / 255.

    batches = []
    for images, target in loader:
        batches.append(images.to(device))
        if len(batches) >= args.warmup + args.iters:
            break

    results = {}
    for method in ['ttc', 'nac']:
        # warmup
        for images in batches[:args.warmup]:
            if method == 'nac':
                nac_counterattack(model, images, prompter, add_prompter, alpha,
                                  args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0,
                                  nac_momentum=0.9)
            else:
                ttc_counterattack(model, images, prompter, add_prompter, alpha,
                                  args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0)
        torch.cuda.synchronize()

        per_image_ms = []
        for images in tqdm(batches[args.warmup:], desc=f'{method}-K{args.K}'):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if method == 'nac':
                nac_counterattack(model, images, prompter, add_prompter, alpha,
                                  args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0,
                                  nac_momentum=0.9)
            else:
                ttc_counterattack(model, images, prompter, add_prompter, alpha,
                                  args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0)
            torch.cuda.synchronize()
            per_image_ms.append((time.perf_counter() - t0) * 1000.0 / images.size(0))

        per_image_ms = np.array(per_image_ms)
        results[method] = (per_image_ms.mean(), per_image_ms.std())
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"{method.upper()}-K{args.K}: {per_image_ms.mean():.2f} +/- {per_image_ms.std():.2f} ms/image "
              f"(peak GPU mem {mem:.0f} MiB, {len(per_image_ms)} batches x {args.batch_size})")
        torch.cuda.reset_peak_memory_stats()

    print("\nSummary (per-image counterattack latency, ms):")
    for method, (m, s) in results.items():
        print(f"  {method.upper()}: {m:.2f} +/- {s:.2f}")


if __name__ == '__main__':
    main()
