"""Latency benchmark: per-image wall-clock cost of test-time counterattacks.

Measures the defense-only cost (counterattack on clean images, no attack
generation) on CIFAR-10 with CLIP ViT-B/32. Supports the paper's claims that
(a) NAC adds no measurable cost over TTC at identical K, and (b) nominal
step-size budget (alpha x K) and actual compute must be reported separately.

Covers the optimizer family (ttc, nac, momentum, pure_la, adam, adagrad,
rmsprop, rmsprop_bc) plus DOC and MAC so the unified grid table can carry an
honest cost column for every method it compares.

Usage:
  python scripts/benchmark_latency.py --methods ttc,nac --K 2
  python scripts/benchmark_latency.py --methods doc --K 4 --alpha 3
  python scripts/benchmark_latency.py --methods mac --K 4 --mac_eps 8 --mac_views 2
"""
import argparse
import datetime
import json
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
from nac_fair_experiment import (
    nac_counterattack, momentum_counterattack, pure_la_counterattack,
    adam_counterattack, adagrad_counterattack, rmsprop_counterattack,
    rmsprop_bc_counterattack, doc_counterattack,
    mac_build_views, mac_build_aug, mac_guided_counterattack,
)
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack

device = "cuda" if torch.cuda.is_available() else "cpu"

# Counterattacks sharing the fair-experiment signature; ttc is special-cased
# (official signature has no preprocess_fn) and DOC/MAC have their own forms.
SIMPLE_METHODS = {
    "nac": nac_counterattack,
    "momentum": momentum_counterattack,
    "pure_la": pure_la_counterattack,
    "adam": adam_counterattack,
    "adagrad": adagrad_counterattack,
    "rmsprop": rmsprop_counterattack,
    "rmsprop_bc": rmsprop_bc_counterattack,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=10, help='warmup batches (not timed)')
    p.add_argument('--iters', type=int, default=60, help='timed batches')
    p.add_argument('--K', type=int, default=2, help='counterattack steps')
    p.add_argument('--alpha', type=float, default=1.0, help='counterattack step size (1/255 units)')
    p.add_argument('--eps_def', type=float, default=4.0, help='defense eps (1/255 units)')
    p.add_argument('--methods', default='ttc,nac', help='comma list of methods to benchmark')
    p.add_argument('--mac_eps', type=float, default=8.0, help='MAC defense eps (1/255 units)')
    p.add_argument('--mac_views', type=int, default=2)
    p.add_argument('--doc_tau', type=float, default=0.155)
    p.add_argument('--doc_temp', type=float, default=75.0)
    p.add_argument('--tag', default='', help='label stored in the JSON output')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    alpha = args.alpha / 255.
    eps = args.eps_def / 255.

    # DOC reads gating hyperparameters from the args namespace.
    doc_args = type("A", (), {"learnable_tau": args.doc_tau, "temperature": args.doc_temp})()

    # MAC: official rule step = eps/4 (normalized scale), views share one aug pipeline.
    mac_eps_norm = args.mac_eps / 255.
    mac_alpha = mac_eps_norm / 4.0
    mac_aug = mac_build_aug() if args.mac_views > 1 else None

    def defend(method, images):
        # Signatures mirror the fair-experiment call sites exactly:
        # ttc/nac have no preprocess_fn; momentum/pure_la/adam/adagrad/
        # rmsprop(_bc) do.
        if method == 'ttc':
            return ttc_counterattack(model, images, prompter, add_prompter, alpha,
                                     args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0,
                                     clip_visual=None)
        if method == 'nac':
            return nac_counterattack(model, images, prompter, add_prompter, alpha,
                                     args.K, 'l_inf', eps, tau_thres=0.2, beta=2.0,
                                     clip_visual=None, nac_momentum=0.9)
        if method in SIMPLE_METHODS:
            return SIMPLE_METHODS[method](model, images, prompter, add_prompter,
                                          alpha, args.K, 'l_inf', eps,
                                          tau_thres=0.2, beta=2.0, clip_visual=None,
                                          preprocess_fn=clip_img_preprocessing)
        if method == 'doc':
            return doc_counterattack(doc_args, model, images, prompter, add_prompter,
                                     alpha, args.K, 'l_inf', eps, beta=2.0, clip_visual=None)
        if method == 'mac':
            views = mac_build_views(images, args.mac_views, mac_aug)
            return mac_guided_counterattack(model, views, prompter, add_prompter,
                                            alpha=mac_alpha, attack_iters=args.K,
                                            norm='l_inf', epsilon=mac_eps_norm,
                                            tau_thres=0.7, tau_temp=0.01,
                                            clip_visual=None, aug=mac_aug)
        raise ValueError(f"unknown method: {method}")

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    batches = []
    for images, target in loader:
        batches.append(images.to(device))
        if len(batches) >= args.warmup + args.iters:
            break

    results = {}
    for method in methods:
        for images in batches[:args.warmup]:
            defend(method, images)
        torch.cuda.synchronize()

        per_image_ms = []
        for images in tqdm(batches[args.warmup:], desc=f'{method}-K{args.K}'):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            defend(method, images)
            torch.cuda.synchronize()
            per_image_ms.append((time.perf_counter() - t0) * 1000.0 / images.size(0))

        per_image_ms = np.array(per_image_ms)
        mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        results[method] = {
            "mean_ms": round(float(per_image_ms.mean()), 3),
            "std_ms": round(float(per_image_ms.std()), 3),
            "peak_gpu_mib": round(float(mem), 0),
            "n_batches": len(per_image_ms),
        }
        print(f"{method.upper()}-K{args.K}: {per_image_ms.mean():.2f} +/- {per_image_ms.std():.2f} ms/image "
              f"(peak GPU mem {mem:.0f} MiB, {len(per_image_ms)} batches x {args.batch_size})")
        torch.cuda.reset_peak_memory_stats()

    print("\nSummary (per-image counterattack latency, ms):")
    for method, r in results.items():
        print(f"  {method.upper()}: {r['mean_ms']:.2f} +/- {r['std_ms']:.2f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join('results', 'benchmark_latency')
    os.makedirs(outdir, exist_ok=True)
    slug = (args.tag or f"K{args.K}_a{args.alpha:g}_{'-'.join(methods)}").replace('/', '_')
    out_path = os.path.join(outdir, f"latency_{slug}_{stamp}.json")
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump({
            "generated_at": stamp,
            "dataset": args.dataset, "arch": "ViT-B/32", "K": args.K,
            "alpha_255": args.alpha, "eps_def_255": args.eps_def,
            "mac_eps_255": args.mac_eps, "mac_views": args.mac_views,
            "batch_size": args.batch_size, "warmup": args.warmup,
            "iters": args.iters, "seed": args.seed, "results": results,
        }, fh, indent=1)
    print(f"JSON: {out_path}")


if __name__ == '__main__':
    main()
