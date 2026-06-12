#!/usr/bin/env python3
"""
NAC: Embedding smoothness measurement on ViT-B/32 and RN50.

Measures the Lipschitz constant of the CLIP image encoder around a
random batch by computing ||∇f(x)||/||x||, where f is the encoder
and x is the (perturbed) image. A smaller value indicates a more
locally smooth embedding function.

This is a *direct* empirical test of the paper's local smoothness
hypothesis for why NAC fails on RN50 (the paper currently relies
on a post-hoc narrative without evidence).

Output: results/smoothness/{arch}_seed{seed}.json
"""
import os
import sys
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import AverageMeter, accuracy, convert_models_to_fp32, load_val_dataset, get_text_prompts_val
from func import clip_img_preprocessing, multiGPU_CLIP


def measure_lipschitz(model, prompter, add_prompter, text_tokens, loader,
                     dataset_name, n_batches=10, eps=1.0/255):
    """Measure ||grad|| / ||input|| as a Lipschitz proxy."""
    lip_vals = []
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        imgs = imgs.cuda()
        x = imgs.clone().detach().requires_grad_(True)
        with autocast():
            out, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                        clip_img_preprocessing(x),
                                        text_tokens, prompt_token=None,
                                        dataset_name=dataset_name)
            # Use a target output sum to compute a representative gradient
            target = out.sum(dim=1).mean()
        grad = torch.autograd.grad(target, x)[0]
        grad_norm = grad.flatten(1).norm(dim=1).mean().item()
        x_norm = imgs.flatten(1).norm(dim=1).mean().item()
        # Local Lipschitz: ||grad||/||x|| (relative gradient)
        local_lip = grad_norm / max(x_norm, 1e-8)
        lip_vals.append(local_lip)
    return float(np.mean(lip_vals)), float(np.std(lip_vals)), lip_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--outdir", default="./results/smoothness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_batches", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    results = {}
    for arch_name, arch_id in [("ViT-B/32", "ViT-B/32"), ("RN50", "RN50")]:
        print(f"=== {arch_name} ===")
        model, _ = clip.load(arch_id, "cuda", jit=False, prompt_len=0)
        for p in model.parameters():
            p.requires_grad = False
        convert_models_to_fp32(model)
        model = torch.nn.DataParallel(model).eval()
        prompter = NullPrompter()
        add_prompter = TokenPrompter(0)
        prompter = torch.nn.DataParallel(prompter).cuda()
        add_prompter = torch.nn.DataParallel(add_prompter).cuda()
        val_dataset, val_loader = load_val_dataset(
            type("A", (), {"root": args.root,
                           "imagenet_root": os.path.join(args.root, "imagenet-100", "imagenet_folder"),
                           "tinyimagenet_root": os.path.join(args.root, "tiny-imagenet-200", "tiny-imagenet-200"),
                           "batch_size": 16, "num_workers": 2})(),
            "cifar10",
        )
        texts = get_text_prompts_val([val_dataset], ["cifar10"])[0]
        text_tokens = clip.tokenize(texts).cuda()
        mean_lip, std_lip, all_vals = measure_lipschitz(
            model, prompter, add_prompter, text_tokens, val_loader,
            "cifar10", n_batches=args.n_batches,
        )
        results[arch_name] = {
            "mean_local_lipschitz": mean_lip,
            "std_local_lipschitz": std_lip,
            "per_batch": all_vals,
        }
        print(f"  {arch_name}: local Lipschitz = {mean_lip:.6e} +/- {std_lip:.6e}")
        del model
        torch.cuda.empty_cache()

    out_path = os.path.join(args.outdir, f"seed_{args.seed}.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[OK] Saved: {out_path}")

    # Quick interpretation
    vit_lip = results["ViT-B/32"]["mean_local_lipschitz"]
    rn50_lip = results["RN50"]["mean_local_lipschitz"]
    if rn50_lip > vit_lip:
        ratio = rn50_lip / vit_lip
        print(f"\nInterpretation: RN50's local Lipschitz is {ratio:.2f}x ViT's.")
        print("  -> Consistent with paper's hypothesis: RN50 is less locally smooth, "
              "NAC's Nesterov look-ahead provides less reliable descent direction.")
    else:
        print("\nInterpretation: Lipschitz measurements do NOT support the paper's "
              "post-hoc narrative that RN50 is less smooth. NAC's RN50 failure requires "
              "a different explanation.")


if __name__ == "__main__":
    main()
