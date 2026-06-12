#!/usr/bin/env python3
"""
NAC: 6-dataset Table 1 with DTD included (correcting earlier exclusion).

Re-runs the 5-dataset multi-eps evaluation including DTD to compute
the full 6-dataset average. Used to update Table 2.
"""
import os
import sys
import json
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import AverageMeter, accuracy, convert_models_to_fp32, load_val_dataset, get_text_prompts_val
from func import clip_img_preprocessing, multiGPU_CLIP
from nac_fair_experiment import nac_counterattack
from test_time_counterattack import tau_thres_weighted_counterattacks as ttc_counterattack
from attacks import attack_pgd


def run_one(arch, dataset, eps, method, seed, root, batch_size, num_workers, outdir):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, _ = clip.load(arch, "cuda", jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).eval()
    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    val_dataset, val_loader = load_val_dataset(
        type("A", (), {"root": root,
                       "imagenet_root": os.path.join(root, "imagenet-100", "imagenet_folder"),
                       "tinyimagenet_root": os.path.join(root, "tiny-imagenet-200", "tiny-imagenet-200"),
                       "batch_size": batch_size, "num_workers": num_workers})(),
        "dtd",
    )
    texts = get_text_prompts_val([val_dataset], ["dtd"])[0]
    text_tokens = clip.tokenize(texts).cuda()
    criterion = nn.CrossEntropyLoss(reduction="sum").cuda()

    clean_m = AverageMeter("clean", ":.2f")
    adv_m = AverageMeter("adv", ":.2f")
    def_m = AverageMeter("def", ":.2f")
    t0 = time.time()
    for imgs, target in val_loader:
        imgs = imgs.cuda(); target = target.cuda()
        with autocast():
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(imgs),
                                    text_tokens, prompt_token=None,
                                    dataset_name="dtd")
                clean_m.update(accuracy(out[0], target, topk=(1,))[0].item(), imgs.size(0))
            delta_atk = attack_pgd(
                type("A", (), {"cache": "./cache"})(),
                prompter, model, None, None, add_prompter, criterion,
                imgs, target, 1.0 / 255, 10, "l_inf",
                text_tokens=text_tokens, epsilon=eps, dataset_name=dataset,
            )
            attacked = imgs + delta_atk
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(attacked),
                                    text_tokens, prompt_token=None,
                                    dataset_name=dataset)
                adv_m.update(accuracy(out[0], target, topk=(1,))[0].item(), imgs.size(0))
            if method == "nac":
                delta_def = nac_counterattack(
                    model, attacked.data, prompter, add_prompter,
                    alpha=1.0 / 255, attack_iters=2, norm="l_inf",
                    epsilon=4.0 / 255, visual_model_orig=None,
                    tau_thres=0.2, beta=2.0, clip_visual=None,
                    nac_momentum=0.9,
                )
            else:
                delta_def = ttc_counterattack(
                    model, attacked.data, prompter, add_prompter,
                    alpha=1.0 / 255, attack_iters=2, norm="l_inf",
                    epsilon=4.0 / 255, visual_model_orig=None,
                    tau_thres=0.2, beta=2.0, clip_visual=None,
                )
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(attacked + delta_def),
                                    text_tokens, prompt_token=None,
                                    dataset_name=dataset)
                def_m.update(accuracy(out[0], target, topk=(1,))[0].item(), imgs.size(0))
    result = {
        "arch": arch, "dataset": dataset, "eps": eps,
        "method": method, "seed": seed,
        "clean": clean_m.avg, "adv": adv_m.avg, "defended": def_m.avg,
        "gain": def_m.avg - adv_m.avg,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{method}_eps{int(eps*255)}_seed{seed}.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"[OK] {dataset} eps={eps:.4f} {method}: clean={result['clean']:.2f} adv={result['adv']:.2f} def={result['defended']:.2f} gain={result['gain']:+.2f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B/32")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--outdir", default="./results/dtd_included")
    args = ap.parse_args()

    # DTD at 3 eps levels, both methods
    results = []
    for eps in [1, 2, 4]:
        for method in ["ttc", "nac"]:
            r = run_one(args.arch, "DTD", eps / 255.0, method,
                        args.seed, args.root, args.batch_size,
                        args.num_workers, args.outdir)
            if r:
                results.append(r)
    # Print DTD-only summary
    print("\n=== DTD-included Multi-eps (CIFAR equivalent: see eps_scan) ===")
    print(f"{'eps':>5} {'method':>6} {'clean':>7} {'adv':>7} {'def':>7} {'gain':>+7}")
    for r in results:
        print(f"{r['eps']*255:>5.2f} {r['method']:>6} {r['clean']:>7.2f} {r['adv']:>7.2f} {r['defended']:>7.2f} {r['gain']:>+7.2f}")


if __name__ == "__main__":
    main()
