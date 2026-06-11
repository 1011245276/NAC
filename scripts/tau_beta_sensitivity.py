#!/usr/bin/env python3
"""
NAC tau_thres / beta sensitivity scan.

Sweeps tau_thres in {0.1, 0.2, 0.3} and beta in {1.0, 2.0, 3.0}
on CIFAR-10, PGD eps=4/255, K=2, seed=0.

Usage:
    python scripts/tau_beta_sensitivity.py
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import AverageMeter, accuracy, convert_models_to_fp32, load_val_dataset, get_text_prompts_val
from func import clip_img_preprocessing, multiGPU_CLIP
from nac_fair_experiment import nac_counterattack
from attacks import attack_pgd


def run_one(arch, tau, beta, seed, root, batch_size, num_workers, outdir):
    name = f"nac_tau{tau}_beta{beta}"
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{name}_seed{seed}.json")
    if os.path.exists(log_path):
        print(f"[skip] {log_path} already exists")
        return json.load(open(log_path))

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
        type("A", (), {"root": root, "imagenet_root": os.path.join(root, "imagenet-100", "imagenet_folder"),
                       "tinyimagenet_root": os.path.join(root, "tiny-imagenet-200", "tiny-imagenet-200"),
                       "batch_size": batch_size, "num_workers": num_workers})(),
        "cifar10",
    )
    texts = get_text_prompts_val([val_dataset], ["cifar10"])[0]
    text_tokens = clip.tokenize(texts).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum").cuda()

    clean_meter = AverageMeter("clean", ":.2f")
    adv_meter = AverageMeter("adv", ":.2f")
    def_meter = AverageMeter("def", ":.2f")

    for images, target in val_loader:
        images = images.cuda(); target = target.cuda()
        with autocast():
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(images),
                                    text_tokens, prompt_token=None,
                                    dataset_name="cifar10")
                clean_meter.update(accuracy(out[0], target, topk=(1,))[0].item(), images.size(0))

            delta_atk = attack_pgd(
                type("A", (), {"cache": "./cache"})(),
                prompter, model, None, None, add_prompter, criterion,
                images, target, 1.0 / 255, 10, "l_inf",
                text_tokens=text_tokens, epsilon=4.0 / 255, dataset_name="cifar10",
            )
            attacked = images + delta_atk
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(attacked),
                                    text_tokens, prompt_token=None,
                                    dataset_name="cifar10")
                adv_meter.update(accuracy(out[0], target, topk=(1,))[0].item(), images.size(0))

            delta_def = nac_counterattack(
                model, attacked.data, prompter, add_prompter,
                alpha=1.0 / 255, attack_iters=2, norm="l_inf", epsilon=4.0 / 255,
                visual_model_orig=None,
                tau_thres=tau, beta=beta, clip_visual=None,
                nac_momentum=0.9,
            )
            with torch.no_grad():
                out = multiGPU_CLIP(None, None, None, model,
                                    clip_img_preprocessing(attacked + delta_def),
                                    text_tokens, prompt_token=None,
                                    dataset_name="cifar10")
                def_meter.update(accuracy(out[0], target, topk=(1,))[0].item(), images.size(0))

    result = {
        "arch": arch, "tau": tau, "beta": beta, "seed": seed,
        "clean": clean_meter.avg, "adv": adv_meter.avg, "defended": def_meter.avg,
        "gain": def_meter.avg - adv_meter.avg,
    }
    json.dump(result, open(log_path, "w"), indent=2)
    print(f"[OK] {name}: clean={result['clean']:.2f} adv={result['adv']:.2f} def={result['defended']:.2f} gain={result['gain']:+.2f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B/32")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--outdir", default="./results/tau_beta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    configs = [
        (0.1, 2.0),  # tau variation
        (0.2, 2.0),
        (0.3, 2.0),
        (0.2, 1.0),  # beta variation
        (0.2, 2.0),
        (0.2, 3.0),
    ]
    results = []
    for tau, beta in configs:
        r = run_one(args.arch, tau, beta, args.seed,
                    args.root, args.batch_size, args.num_workers, args.outdir)
        if r:
            results.append(r)
    out_json = os.path.join(args.outdir, "sensitivity_summary.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\n[OK] Summary: {out_json}")
    print("Tau/Beta sensitivity (CIFAR-10, PGD eps=4/255, K=2, NAC m=0.9):")
    print(f"{'tau':>6} {'beta':>6} {'clean':>7} {'adv':>7} {'def':>7} {'gain':>+7}")
    for r in results:
        print(f"{r['tau']:>6.2f} {r['beta']:>6.2f} {r['clean']:>7.2f} {r['adv']:>7.2f} {r['defended']:>7.2f} {r['gain']:>+7.2f}")


if __name__ == "__main__":
    main()
