#!/usr/bin/env python3
"""Diagnostic: dump batch-1 RNG-independent artifacts from the OFFICIAL MAC pipeline.

Run with:
  cwd = external/MAC/code
  PYTHONPATH = external/MAC
  python <this script> --out <npz>

Dumps: images, target, text token ids, prompt texts, clean (no-defence) logits
for the very first validation batch, replicating mac.py main() setup exactly
for everything up to the first iter(val_loader).
"""
import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"f:\codex\NAC\external\MAC")
sys.path.insert(0, r"f:\codex\NAC\external\MAC\code")

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32, accuracy
from func import clip_img_preprocessing, multiGPU_CLIP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='F:/codex/NAC/data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    a = ap.parse_args()

    class A:
        pass

    args = A()
    args.root = a.root
    args.batch_size = a.batch_size
    args.num_workers = a.num_workers
    args.evaluate = True
    args.test_set = ['DTD']
    args.arch = 'ViT-B/32'
    args.image_size = 224
    args.test_attack_type = 'pgd'
    args.test_eps = 4 / 255.
    args.test_numsteps = 100
    args.test_stepsize = 1 / 255.
    args.mac_eps = 8 / 255.
    args.mac_numsteps = 4
    args.tau_thres = 0.7
    args.tau_temp = 0.01
    args.num_views = 2
    args.seed = 1
    args.cache = './cache'
    args.method = 'null_patch'
    args.prompt_size = 30
    args.add_prompt_size = 0
    args.dataset = 'tinyImageNet'
    args.model = 'clip'
    args.name = ''
    args.victim_resume = None
    args.outdir = '.'

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda'
    model, _ = clip.load(args.arch, device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    clip_visual = None
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = NullPrompter()
    add_prompter = TokenPrompter(0)
    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()
    criterion_attack = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    val_dataset, val_loader = load_val_dataset(args, 'DTD')
    texts = get_text_prompts_val([val_dataset], ['DTD'])[0]
    text_tokens = clip.tokenize(texts).to(device)

    images, target = next(iter(val_loader))
    images = images.to(device)
    target = target.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(images)),
                text_tokens, prompt_token=None, dataset_name='DTD'
            )

    acc = accuracy(out.float(), target, topk=(1,))[0].item()
    np.savez(
        a.out,
        images=images.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
        tokens=text_tokens.detach().cpu().numpy(),
        texts=np.array(texts),
        clean_logits=out.float().detach().cpu().numpy(),
    )
    print(f"[official] saved {a.out}")
    print(f"[official] images {tuple(images.shape)} target[:8] {target[:8].tolist()} "
          f"clean_top1 {acc:.2f} n_texts {len(texts)}")
    print(f"[official] texts[:2] {texts[:2]}")
    print(f"[official] classes_len {len(val_dataset.classes) if hasattr(val_dataset,'classes') else 'NA'}")


if __name__ == '__main__':
    main()