#!/usr/bin/env python3
"""Diagnostic: dump batch-1 RNG-independent artifacts from the PORTED MAC pipeline.

Run with:
  cwd = nac_project
  python scripts/mac_dbg_dump_port.py --out <npz>

Mirrors scripts/mac_dbg_dump_official.py but uses the NAC harness modules and
the forced MAC prompt ("A photo of a {}.").
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replace import clip
from models.prompters import NullPrompter
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
    args.imagenet_root = os.path.join(a.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(a.root, 'tiny-imagenet-200', 'tiny-imagenet-200')
    args.batch_size = a.batch_size
    args.num_workers = a.num_workers
    args.evaluate = True
    args.prompt_template = 'A photo of a {}.'

    model, _ = clip.load('ViT-B/32', 'cuda', jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()

    val_dataset, val_loader = load_val_dataset(args, 'dtd')
    texts = get_text_prompts_val([val_dataset], ['dtd'])[0]
    texts = get_text_prompts_val([val_dataset], ['dtd'],
                                 template=args.prompt_template, force_template=True)[0]
    text_tokens = clip.tokenize(texts).cuda()

    images, target = next(iter(val_loader))
    images = images.cuda()
    target = target.cuda()

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(images)),
                text_tokens, prompt_token=None, dataset_name='dtd'
            )

    acc = accuracy(out, target, topk=(1,))[0].item()
    np.savez(
        a.out,
        images=images.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
        tokens=text_tokens.detach().cpu().numpy(),
        texts=np.array(texts),
        clean_logits=out.float().detach().cpu().numpy(),
    )
    print(f"[port] saved {a.out}")
    print(f"[port] images {tuple(images.shape)} target[:8] {target[:8].tolist()} "
          f"clean_top1 {acc:.2f} n_texts {len(texts)}")
    print(f"[port] texts[:2] {texts[:2]}")
    print(f"[port] classes_len {len(val_dataset.classes) if hasattr(val_dataset,'classes') else 'NA'}")


if __name__ == '__main__':
    main()