# -*- coding: utf-8 -*-
"""
EOT (expectation-over-defense-initialization) defense-aware adaptive PGD.

Extends scripts/adaptive_attack_pilot.py per the 2026-10-14 cross-review
(07 §4.1-5 / 06_external_audit D-4): at each attack step the counterattack is
re-run from M fresh random initializations (the counterattack functions
re-initialize their internal delta via uniform_ on every call), the defended
outputs are averaged in CE space, and one sign-step is taken on the attack
perturbation. This replaces the pilot's implicit single-sample stochasticity
with an explicit M-sample expectation.

Protocol: CIFAR-10, ViT-B/32, eps_att=4/255, PGD-10 step=1/255, K=2,
alpha=1/255, eps_def=4/255, bs=32. Methods: ttc / nac / doc (DOC uses the
official README gating tau=0.155, T=75.0, as in the pilot).
"""
import sys, os, json, logging, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import random

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, AverageMeter, accuracy, clip_img_preprocessing,
    doc_counterattack, nac_counterattack, ttc_counterattack,
)

clip_load = clip.load
from utils import convert_models_to_fp32

device = "cuda" if torch.cuda.is_available() else "cpu"


def eot_adaptive_pgd(counter_fn, mk, model, images, target, text_tokens,
                     prompter, add_prompter, eps, alpha, steps, dataset, eot_m):
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)
    delta.requires_grad = True
    for _s in range(steps):
        x_attack = torch.clamp(images + delta, 0, 1)
        # (i)+(ii) M defense re-runs from M fresh inits; average CE over the
        # M defended outputs (true EOT over the defense's random init).
        losses = []
        for _m in range(eot_m):
            Delta = counter_fn(**mk(x_attack.detach())).detach()
            defended = torch.clamp(x_attack + Delta, 0, 1)
            output, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
            losses.append(F.cross_entropy(output, target))
        loss = torch.stack(losses).mean()
        # (iii) sign-gradient update on the attack perturbation
        grad = torch.autograd.grad(loss, delta)[0]
        delta = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps).detach()
        delta.requires_grad = True
    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac', 'doc'])
    ap.add_argument('--eot_inits', type=int, default=5)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--max_batches', type=int, default=0,
                    help='V5 smoke test only: limit val batches (0 = all). '
                         'Does not touch RNG or computation for full runs.')
    ap.add_argument('--dataset', default='cifar10',
                    help='E2 extension: validation dataset name passed to '
                         'load_val_dataset/get_text_prompts_val (cifar10, '
                         'stl10, dtd, ...).')
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = cli.outdir or os.path.join('results', 'adaptive_attack_eot', stamp)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'eot.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
        '--test_set', cli.dataset, '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', cli.dataset, '--seed', '0', '--num_seeds', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
        '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
        '--learnable_tau', '0.155', '--temperature', '75.0',
        '--counterattack', 'ttc', '--outdir', outdir,
    ]
    from nac_fair_experiment import parse_options
    args = parse_options()
    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    logging.info("EOT adaptive attack | M=%d seeds=%s methods=%s | K=2 alpha=1/255 eps=4/255 bs=32",
                 cli.eot_inits, cli.seeds, cli.methods)
    print(f"EOT adaptive attack M={cli.eot_inits} seeds={cli.seeds} methods={cli.methods}; log:", log_path)

    model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    val_dataset, val_loader = load_val_dataset(args, cli.dataset)
    texts = get_text_prompts_val([val_dataset], [cli.dataset])[0]
    text_tokens = clip.tokenize(texts).to(device)

    class _A: pass
    docargs = _A()
    docargs.learnable_tau = args.learnable_tau
    docargs.temperature = args.temperature

    def mk(name, x):
        if name == 'ttc':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None)
        if name == 'nac':
            return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                        norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                        beta=args.beta, clip_visual=None,
                        nac_momentum=args.nac_momentum, clamp_lookahead=False)
        return dict(args=docargs, model=model, X=x, prompter=prompter,
                    add_prompter=add_prompter, alpha=args.ttc_stepsize,
                    attack_iters=args.ttc_numsteps, norm='l_inf',
                    epsilon=args.ttc_eps, beta=args.beta, clip_visual=None)

    fns = {'ttc': ttc_counterattack, 'nac': nac_counterattack, 'doc': doc_counterattack}
    results = {}
    out_json = os.path.join(outdir, 'eot_results.json')
    for seed in cli.seeds:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for name in cli.methods:
            fn = fns[name]
            top1_adv = AverageMeter('Adv', ':.2f')
            top1_def = AverageMeter(f'Def+{name}', ':.2f')
            # V5 per-sample records (appended inside existing no_grad blocks;
            # no RNG consumers added -> full-run behaviour bit-identical)
            ps_def_correct, ps_adv_correct, ps_l2, ps_linf, ps_index = [], [], [], [], []
            t0 = time.time()
            logging.info(f"=== seed {seed} | EOT M={cli.eot_inits} vs {name} ===")
            print(f"=== seed {seed} | EOT M={cli.eot_inits} vs {name} ===")
            for bi, (images, target) in enumerate(val_loader):
                if cli.max_batches and bi >= cli.max_batches:
                    break
                images, target = images.to(device), target.to(device)
                delta_atk = eot_adaptive_pgd(
                    fn, lambda x, _n=name: mk(_n, x), model, images, target,
                    text_tokens, prompter, add_prompter,
                    args.test_eps, args.test_stepsize, args.test_numsteps, cli.dataset,
                    cli.eot_inits)
                attacked = torch.clamp(images + delta_atk, 0, 1)
                with torch.no_grad():
                    out_adv, _, _, _ = multiGPU_CLIP(
                        None, None, None, model, prompter(clip_img_preprocessing(attacked)),
                        text_tokens=text_tokens, prompt_token=None, dataset_name=cli.dataset)
                    top1_adv.update(accuracy(out_adv, target, topk=(1,))[0].item(), images.size(0))
                    ps_adv_correct.extend((out_adv.argmax(1) == target).cpu().tolist())
                    ps_l2.extend(delta_atk.view(delta_atk.size(0), -1).norm(dim=1).cpu().tolist())
                    ps_linf.extend(delta_atk.view(delta_atk.size(0), -1).abs().amax(dim=1).cpu().tolist())
                    ps_index.extend((bi * args.batch_size + torch.arange(images.size(0))).tolist())
                with torch.enable_grad():
                    Delta = fn(**mk(name, attacked.data))
                with torch.no_grad():
                    out_def, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(torch.clamp(attacked + Delta, 0, 1))),
                        text_tokens=text_tokens, prompt_token=None, dataset_name=cli.dataset)
                    top1_def.update(accuracy(out_def, target, topk=(1,))[0].item(), images.size(0))
                    ps_def_correct.extend((out_def.argmax(1) == target).cpu().tolist())
                if bi % 50 == 0:
                    logging.info(f"  seed {seed} {name} batch {bi}: adv={top1_adv.avg:.2f} def={top1_def.avg:.2f}")
                    print(f"  seed {seed} {name} batch {bi}/313: adv={top1_adv.avg:.2f} def={top1_def.avg:.2f}", flush=True)
                    torch.cuda.empty_cache()
            key = f"{name}_seed{seed}"
            results[key] = dict(adv=round(top1_adv.avg, 2), defended=round(top1_def.avg, 2),
                                gain=round(top1_def.avg - top1_adv.avg, 2),
                                elapsed_min=round((time.time() - t0) / 60, 1),
                                per_sample=dict(
                                    n=len(ps_def_correct),
                                    defended_correct=ps_def_correct,
                                    adv_correct=ps_adv_correct,
                                    l2_norm=[round(v, 4) for v in ps_l2],
                                    linf_norm=[round(v, 6) for v in ps_linf],
                                    image_index=ps_index),
                                attack_config=dict(method=name, seed=seed,
                                                   eot_m=cli.eot_inits,
                                                   eps_att=round(args.test_eps, 6),
                                                   steps=args.test_numsteps))
            logging.info(f"{key}: {results[key]}")
            print(f"{key}: {results[key]}", flush=True)
            torch.cuda.empty_cache()
            with open(out_json, 'w') as f:
                json.dump(dict(config=f"{cli.dataset.upper()}, eps_att=4/255, PGD-10 step=1/255, K=2, alpha=1/255, eps_def=4/255, bs=32, EOT M={cli.eot_inits} over defense random init, first-order defense-aware adaptive PGD, seeds={cli.seeds}",
                               results=results), f, indent=2)
    logging.info("DONE " + json.dumps(results))
    print("DONE:", json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
