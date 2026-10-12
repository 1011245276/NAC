# -*- coding: utf-8 -*-
"""
Adaptive (defense-aware) PGD pilot — closes the gradient-masking gap flagged in
the paper's Limitations and motivated by CLIP-MAD (Zhang et al., ICML 2026).

Attack model: first-order adaptive PGD. At each attack step the adversary
(i) runs the defense's counterattack on the current adversarial image (response
treated as fixed, i.e. stop-gradient through the counterattack's internal
optimization), (ii) computes zero-shot CE loss on the DEFENDED image, and
(iii) updates the perturbation with sign-gradient wrt the adversarial image.
This targets the post-defense classifier rather than the undefended one, so
residual robustness cannot be attributed purely to the attack being unaware of
the defense (single EOT sample per step; the counterattack's random delta_init
re-randomizes across attack steps, giving implicit stochasticity).

Methods: TTC / NAC / DOC (K=2, alpha=1/255, eps_def=4/255) on CIFAR-10,
eps_att=4/255, 10 attack steps, step 1/255, bs=32, seed 0.
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
    parse_options, load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, AverageMeter, accuracy, clip_img_preprocessing,
    doc_counterattack, nac_counterattack, ttc_counterattack,
)

clip_load = clip.load
from utils import convert_models_to_fp32

device = "cuda" if torch.cuda.is_available() else "cpu"


def adaptive_pgd_attack(counter_fn, mk, model, images, target, text_tokens,
                        prompter, add_prompter, eps, alpha, steps, dataset):
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)
    delta.requires_grad = True
    for _s in range(steps):
        x_attack = torch.clamp(images + delta, 0, 1)
        # (i) counterattack response as a fixed operator
        Delta = counter_fn(**mk(x_attack.detach())).detach()
        # (ii) CE loss on the defended image
        defended = torch.clamp(x_attack + Delta, 0, 1)
        output, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(defended)),
            text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
        loss = F.cross_entropy(output, target)
        # (iii) sign-gradient update on the attack perturbation
        grad = torch.autograd.grad(loss, delta)[0]
        delta = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps).detach()
        delta.requires_grad = True
    return delta.detach()


def main():
    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
        '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
        '--learnable_tau', '0.155', '--temperature', '75.0',
        '--counterattack', 'ttc', '--outdir', './results/adaptive_attack',
    ]
    args = parse_options()
    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    outdir = './results/adaptive_attack'
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'pilot_2026_10_12.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Adaptive defense-aware PGD pilot | K=2 alpha=1/255 eps=4/255 bs=32 seed=0")
    print("Adaptive attack pilot starting; log:", log_path)

    seed = 0
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, _ = clip_load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters():
        p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    val_dataset, val_loader = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([val_dataset], ['cifar10'])[0]
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
    for name in ['ttc', 'nac', 'doc']:
        fn, kw = fns[name], mk
        top1_adv = AverageMeter('Adv', ':.2f')
        top1_def = AverageMeter(f'Def+{name}', ':.2f')
        t0 = time.time()
        logging.info(f"=== adaptive attack vs {name} ===")
        print(f"=== adaptive attack vs {name} ===")
        for bi, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            delta_atk = adaptive_pgd_attack(
                fn, lambda x, _n=name: mk(_n, x), model, images, target,
                text_tokens, prompter, add_prompter,
                args.test_eps, args.test_stepsize, args.test_numsteps, 'cifar10')
            attacked = torch.clamp(images + delta_atk, 0, 1)
            with torch.no_grad():
                out_adv, _, _, _ = multiGPU_CLIP(
                    None, None, None, model, prompter(clip_img_preprocessing(attacked)),
                    text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
                top1_adv.update(accuracy(out_adv, target, topk=(1,))[0].item(), images.size(0))
            # NOTE: counterattack must run OUTSIDE no_grad (it internally
            # backprops wrt its own delta); re-enable grad mode explicitly.
            with torch.enable_grad():
                Delta = fn(**mk(name, attacked.data))
            with torch.no_grad():
                out_def, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(torch.clamp(attacked + Delta, 0, 1))),
                    text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
                top1_def.update(accuracy(out_def, target, topk=(1,))[0].item(), images.size(0))
            if bi % 50 == 0:
                logging.info(f"  {name} batch {bi}: adv={top1_adv.avg:.2f} def={top1_def.avg:.2f}")
                print(f"  {name} batch {bi}/313: adv={top1_adv.avg:.2f} def={top1_def.avg:.2f}")
                torch.cuda.empty_cache()
        results[name] = dict(adv=round(top1_adv.avg, 2), defended=round(top1_def.avg, 2),
                             gain=round(top1_def.avg - top1_adv.avg, 2),
                             elapsed_min=round((time.time() - t0) / 60, 1))
        logging.info(f"{name}: {results[name]}")
        print(f"{name}: {results[name]}")
        torch.cuda.empty_cache()
        with open(os.path.join(outdir, 'pilot_results.json'), 'w') as f:
            json.dump(dict(config='CIFAR-10, eps_att=4/255, PGD-10 step=1/255, K=2, alpha=1/255, eps_def=4/255, bs=32, seed=0, first-order defense-aware adaptive PGD', results=results), f, indent=2)
    logging.info("DONE " + json.dumps(results))
    print("DONE:", json.dumps(results))


if __name__ == '__main__':
    main()
