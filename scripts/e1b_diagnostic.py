# -*- coding: utf-8 -*-
"""E1b ranking-mechanism diagnostic (pre-registered 2026-10-28, see
thesis_proposal/研究二实验计划.md E1b 执行预登记).

For each method in {ttc, nac, doc}: run the E-E-lineage instrumented attack
(mean-of-CE aggregation, M_att=5 draws per step, 10 steps, K=2 counterattack,
bs=32, seed 0, 256 CIFAR-10 samples) and record, per (sample, step):
  - per-draw gradient norms (grad of CE_draw w.r.t. the attack delta);
  - within-sample pairwise cosines across the 5 draw gradients (10 pairs);
  - cosine of each draw gradient to the 5-draw mean gradient.
Aggregates per method -> results/e1b_diagnostic/e1b_stats_seed0.json.
Read-only w.r.t. production code; the attack itself proceeds on the mean
gradient (sign step) so gradients are on-trajectory.
"""
import sys, os, json, time, logging
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import torch
import torch.nn.functional as F
from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (load_val_dataset, get_text_prompts_val,
                                 multiGPU_CLIP, clip_img_preprocessing,
                                 doc_counterattack, nac_counterattack,
                                 ttc_counterattack)
from utils import convert_models_to_fp32

device = "cuda"
clip_load = clip.load

sys.argv = [
    'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
    '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
    '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
    '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
    '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
    '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
    '--learnable_tau', '0.155', '--temperature', '75.0',
    '--counterattack', 'ttc', '--outdir', os.path.join('results', '_e1b_diag'),
]
from nac_fair_experiment import parse_options
args = parse_options()
args.ttc_eps = args.ttc_eps / 255.
args.ttc_stepsize = args.ttc_stepsize / 255.
args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

M_ATT, STEPS, MAX_BATCHES, EPS, ALPHA = 5, 10, 8, 4 / 255., 1 / 255.

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
    # doc_counterattack takes the full parsed options namespace
    return dict(args=args, model=model, X=x, prompter=prompter,
                add_prompter=add_prompter, alpha=args.ttc_stepsize,
                attack_iters=args.ttc_numsteps, norm='l_inf',
                epsilon=args.ttc_eps, beta=args.beta, clip_visual=None)


CFN = {'ttc': ttc_counterattack, 'nac': nac_counterattack, 'doc': doc_counterattack}


def draw_grad(name, x_attack, target, delta):
    """One EOT draw: counterattack (detached) -> defended logits -> CE grad
    w.r.t. the attack delta.  Graph freed on return.  NOTE: the counter fn
    receives x_attack.DETACH() (as in the production pipelines) -- passing
    the grad-carrying tensor makes the counter's internal clamp output a
    non-leaf and its requires_grad assignment raises RuntimeError."""
    kw = mk(name, x_attack.detach())
    Delta = CFN[name](**kw).detach()
    defended = torch.clamp(x_attack + Delta, 0, 1)
    out, _, _, _ = multiGPU_CLIP(
        None, None, None, model, prompter(clip_img_preprocessing(defended)),
        text_tokens=text_tokens, prompt_token=None, dataset_name='cifar10')
    loss = F.cross_entropy(out, target)
    # retain_graph=True: the x_attack prefix graph (clamp(images + delta)) is
    # SHARED by the M_att draws of one step; the first autograd.grad would
    # otherwise free it and the second draw would raise
    # "backward through the graph a second time".  The graph is freed when
    # the step's tensors go out of scope.
    g = torch.autograd.grad(loss, delta, retain_graph=True)[0]
    return g.detach()


def stats_for_grads(grads):
    """grads: list of M tensors [B, D].  Returns per-sample pairwise cos
    (mean over the 10 pairs), per-draw norms, cos-to-mean."""
    G = torch.stack(grads, 0)                      # [M, B, D]
    Gn = F.normalize(G, dim=2)
    norms = G.flatten(2).norm(dim=2)               # [M, B]
    cos_pairs = []
    for i in range(M_ATT):
        for j in range(i + 1, M_ATT):
            cos_pairs.append((Gn[i] * Gn[j]).sum(dim=1))   # [B]
    mean_g = G.mean(0)                              # [B, D]
    mean_gn = F.normalize(mean_g, dim=1)
    cos_to_mean = torch.stack([(Gn[m] * mean_gn).sum(dim=1)
                               for m in range(M_ATT)])      # [M, B]
    return (norms.mean().item(), norms.std().item(),
            torch.stack(cos_pairs).mean().item(),
            cos_to_mean.mean().item())


def main():
    import numpy as np
    outdir = os.path.join(HERE, '..', 'results', 'e1b_diagnostic')
    os.makedirs(outdir, exist_ok=True)
    for seed in (0, 1, 2):
        out_json = os.path.join(outdir, f'e1b_stats_seed{seed}.json')
        if os.path.exists(out_json):
            print(f'[skip] seed {seed} already done', flush=True)
            continue
        log_path = os.path.join(outdir, f'e1b_seed{seed}.log')
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format="%(asctime)s - %(message)s", force=True)
        out = {}
        for name in ('ttc', 'nac', 'doc'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            agg = dict(pair_cos=[], norm_mean=[], norm_std=[], cos_to_mean=[])
            t0 = time.time()
            logging.info("=== seed %d | %s ===", seed, name)
            for bi, (images, target) in enumerate(val_loader):
                if bi >= MAX_BATCHES:
                    break
                images, target = images.to(device), target.to(device)
                delta = torch.clamp(torch.zeros_like(images).uniform_(-EPS, EPS),
                                    -EPS, EPS).requires_grad_(True)
                for _s in range(STEPS):
                    x_attack = torch.clamp(images + delta, 0, 1)
                    grads = []
                    for _m in range(M_ATT):
                        grads.append(draw_grad(name, x_attack, target, delta))
                    nm, ns, pc, cm = stats_for_grads(grads)
                    agg['norm_mean'].append(nm)
                    agg['norm_std'].append(ns)
                    agg['pair_cos'].append(pc)
                    agg['cos_to_mean'].append(cm)
                    mean_g = torch.stack(grads, 0).mean(0)
                    delta = torch.clamp(delta + ALPHA * torch.sign(mean_g),
                                        -EPS, EPS).detach().requires_grad_(True)
                    del grads
                logging.info("%s batch %d (batch-mean over %d steps): "
                             "pair_cos=%.4f cos_to_mean=%.4f norm=%.4f",
                             name, bi, STEPS,
                             float(np.mean(agg['pair_cos'][-STEPS:])),
                             float(np.mean(agg['cos_to_mean'][-STEPS:])),
                             float(np.mean(agg['norm_mean'][-STEPS:])))
                if bi % 2 == 0:
                    print(f"  s{seed} {name} batch {bi} (mean over "
                          f"{STEPS} steps): pair_cos="
                          f"{float(np.mean(agg['pair_cos'][-STEPS:])):.4f} "
                          f"cos_to_mean="
                          f"{float(np.mean(agg['cos_to_mean'][-STEPS:])):.4f}",
                          flush=True)
                    torch.cuda.empty_cache()
            out[name] = dict(
                n_batches=MAX_BATCHES, m_att=M_ATT, steps=STEPS,
                pair_cos_mean=float(np.mean(agg['pair_cos'])),
                dispersion_mean=float(1.0 - np.mean(agg['pair_cos'])),
                norm_mean=float(np.mean(agg['norm_mean'])),
                norm_std_over_mean=float(np.mean(agg['norm_std']) / max(np.mean(agg['norm_mean']), 1e-12)),
                cos_to_mean_mean=float(np.mean(agg['cos_to_mean'])),
                per_batch=dict(pair_cos=agg['pair_cos'],
                               cos_to_mean=agg['cos_to_mean'],
                               norm_mean=agg['norm_mean'],
                               norm_std=agg['norm_std']),
                elapsed_min=round((time.time() - t0) / 60, 1))
            logging.info("[%s] dispersion=%.4f cos_to_mean=%.4f", name,
                         out[name]['dispersion_mean'],
                         out[name]['cos_to_mean_mean'])
            print(f"[s{seed} {name}] dispersion={out[name]['dispersion_mean']:.4f} "
                  f"cos_to_mean={out[name]['cos_to_mean_mean']:.4f}", flush=True)
            torch.cuda.empty_cache()
        with open(out_json, 'w') as f:
            json.dump(dict(config=(f'CIFAR-10, 256 samples (8 batches), '
                                   f'seed {seed}, M_att=5 draws/step, '
                                   f'steps=10, K=2, bs=32, mean-of-CE '
                                   f'instrumented attack (E-E lineage, '
                                   f'disclosed); per-draw gradient statistics, '
                                   f'per-batch arrays included'),
                           results=out), f, indent=2)
        logging.info("written: %s", out_json)
        print(f'written: {out_json}', flush=True)
    print('[E1b recollection complete: seeds 0-2]', flush=True)


if __name__ == '__main__':
    main()
