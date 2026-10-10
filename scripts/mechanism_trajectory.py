"""
Mechanism trajectory diagnostic (SMALL, ~5 batches).

Question B/C/D: does NAC change the counterattack trajectory / optimisation
efficiency, and is there a verifiable mechanism?

Observation: TTC's step update (delta += alpha*sign(grad), fwd at X+delta) is
EXACTLY NAC with nac_momentum=0 (look-ahead point reduces to X+delta, velocity
reduces to alpha*sign(grad)). So a single instrumented function parameterised
by momentun covers both, guaranteeing bitwise-fair comparison of the same code
path:
  - nac_momentum=0.0  -> TTC (plain PGD counter-attack)
  - nac_momentum=0.9  -> NAC (Nesterov look-ahead)

Per-step trajectory recorded for the counter-attack objective
    J_t = || f(X + look_ahead_t; token) - f(X_ori) ||^2  (batch-mean)
plus gradient-direction consistency cos(g_t, g_{t-1}).

Output: results/adaptive_pilot/mechanism_traj.npz + .json
"""
from __future__ import print_function

import os
import sys
import json

import numpy as np
import torch

_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)

from nac_fair_experiment import parse_options  # noqa: E402
from func import device, clip_img_preprocessing, multiGPU_CLIP  # noqa: E402
from models.prompters import TokenPrompter, NullPrompter  # noqa: E402
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32  # noqa: E402
from attacks import attack_pgd  # noqa: E402
from replace import clip  # noqa: E402


@torch.no_grad()
def embed(model, prompter, images, text_tokens, dataset_name):
    logits, _, _, _ = multiGPU_CLIP(
        None, None, None, model, prompter(clip_img_preprocessing(images)),
        text_tokens, prompt_token=None, dataset_name=dataset_name)
    return logits


def per_cos(a, b):
    a = a.detach().reshape(a.size(0), -1).float()
    b = b.detach().reshape(b.size(0), -1).float()
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def instrumented_ca(model, X, prompter, add_prompter, alpha, attack_iters,
                    epsilon, tau_thres, beta, nac_momentum):
    """One function: nac_momentum=0 -> TTC, >0 -> NAC. Records per-step traj."""
    lower_limit, upper_limit = 0., 1.

    def clamp(X_, lower, upper):
        return torch.max(torch.min(X_, upper), lower)

    delta = torch.zeros_like(X)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    tunable = [n for n, p in model.module.named_parameters() if p.requires_grad]
    for n, p in model.module.named_parameters():
        p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori = model.module.encode_image(
            prompter(clip_img_preprocessing(X)), prompt_token)
        X_ori_norm = torch.norm(X_ori, dim=-1)

    velocity = torch.zeros_like(delta)
    traj = {'obj': [], 'gcos_cons': [], 'feat_jump': []}
    prev_grad = None
    prev_feat = None

    for step in range(attack_iters):
        look_ahead = X + delta + nac_momentum * velocity
        feat = model.module.encode_image(
            prompter(clip_img_preprocessing(look_ahead)), prompt_token)
        if step == 0:
            diff_ratio = (torch.norm(feat - X_ori, dim=-1) / X_ori_norm)
        scheme = (tau_thres - diff_ratio).sign()
        obj = ((((feat - X_ori) ** 2).sum(1))).mean().detach().item()
        traj['obj'].append(obj)
        l2 = ((((feat - X_ori) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2, delta)[0]
        if prev_grad is not None:
            traj['gcos_cons'].append(per_cos(grad, prev_grad).mean().item())
        if prev_feat is not None:
            traj['feat_jump'].append(
                (feat - prev_feat).float().pow(2).sum(1).mean().item())
        prev_grad, prev_feat = grad.detach(), feat.detach()

        velocity = nac_momentum * velocity + alpha * torch.sign(grad)
        d = delta + velocity
        d = torch.clamp(d, min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - X, upper_limit - X)
        delta.data[:] = d

    for n, p in model.module.named_parameters():
        if n in tunable:
            p.requires_grad = True
    return delta.data, traj


def main():
    sys.argv = [
        'mech', '--batch_size', '32', '--num_workers', '1',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd',
        '--test_eps', '4', '--test_numsteps', '10', '--test_stepsize', '1',
        '--ttc_eps', '4', '--ttc_numsteps', '2', '--ttc_stepsize', '1',
        '--tau_thres', '0.2', '--beta', '2.0',
        '--seed', '0', '--outdir', os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot'),
    ]
    args = parse_options()
    args.test_eps /= 255.
    args.test_stepsize /= 255.
    args.ttc_stepsize /= 255.
    args.ttc_eps /= 255.

    outdir = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
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
    crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    K, eps = 4, 4. / 255.
    n_batch = 5
    acc = {'ttc': {k: [] for k in range(1, K + 1)},
           'nac': {k: [] for k in range(1, K + 1)}}
    # per-step objective, acc (objective minimised *during* counter-attack is
    # not monotone in defended accuracy, but equal-compute comparison is fair)
    for bi, (images, target) in enumerate(val_loader):
        if bi >= n_batch:
            break
        images = images.to(device); target = target.to(device)
        with torch.amp.autocast('cuda'):
            delta_atk = attack_pgd(
                args, prompter, model, None, None, add_prompter, crit,
                images, target, args.test_stepsize, args.test_numsteps, 'l_inf',
                text_tokens=text_tokens, epsilon=args.test_eps,
                dataset_name='cifar10')
            attacked = images + delta_atk
        for name, mom in (('ttc', 0.0), ('nac', 0.9)):
            for k in range(1, K + 1):
                torch.manual_seed(10000 + bi)
                with torch.amp.autocast('cuda'):
                    dd, _ = instrumented_ca(
                        model, attacked.data, prompter, add_prompter,
                        alpha=args.ttc_stepsize, attack_iters=k, epsilon=eps,
                        tau_thres=args.tau_thres, beta=args.beta,
                        nac_momentum=mom)
                    with torch.no_grad():
                        lg = embed(model, prompter, attacked + dd,
                                   text_tokens, 'cifar10').float()
                acc[name][k].append((lg.argmax(1) == target).cpu().numpy())
        # full-K trajectory (momentum only) for per-step obj on same batch
        with torch.amp.autocast('cuda'):
            torch.manual_seed(10000 + bi)
            _, traj_t = instrumented_ca(
                model, attacked.data, prompter, add_prompter,
                alpha=args.ttc_stepsize, attack_iters=K, epsilon=eps,
                tau_thres=args.tau_thres, beta=args.beta, nac_momentum=0.0)
            torch.manual_seed(10000 + bi)
            _, traj_n = instrumented_ca(
                model, attacked.data, prompter, add_prompter,
                alpha=args.ttc_stepsize, attack_iters=K, epsilon=eps,
                tau_thres=args.tau_thres, beta=args.beta, nac_momentum=0.9)
        print('[mech] batch %d: TTC obj-end=%.3f NAC obj-end=%.3f '
              '(K=%d first-step cos: TTC=%.2f NAC=%.2f)' %
              (bi, traj_t['obj'][-1], traj_n['obj'][-1], K,
               traj_t['gcos_cons'][0] if traj_t['gcos_cons'] else float('nan'),
               traj_n['gcos_cons'][0] if traj_n['gcos_cons'] else float('nan')),
              flush=True)

    out = {'acc': {'ttc': {k: float(np.mean(acc['ttc'][k]) * 100) for k in range(1, 5)},
                   'nac': {k: float(np.mean(acc['nac'][k]) * 100) for k in range(1, 5)}}}
    base = None
    # recompute a sample-batch trajectory average is in traj_t/n per batch; keep last
    out['traj_ttc'] = traj_t
    out['traj_nac'] = traj_n
    np.savez_compressed(os.path.join(outdir, 'mechanism_traj.npz'),
                        **{'ttc_acc_k%d' % k: np.concatenate(acc['ttc'][k]) for k in range(1, 5)},
                        **{'nac_acc_k%d' % k: np.concatenate(acc['nac'][k]) for k in range(1, 5)})
    with open(os.path.join(outdir, 'mechanism_traj.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print('[mech] 5-batch acc: TTC%s NAC%s' % (
        [round(out['acc']['ttc'][k], 2) for k in range(1, 5)],
        [round(out['acc']['nac'][k], 2) for k in range(1, 5)]))
    print('[mech] ttc obj per step:', [round(x, 3) for x in traj_t['obj']])
    print('[mech] nac obj per step:', [round(x, 3) for x in traj_n['obj']])
    print('[mech] ttc cos-cons:', [round(x, 3) for x in traj_t['gcos_cons']])
    print('[mech] nac cos-cons:', [round(x, 3) for x in traj_n['gcos_cons']])
    print('[mech] ALL DONE')


if __name__ == '__main__':
    main()