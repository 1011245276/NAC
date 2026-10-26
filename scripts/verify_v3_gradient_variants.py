# -*- coding: utf-8 -*-
"""V3 acceptance (verification/audit_eot_attack.md section 三·C): STE
graph-connected counterattack reimplementation vs detach baseline.

Structural fact (verified by reading source, pre-registered):
the deployed TTC/NAC counterattack inner loops materialize their trajectory
via `delta.data[:] = d` and `deltas_per_step.append(delta.data.clone())`
(test_time_counterattack.py L134-135, nac_fair_experiment.py L252-253), so
Delta has NO autograd path to its input at all - independent of sign().
A "gradient through the counterattack" therefore cannot be obtained by
patching sign; it requires a forward-equivalent graph-connected
reimplementation.

Variants (proxy gradient explicitly stated, per external-review requirement)
----------------------------------------------------------------------------
detach : deployed functions as-is (Delta detached; direct-term gradient only).
STE    : line-by-line reimplementation of the same inner algorithm with
         exactly three disclosed changes (values unchanged):
         1. `delta.data[:] = d`  -> graph-connected `delta = d`;
         2. every `sign(z)`      -> `ste_sign(z) = sign(z).detach() + z - z.detach()`
            (forward value identical; Jacobian proxied by the IDENTITY);
         3. inner `autograd.grad(..., create_graph=True)` so the outer
            backward can flow through the inner graph.
         Everything else (X_ori_reps no_grad, tau gating, step weighting,
         clamp order) is kept line-identical to the deployed implementation.
BPDA   : coincides with the STE reimplementation here (forward real values,
         backward through the identity proxy of sign) - no separate variant.

Pre-registered rules (before any result):
1. Equivalence gate: same xi seed -> max|Delta_ste - Delta_deployed| < 1e-5
   (2 batches x 2 methods), else the STE reimplementation is invalid: STOP.
2. Direction: cos(g_ste, g_detach) at a fixed delta0 / xi, reported per batch.
3. Strength: 128 samples x PGD-10 x M_att=1 x seed 0; robust accuracy of the
   DEPLOYED defense (single eval, fixed eval seed) after each variant's
   attack; per-sample success vectors saved (V5 practice moved forward).
4. Interpretation bands: |diff| <= 1pp -> detach approximation sufficient;
   STE robust rate LOWER (stronger attack) -> detach OVERSTATES robustness;
   STE rate HIGHER -> detach UNDERSTATES attack strength. Direction and
   magnitude are decided by the data, NOT presumed.

Run (cwd = nac_project):
    python scripts/verify_v3_gradient_variants.py --methods ttc nac
"""
import argparse, json, logging, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, clip_img_preprocessing,
    nac_counterattack, ttc_counterattack,
)
from utils import convert_models_to_fp32

clip_load = clip.load
device = "cuda" if torch.cuda.is_available() else "cpu"

# Double backward through scaled_dot_product_attention is only implemented by
# the MATH backend (efficient/flash lack second derivatives). Scoped to the
# STE inner-loop encode calls only; deployed-function calls keep default
# kernels so the detach baseline matches production numerics.
from torch.nn.attention import sdpa_kernel, SDPBackend


def _sdp_math():
    """Fresh context each use (sdpa_kernel CMs are single-use)."""
    return sdpa_kernel(SDPBackend.MATH)

# ---- pre-registered constants ----
XI_SEED = 1234        # xi for equivalence / direction checks
DELTA_SEED = 7        # attack start delta for direction check
EVAL_SEED = 999       # defense re-run seed for success evaluation
EQ_TOL = 1e-5
N_BATCHES = 8         # 8 x 16 = 128 samples (audit V3 requirement: >=128)
BATCH_SIZE = 16       # smaller than production 32: STE retains inner graphs


def ste_sign(t):
    """Forward value = sign(t); backward = identity (proxy Jacobian)."""
    return torch.sign(t).detach() + t - t.detach()


# --------------------------------------------------------------------------
# Forward-equivalent graph-connected reimplementations (STE). Line-by-line
# mirrors of the deployed loops; only the three disclosed changes above.
# --------------------------------------------------------------------------
def ttc_counterattack_ste(model, X, prompter, add_prompter, alpha, attack_iters,
                          norm, epsilon, tau_thres, beta, create_graph=True):
    lower_limit, upper_limit = 0, 1

    def clamp(X, lower, upper):
        return torch.max(torch.min(X, upper), lower)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta = delta.detach().requires_grad_(True)
    if attack_iters == 0:
        return delta.data

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X)), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]
    diff_ratio = None
    for _step_id in range(attack_iters):
        prompted_images = prompter(clip_img_preprocessing(X + delta))
        with _sdp_math():
            X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        if _step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm
        scheme_sign = ste_sign(tau_thres - diff_ratio)
        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta,
                                   create_graph=create_graph)[0]
        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * ste_sign(g), min=-epsilon, max=epsilon)
        else:
            raise ValueError
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta = d                                    # graph-connected
        deltas_per_step.append(delta.clone() if create_graph
                               else delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0) \
                   .expand(X.size(0), -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights = weights / weights.sum(dim=1, keepdim=True)  # NOT inplace: backward needs exp's output
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)
    return Delta


def nac_counterattack_ste(model, X, prompter, add_prompter, alpha, attack_iters,
                          norm, epsilon, tau_thres, beta, nac_momentum=0.9,
                          create_graph=True):
    lower_limit, upper_limit = 0, 1

    def clamp(X, lower, upper):
        return torch.max(torch.min(X, upper), lower)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta = delta.detach().requires_grad_(True)
    if attack_iters == 0:
        return delta.data

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X)), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]
    diff_ratio = None
    velocity = torch.zeros_like(delta)
    for _step_id in range(attack_iters):
        look_ahead = X + delta + nac_momentum * velocity   # clamp_lookahead=False
        prompted_images = prompter(clip_img_preprocessing(look_ahead))
        with _sdp_math():
            X_att_reps = model.module.encode_image(prompted_images, prompt_token)
        if _step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm
        scheme_sign = ste_sign(tau_thres - diff_ratio)
        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta,
                                   create_graph=create_graph)[0]
        velocity = nac_momentum * velocity + alpha * ste_sign(grad)
        d = delta + velocity
        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        else:
            raise ValueError
        d = clamp(d, lower_limit - X, upper_limit - X)
        delta = d                                    # graph-connected
        deltas_per_step.append(delta.clone() if create_graph
                               else delta.data.clone())

    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0) \
                   .expand(X.size(0), -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights = weights / weights.sum(dim=1, keepdim=True)  # NOT inplace
    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)
    return Delta


def ce_loss(model, prompter, text_tokens, dataset, defended, target):
    output, _, _, _ = multiGPU_CLIP(
        None, None, None, model,
        prompter(clip_img_preprocessing(defended)),
        text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
    return output, F.cross_entropy(output, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--n_batches', type=int, default=N_BATCHES)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    ap.add_argument('--steps', type=int, default=10)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'verify_v3_gradient_variants', stamp)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outdir, 'v3.log'),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("V3 | methods=%s n_batches=%d bs=%d steps=%d",
                 cli.methods, cli.n_batches, cli.batch_size, cli.steps)
    print(f"V3 gradient-variant check; outdir: {outdir}", flush=True)

    # ---- identical init path to the production attacker ----
    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', '32',
        '--test_set', 'cifar10', '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', 'cifar10', '--seed', '0', '--num_seeds', '1',
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
    args.batch_size = cli.batch_size

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
        return dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                    alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                    norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                    beta=args.beta, clip_visual=None,
                    nac_momentum=args.nac_momentum, clamp_lookahead=False)

    deployed = {'ttc': ttc_counterattack, 'nac': nac_counterattack}
    ste = {'ttc': ttc_counterattack_ste, 'nac': nac_counterattack_ste}

    def ste_call(name, x, create_graph=True):
        kw = dict(model=model, X=x, prompter=prompter, add_prompter=add_prompter,
                  alpha=args.ttc_stepsize, attack_iters=args.ttc_numsteps,
                  norm='l_inf', epsilon=args.ttc_eps, tau_thres=args.tau_thres,
                  beta=args.beta, create_graph=create_graph)
        if name == 'nac':
            kw['nac_momentum'] = args.nac_momentum
        return ste[name](**kw)

    eps, alpha, steps = args.ttc_eps, args.ttc_stepsize, cli.steps

    results = {}
    for name in cli.methods:
        per_sample = {'detach': [], 'ste': []}
        norms = {'detach': [], 'ste': []}
        dir_cos, eq_max = [], None

        for bi, (images, target) in enumerate(val_loader):
            if bi >= cli.n_batches:
                break
            images, target = images.to(device), target.to(device)

            # ---- (1) forward-equivalence gate (batch 0 and 1 only) ----
            if bi < 2:
                torch.manual_seed(XI_SEED)
                torch.cuda.manual_seed_all(XI_SEED)
                with torch.no_grad():
                    pass  # deployed fn needs grad mode internally; run as-is
                x_eq = images.clone()
                # Same MATH backend for BOTH sides: the gate tests algorithmic
                # equivalence, not kernel numerics (efficient vs math differ
                # at fp32 rounding level).
                with _sdp_math():
                    Delta_dep = deployed[name](**mk(name, x_eq)).detach()
                torch.manual_seed(XI_SEED)
                torch.cuda.manual_seed_all(XI_SEED)
                Delta_stv = ste_call(name, x_eq, create_graph=False).detach()
                diff = float((Delta_dep - Delta_stv).abs().max())
                eq_max = diff if eq_max is None else max(eq_max, diff)
                logging.info("%s batch %d equivalence max|diff|=%.3e",
                             name, bi, diff)
                print(f"  {name} batch {bi} equivalence max|diff|={diff:.3e}",
                      flush=True)
                if diff >= EQ_TOL:
                    print(f"[{name}] EQUIVALENCE GATE FAILED ({diff:.3e}) "
                          f"- STE reimplementation invalid, STOP", flush=True)
                    results[name] = dict(equivalence_max_diff=diff,
                                         verdict='FAIL_EQUIVALENCE')
                    results['_stop'] = True
                    break

            # ---- (2) direction check at fixed delta0 / xi ----
            torch.manual_seed(DELTA_SEED + bi)
            torch.cuda.manual_seed_all(DELTA_SEED + bi)
            delta0 = torch.clamp(
                torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
            delta0.requires_grad_(True)
            torch.manual_seed(XI_SEED)
            torch.cuda.manual_seed_all(XI_SEED)
            x0 = torch.clamp(images + delta0, 0, 1)
            Delta_d = deployed[name](**mk(name, x0.detach())).detach()
            _, loss_d = ce_loss(model, prompter, text_tokens, 'cifar10',
                                torch.clamp(x0 + Delta_d, 0, 1), target)
            g_d = torch.autograd.grad(loss_d, delta0)[0]
            torch.manual_seed(XI_SEED)
            torch.cuda.manual_seed_all(XI_SEED)
            x0 = torch.clamp(images + delta0, 0, 1)
            Delta_s = ste_call(name, x0, create_graph=True)
            _, loss_s = ce_loss(model, prompter, text_tokens, 'cifar10',
                                torch.clamp(x0 + Delta_s, 0, 1), target)
            g_s = torch.autograd.grad(loss_s, delta0)[0]
            cos = float(F.cosine_similarity(g_d.flatten(), g_s.flatten(), dim=0))
            dir_cos.append(cos)
            logging.info("%s batch %d direction cos=%.4f", name, bi, cos)
            print(f"  {name} batch {bi} direction cos={cos:.4f}", flush=True)

            # ---- (3) attacks: detach baseline and STE, M_att=1 ----
            for variant in ('detach', 'ste'):
                torch.manual_seed(1000 + bi)          # attack start (shared)
                torch.cuda.manual_seed_all(1000 + bi)
                delta = torch.clamp(
                    torch.zeros_like(images).uniform_(-eps, eps), -eps, eps)
                delta.requires_grad_(True)
                for _s in range(steps):
                    x_attack = torch.clamp(images + delta, 0, 1)
                    if variant == 'detach':
                        Delta = deployed[name](**mk(name, x_attack.detach())).detach()
                    else:
                        Delta = ste_call(name, x_attack, create_graph=True)
                    defended = torch.clamp(x_attack + Delta, 0, 1)
                    _, loss = ce_loss(model, prompter, text_tokens, 'cifar10',
                                      defended, target)
                    grad = torch.autograd.grad(loss, delta)[0]
                    delta = torch.clamp(delta + alpha * torch.sign(grad),
                                        -eps, eps).detach()
                    delta.requires_grad_(True)
                delta = delta.detach()
                # ---- success eval with the DEPLOYED defense ----
                torch.manual_seed(EVAL_SEED)
                torch.cuda.manual_seed_all(EVAL_SEED)
                x_adv = torch.clamp(images + delta, 0, 1)
                Delta_ev = deployed[name](**mk(name, x_adv.detach())).detach()
                defended = torch.clamp(x_adv + Delta_ev, 0, 1)
                with torch.no_grad():
                    output, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(defended)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name='cifar10')
                    pred = output.argmax(dim=1)
                    success = (pred != target).cpu().tolist()  # defense failed
                per_sample[variant].extend(success)
                norms[variant].extend(
                    delta.view(delta.size(0), -1).norm(dim=1).cpu().tolist())
                logging.info("%s batch %d variant=%s success=%.3f",
                             name, bi, variant, float(np.mean(success)))
                print(f"  {name} batch {bi} {variant} attack-success="
                      f"{float(np.mean(success)):.3f}", flush=True)

        if results.get('_stop'):
            break

        # ---- aggregate ----
        sd = np.array(per_sample['detach'], dtype=bool)
        ss = np.array(per_sample['ste'], dtype=bool)
        n = len(sd)
        rate = {v: 100.0 * float(np.mean(per_sample[v])) for v in per_sample}
        paired = dict(
            n=int(n),
            both_fail=int(np.sum(sd & ss)),
            only_detach_fail=int(np.sum(sd & ~ss)),
            only_ste_fail=int(np.sum(~sd & ss)),
            neither=int(np.sum(~sd & ~ss)),
        )
        verdict = 'OK'
        results[name] = dict(
            n_samples=n, robust_rate_detach=100.0 - rate['detach'],
            robust_rate_ste=100.0 - rate['ste'],
            attack_success_detach=rate['detach'],
            attack_success_ste=rate['ste'],
            robust_rate_diff_ste_minus_detach=
                (100.0 - rate['ste']) - (100.0 - rate['detach']),
            direction_cos_mean=float(np.mean(dir_cos)),
            direction_cos_per_batch=dir_cos,
            equivalence_max_diff=eq_max,
            paired=paired,
            per_sample_attack_success=per_sample,
            l2_perturb_norm_mean={v: float(np.mean(norms[v]))
                                  for v in norms},
            verdict=verdict)
        logging.info("%s robust detach=%.2f%% ste=%.2f%% diff=%.2fpp cos=%.4f",
                     name, results[name]['robust_rate_detach'],
                     results[name]['robust_rate_ste'],
                     results[name]['robust_rate_diff_ste_minus_detach'],
                     results[name]['direction_cos_mean'])
        print(f"[{name}] robust_detach={results[name]['robust_rate_detach']:.2f}% "
              f"robust_ste={results[name]['robust_rate_ste']:.2f}% "
              f"diff={results[name]['robust_rate_diff_ste_minus_detach']:+.2f}pp "
              f"cos={results[name]['direction_cos_mean']:.4f}", flush=True)

    results.pop('_stop', None)
    with open(os.path.join(outdir, 'v3_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"V3 done -> {outdir}", flush=True)


if __name__ == '__main__':
    main()
