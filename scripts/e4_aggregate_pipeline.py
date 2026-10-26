# -*- coding: utf-8 -*-
"""
E4: defense-side aggregation enhancer (mean-logits / majority-vote) and
aggregate-aware EOT attack, per thesis_proposal/研究二实验计划.md E4 hard
acceptance: the attack objective must match the ACTUAL deployed aggregate
output when M_def > 1.

Targets (pre-registered, disclosed):
  * mean_logits deployment : pred = argmax( (1/M_def) * sum_m logits_m ).
    Attack objective: CE( mean_{m=1..M_att} logits_m , y ) -- gradient flows
    through the averaged logits (NOT the mean of per-member CE losses used by
    adaptive_attack_eot.py, which is correct only for M_def=1 deployment).
  * majority deployment    : pred_m = argmax(logits_m) per member, final pred =
    torch.mode over members (deterministic smallest-label tie-break,
    DISCLOSED; degenerate for M_def=2 two-way splits -- reported, not hidden).
    Attack proxy (pre-registered): soft vote CE( mean_m logits_m , y ) --
    gradient-identical to the mean-logits objective; attack success is ALWAYS
    evaluated against the real majority-vote deployment output.

Protocol: CIFAR-10, ViT-B/32, eps_att=4/255, PGD-10 step=1/255, K=2,
alpha=1/255, eps_def=4/255, bs=32; M_att EOT draws per attack step; deployment
eval reruns the counterattack M_def times with fresh random inits (independent
of the attack draws); per-sample records per (method, M_def, aggregation) per
audit V5 spec; Wilson 95% CI computed inline.
"""
import sys, os, json, logging, time, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import random

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from nac_fair_experiment import (
    load_val_dataset, get_text_prompts_val,
    multiGPU_CLIP, clip_img_preprocessing,
    nac_counterattack, ttc_counterattack,
)

clip_load = clip.load
from utils import convert_models_to_fp32

device = "cuda" if torch.cuda.is_available() else "cpu"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def eot_attack_mean_logits(counter_fn, mk, model, images, target, text_tokens,
                           prompter, dataset, eps, alpha, steps, m_att):
    """Aggregate-aware EOT attack: objective = CE(mean_m logits_m, y).

    The counterattack Delta is re-run from m_att fresh random inits per step
    (the counter functions re-initialize their internal delta via uniform_ on
    every call); Delta is .detach()-truncated exactly like the audited
    production attacker (same disclosed approximation, NOT a full white-box
    attack).  Gradient flows through the MEAN LOGITS aggregation, matching the
    deployed mean-logits defense and serving as the pre-registered soft-vote
    proxy for the majority defense.
    """
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)
    delta.requires_grad = True
    for _s in range(steps):
        x_attack = torch.clamp(images + delta, 0, 1)
        logits = []
        for _m in range(m_att):
            Delta = counter_fn(**mk(x_attack.detach())).detach()
            defended = torch.clamp(x_attack + Delta, 0, 1)
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
            logits.append(out)
        mean_logits = torch.stack(logits, dim=0).mean(dim=0)
        loss = F.cross_entropy(mean_logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps).detach()
        delta.requires_grad = True
    return delta.detach()


def deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                     dataset, m_def):
    """Run the deployed aggregate defense: M_def independent counterattacks
    (fresh random inits), return per-member logits [M_def, B, C].

    Counter fns rely on autograd internally (their inner loop calls
    torch.autograd.grad), so each run is wrapped in enable_grad exactly like
    the audited deployment path in adaptive_attack_eot.py; the classification
    forward runs under no_grad."""
    member_logits = []
    for _j in range(m_def):
        with torch.enable_grad():
            Delta = counter_fn(**mk(images))
        with torch.no_grad():
            defended = torch.clamp(images + Delta, 0, 1)
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None, dataset_name=dataset)
        member_logits.append(out)
    return torch.stack(member_logits, dim=0)


def aggregate_predicts(logits_stack):
    """Two aggregation rules over member logits [M_def, B, C].

    mean_logits : argmean over members.
    majority    : per-member argmax, then torch.mode over members (ties ->
                  smallest label; deterministic, disclosed).
    Returns (pred_mean, pred_majority, member_preds [M_def, B]).
    """
    pred_mean = logits_stack.mean(dim=0).argmax(dim=1)
    member_preds = logits_stack.argmax(dim=2)          # [M_def, B]
    modes = torch.mode(member_preds, dim=0).values      # smallest-label tie-break
    return pred_mean, modes, member_preds


def selfcheck(counter_fn, mk, model, val_iter, text_tokens, prompter,
              dataset, m_def, log):
    """E4 hard-acceptance preflight (logged, results embedded in JSON):
    S1 M_def=1 mean-logits deployment == plain single-run deployment;
    S2 attack loss graph carries the aggregation (mean backward);
    S3 clean-image sanity: |clean(M_def) - clean(1)| <= 0.5pp (E5 constraint 1)
       for both aggregations, computed on the selfcheck batches."""
    images, target = next(val_iter)
    images, target = images.to(device), target.to(device)
    # S1: identity check
    ls1 = deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                           dataset, 1)
    pm1, pj1, _ = aggregate_predicts(ls1)
    ok_s1 = bool(torch.equal(pm1, ls1[0].argmax(dim=1)))
    # S2: graph carries mean aggregation
    x = images.clone().requires_grad_(True)
    Delta = counter_fn(**mk(x.detach())).detach()
    defended = torch.clamp(x + Delta, 0, 1)
    out, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                 prompter(clip_img_preprocessing(defended)),
                                 text_tokens=text_tokens, prompt_token=None,
                                 dataset_name=dataset)
    loss2 = F.cross_entropy(out, target)  # single-member first
    ls2 = []
    for _m in range(2):
        Dm = counter_fn(**mk(x.detach())).detach()
        dm = torch.clamp(x + Dm, 0, 1)
        om, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                    prompter(clip_img_preprocessing(dm)),
                                    text_tokens=text_tokens, prompt_token=None,
                                    dataset_name=dataset)
        ls2.append(om)
    mean_loss = F.cross_entropy(torch.stack(ls2).mean(0), target)
    ok_s2 = bool(mean_loss.requires_grad and
                 mean_loss.grad_fn is not None and
                 'MeanBackward' in str(type(mean_loss.grad_fn)))
    # S3: clean sanity at M_def vs 1 (both aggregations)
    lsN = deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                           dataset, m_def)
    pmN, pjN, _ = aggregate_predicts(lsN)
    acc = lambda p: (p == target).float().mean().item() * 100
    res = dict(S1_mdef1_identity=ok_s1,
               S2_mean_graph=ok_s2,
               S3_clean_mdef1=dict(mean=round(acc(pm1), 2), majority=round(acc(pj1), 2)),
               S3_clean_mdefN=dict(mean=round(acc(pmN), 2), majority=round(acc(pjN), 2)),
               S3_delta_pp=dict(mean=round(acc(pmN) - acc(pm1), 2),
                                majority=round(acc(pjN) - acc(pj1), 2)))
    res['S3_pass'] = bool(abs(res['S3_delta_pp']['mean']) <= 0.5 and
                          abs(res['S3_delta_pp']['majority']) <= 0.5)
    log.info("E4 selfcheck: " + json.dumps(res))
    print("E4 selfcheck:", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=[0])
    ap.add_argument('--methods', nargs='*', default=['ttc', 'nac'])
    ap.add_argument('--m_att', type=int, default=5)
    ap.add_argument('--m_def_list', type=int, nargs='*', default=[1, 5])
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--max_batches', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = cli.outdir or os.path.join('results', 'e4_aggregate', stamp)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'e4.log')
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

    m_def_max = max(cli.m_def_list)
    logging.info("E4 aggregate pipeline | M_att=%d M_def=%s steps=%d methods=%s seeds=%s",
                 cli.m_att, cli.m_def_list, cli.steps, cli.methods, cli.seeds)
    print(f"E4 aggregate pipeline M_att={cli.m_att} M_def={cli.m_def_list}; log:", log_path)

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

    fns = {'ttc': ttc_counterattack, 'nac': nac_counterattack}
    results = {}
    out_json = os.path.join(outdir, 'e4_results.json')
    for seed in cli.seeds:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for name in cli.methods:
            fn = fns[name]
            mk_fn = lambda x, _n=name: mk(_n, x)
            t0 = time.time()
            logging.info(f"=== seed {seed} | E4 aggregate vs {name} ===")
            # hard-acceptance preflight (2 batches of evidence, cheap)
            val_iter = iter(val_loader)
            chk = selfcheck(fn, mk_fn, model, val_iter, text_tokens, prompter,
                            cli.dataset, m_def_max, logging)
            # per (M_def, aggregation) tallies + per-sample records
            tallies = {md: dict(mean=dict(n=0, k=0, vec=[]),
                                majority=dict(n=0, k=0, vec=[]))
                       for md in cli.m_def_list}
            adv_tally = dict(n=0, k=0, vec=[])
            ps_l2, ps_linf, ps_index = [], [], []
            eval_time_s = 0.0
            for bi, (images, target) in enumerate(val_loader):
                if cli.max_batches and bi >= cli.max_batches:
                    break
                images, target = images.to(device), target.to(device)
                delta_atk = eot_attack_mean_logits(
                    fn, mk_fn, model, images, target, text_tokens, prompter,
                    cli.dataset, args.test_eps, args.test_stepsize, cli.steps,
                    cli.m_att)
                attacked = torch.clamp(images + delta_atk, 0, 1)
                with torch.no_grad():
                    out_adv, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(attacked)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name=cli.dataset)
                    ok_adv = (out_adv.argmax(1) == target).cpu().tolist()
                    adv_tally['n'] += len(ok_adv)
                    adv_tally['k'] += sum(ok_adv)
                    adv_tally['vec'].extend(ok_adv)
                    ps_l2.extend(delta_atk.view(delta_atk.size(0), -1).norm(dim=1).cpu().tolist())
                    ps_linf.extend(delta_atk.view(delta_atk.size(0), -1).abs().amax(dim=1).cpu().tolist())
                    ps_index.extend((bi * args.batch_size + torch.arange(images.size(0))).tolist())
                    # deployed aggregate evaluation, fresh inits per M_def config
                    for md in cli.m_def_list:
                        te = time.time()
                        ls = deploy_aggregate(fn, mk_fn, model, attacked,
                                              text_tokens, prompter,
                                              cli.dataset, md)
                        eval_time_s += time.time() - te
                        pm, pj, _ = aggregate_predicts(ls)
                        okm = (pm == target).cpu().tolist()
                        okj = (pj == target).cpu().tolist()
                        tallies[md]['mean']['n'] += len(okm)
                        tallies[md]['mean']['k'] += sum(okm)
                        tallies[md]['mean']['vec'].extend(okm)
                        tallies[md]['majority']['n'] += len(okj)
                        tallies[md]['majority']['k'] += sum(okj)
                        tallies[md]['majority']['vec'].extend(okj)
                if bi % 20 == 0:
                    msg = {f"M{md}": {a: round(t[a]['k'] / max(t[a]['n'], 1) * 100, 2)
                                      for a in ('mean', 'majority')}
                           for md, t in tallies.items()}
                    logging.info(f"  seed {seed} {name} batch {bi}: {msg}")
                    print(f"  seed {seed} {name} batch {bi}: {msg}", flush=True)
                    torch.cuda.empty_cache()
            key = f"{name}_seed{seed}"
            entry = dict(adv=round(adv_tally['k'] / max(adv_tally['n'], 1) * 100, 2),
                         per_sample=dict(
                             n=adv_tally['n'],
                             adv_correct=adv_tally['vec'],
                             l2_norm=[round(v, 4) for v in ps_l2],
                             linf_norm=[round(v, 6) for v in ps_linf],
                             image_index=ps_index),
                         selfcheck=chk,
                         deployment_eval_seconds=round(eval_time_s, 1),
                         attack_config=dict(method=name, seed=seed,
                                            m_att=cli.m_att,
                                            m_def_list=cli.m_def_list,
                                            steps=cli.steps,
                                            eps_att=round(args.test_eps, 6),
                                            proxy='CE(mean logits) shared by '
                                                  'mean-logits target and '
                                                  'majority soft-vote proxy '
                                                  '(pre-registered)'),
                         aggregations={})
            for md, t in tallies.items():
                for agg in ('mean', 'majority'):
                    k, n = t[agg]['k'], t[agg]['n']
                    lo, hi = wilson_ci(k, n)
                    entry['aggregations'][f"M{md}_{agg}"] = dict(
                        robust=round(k / max(n, 1) * 100, 2),
                        wilson95=[round(lo * 100, 2), round(hi * 100, 2)],
                        n=n)
            results[key] = entry
            logging.info(f"{key}: adv={entry['adv']} "
                         + json.dumps(entry['aggregations']))
            print(f"{key}: adv={entry['adv']} "
                  + json.dumps(entry['aggregations']), flush=True)
            with open(out_json, 'w') as f:
                json.dump(dict(
                    config=(f"{cli.dataset.upper()}, eps_att=4/255, PGD-{cli.steps} "
                            f"step=1/255, K=2, alpha=1/255, eps_def=4/255, bs=32, "
                            f"M_att={cli.m_att} aggregate-aware EOT attack "
                            f"(CE of mean logits; detach-truncated disclosed), "
                            f"deployment M_def={cli.m_def_list} with fresh inits, "
                            f"majority tie-break=smallest label (disclosed), "
                            f"seeds={cli.seeds}"),
                    results=results), f, indent=2)
            torch.cuda.empty_cache()
    logging.info("DONE")
    print("DONE:", flush=True)


if __name__ == '__main__':
    main()
