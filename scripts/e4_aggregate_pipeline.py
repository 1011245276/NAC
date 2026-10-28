# -*- coding: utf-8 -*-
"""
E4: defense-side aggregation enhancer (mean-logits / majority-vote) and
aggregate-aware EOT attack, per thesis_proposal/研究二实验计划.md E4 hard
acceptance: the attack objective must match the ACTUAL deployed aggregate
output when M_def > 1.

Targets (v1.4 restructure per external audit ruling 2026-10-26)
---------------------------------------------------------------
Deployed mean-logits defense decision: argmax( (1/M_def) * sum_j logits_j ).
The expectation-domain attack objective matching this deployment is

    J(delta) = E_xi[ CE( (1/M_def) * sum_{j=1..M_def} logits_j(delta; xi_j), y ) ]

estimated with M_att independent repetitions (each repetition draws a FRESH
set of M_def counterattack branches, aggregates them, computes one CE, and
the step loss averages the M_att repetition losses).  For M_def=1 this
reduces exactly to mean-of-CE -- the objective of the audited production EOT
attacker (adaptive_attack_eot.py).  The previous implementation
(CE over the mean of M_att SINGLE-branch logits) equals the M_def->infinity
limit proxy and is NOT the deployed-finite-M_def objective (Jensen gap); it
was ruled non-matching and replaced.

Majority vote: decision non-differentiable; pre-registered soft-vote proxy =
the same per-repetition CE(mean of the M_def branch logits); attack success
is ALWAYS evaluated against the real majority-vote deployment output.

detach truncation: the counterattack output Delta is .detach()-ed exactly as
in the audited production attacker (same disclosed approximation, NOT a full
white-box attack).

Deployment evaluation: M_def fresh-init counterattack runs, independent of
the attack draws; mean-logits and majority predictions are both derived from
the same M_def member logits.  Per-sample records (V5 spec) per
(method, M_def target): adv/defended vectors for both aggregations, norms,
indices, counts, Wilson 95% CI.
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
from mac_port import (
    build_aug as mac_build_aug,
    build_views as mac_build_views,
    aggregate_views as mac_aggregate_views,
    multiview_guided_counterattack as mac_guided_counterattack,
)

clip_load = clip.load
from utils import convert_models_to_fp32

device = "cuda" if torch.cuda.is_available() else "cpu"

N_CLEAN_BATCHES = 8      # S3 clean check: 8 x 32 = 256 samples (audit minimum)
CLEAN_DROP_PP = 0.5      # pre-registered ONE-SIDED constraint 1 (E5)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def eot_attack_aggregate(counter_fn, mk, model, images, target, text_tokens,
                         prompter, dataset, eps, alpha, steps, m_att,
                         m_def_target, member_fn=None):
    """Aggregate-aware EOT attack (v1.4 objective).

    Per attack step: for r in 1..m_att repetitions, draw m_def_target fresh
    counterattack branches on the same x_attack, aggregate their logits by
    the MEAN, compute one CE per repetition, and average the repetition
    losses.  m_def_target=1 reduces to mean-of-CE (production EOT attacker).
    Delta is .detach()-truncated per branch (disclosed approximation).

    member_fn (additive, E3a): when given, ONE repetition = member_fn(x_attack)
    returning the method's deployed logits directly (e.g. one full MAC pass
    with view-mean aggregation); the counter_fn/mk single-delta path is
    skipped.  ttc/nac keep the default path byte-identical (member_fn=None).

    Memory: the step gradient equals the MEAN of the per-repetition gradients
    (gradient linearity -- every repetition loss depends on the same delta
    leaf), so each repetition is built and differentiated on its own freshly
    recomputed x_attack, and its graph is freed before the next repetition.
    This bounds peak memory at ONE repetition (M_def branches) instead of
    M_att x M_def branches (the unbounded-graph version OOM'd on the 8GB
    card at M_def=5 with 16.8GB requested).
    """
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, -eps, eps)
    delta.requires_grad = True
    for _s in range(steps):
        grad_acc = torch.zeros_like(delta)
        for _r in range(m_att):
            # x_attack recomputed per repetition: an independent graph path
            # from delta, freed together with the repetition's backward.
            x_attack = torch.clamp(images + delta, 0, 1)
            member_logits = None
            if member_fn is not None:
                agg_logits = member_fn(x_attack)
            else:
                member_logits = []
                for _j in range(m_def_target):
                    Delta = counter_fn(**mk(x_attack.detach())).detach()
                    defended = torch.clamp(x_attack + Delta, 0, 1)
                    out, _, _, _ = multiGPU_CLIP(
                        None, None, None, model,
                        prompter(clip_img_preprocessing(defended)),
                        text_tokens=text_tokens, prompt_token=None,
                        dataset_name=dataset)
                    member_logits.append(out)
                agg_logits = torch.stack(member_logits, dim=0).mean(dim=0)
            loss_r = F.cross_entropy(agg_logits, target)
            g_r = torch.autograd.grad(loss_r, delta)[0]
            grad_acc = grad_acc + g_r
            del member_logits, agg_logits, loss_r, x_attack
        grad = grad_acc / m_att
        delta = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps).detach()
        delta.requires_grad = True
        del grad_acc, grad
    return delta.detach()


def deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                     dataset, m_def, member_fn=None):
    """Run the deployed aggregate defense: M_def independent counterattacks
    (fresh random inits), return per-member logits [M_def, B, C].

    Counter fns rely on autograd internally (their inner loop calls
    torch.autograd.grad), so each run is wrapped in enable_grad exactly like
    the audited deployment path in adaptive_attack_eot.py; the classification
    forward runs under no_grad.  member_fn (E3a): one member = one full
    method-specific defense pass returning deployed logits; the member's
    counterattack stage still runs under enable_grad, its classification
    forward under no_grad (grad_out=False)."""
    member_logits = []
    for _j in range(m_def):
        if member_fn is not None:
            with torch.enable_grad():
                out_j = member_fn(images, grad_out=False)
            member_logits.append(out_j)
            continue
        with torch.enable_grad():
            Delta = counter_fn(**mk(images))
        with torch.no_grad():
            defended = torch.clamp(images + Delta, 0, 1)
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(defended)),
                text_tokens=text_tokens, prompt_token=None,
                dataset_name=dataset)
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


def _walk_graph(node, names, seen, depth=0):
    """Collect autograd node class names reachable from `node`."""
    if node is None or depth > 128 or id(node) in seen:
        return
    seen.add(id(node))
    names.add(type(node).__name__)
    for nxt, _idx in node.next_functions:
        _walk_graph(nxt, names, seen, depth + 1)


def selfcheck(counter_fn, mk, model, val_loader, text_tokens, prompter,
              dataset, m_def, log, member_fn=None):
    """E4 hard-acceptance preflight (v1.4):
    S1 M_def=1 mean-logits deployment == plain single-run deployment;
    S2 the aggregate node is IN the computed graph (MeanBackward reachable
       from the loss) AND the loss gradient reaches the input perturbation;
    S3 clean-image sanity on >=256 samples, ONE-SIDED pre-registered
       constraint: clean(M_def) >= clean(M_def=1) - 0.5pp for both
       aggregations (E5 constraint 1; small-n smoke cannot prove the
       population bound, only exclude gross implementation bugs).
    member_fn (E3a): the S2 branch logits come from the method-specific
    member (for MAC the aggregation node is the official view-mean)."""
    # S1: identity check on one batch
    it = iter(val_loader)
    images, target = next(it)
    images, target = images.to(device), target.to(device)
    ls1 = deploy_aggregate(counter_fn, mk, model, images, text_tokens, prompter,
                           dataset, 1, member_fn=member_fn)
    pm1, pj1, _ = aggregate_predicts(ls1)
    ok_s1 = bool(torch.equal(pm1, ls1[0].argmax(dim=1)))

    # S2: aggregation node in graph + gradient reaches delta
    d0 = torch.zeros_like(images).uniform_(-0.01, 0.01)
    d0.requires_grad_(True)
    x = torch.clamp(images + d0, 0, 1)
    branch = []
    for _m in range(2):
        if member_fn is not None:
            branch.append(member_fn(x))
            continue
        Delta = counter_fn(**mk(x.detach())).detach()
        branch.append(multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(torch.clamp(x + Delta, 0, 1))),
            text_tokens=text_tokens, prompt_token=None,
            dataset_name=dataset)[0])
    mean_loss = F.cross_entropy(torch.stack(branch).mean(0), target)
    names, seen = set(), set()
    _walk_graph(mean_loss.grad_fn, names, seen)
    g = torch.autograd.grad(mean_loss, d0, allow_unused=True,
                            retain_graph=True)[0]
    grad_reaches_delta = bool(g is not None and
                              torch.isfinite(g).all().item())
    # per-branch propagation: every M_def branch output must receive gradient
    branch_grads = torch.autograd.grad(mean_loss, branch, allow_unused=True,
                                       retain_graph=True)
    all_branches_reached = all(bg is not None for bg in branch_grads)
    ok_s2 = bool(any('MeanBackward' in n for n in names) and
                 grad_reaches_delta and all_branches_reached)

    # S3: one-sided clean check on >=256 samples
    n1m = n1j = 0
    k1m = k1j = kNm = kNj = nNm = nNj = 0
    seen_batches = 0
    for bi, (cim, ctg) in enumerate(it):
        if seen_batches >= N_CLEAN_BATCHES:
            break
        cim, ctg = cim.to(device), ctg.to(device)
        _, pj1b, _ = aggregate_predicts(
            deploy_aggregate(counter_fn, mk, model, cim, text_tokens,
                             prompter, dataset, 1, member_fn=member_fn))
        pmNb, pjNb, _ = aggregate_predicts(
            deploy_aggregate(counter_fn, mk, model, cim, text_tokens,
                             prompter, dataset, m_def, member_fn=member_fn))
        okm = (pj1b == ctg).cpu().tolist()
        n1j += len(okm); k1j += sum(okm)
        okmN = (pmNb == ctg).cpu().tolist()
        nNm += len(okmN); kNm += sum(okmN)
        okjN = (pjNb == ctg).cpu().tolist()
        nNj += len(okjN); kNj += sum(okjN)
        seen_batches += 1
    n1m = n1j  # same batches
    k1m = k1j  # mean-logits at M_def=1 == single run (S1)
    drop_mean = 100.0 * (k1m - kNm) / max(nNm, 1)
    drop_maj = 100.0 * (k1j - kNj) / max(nNj, 1)
    res = dict(S1_mdef1_identity=ok_s1,
               S2_mean_node_in_graph=any('MeanBackward' in n for n in names),
               S2_grad_reaches_delta=grad_reaches_delta,
               S2_all_branch_grads_reached=all_branches_reached,
               S2_node_types_sample=sorted(n for n in names if 'Backward' in n)[:8],
               S3_n=seen_batches * images.size(0),
               S3_clean_mdef1_mean_pct=round(100.0 * k1m / max(n1m, 1), 2),
               S3_clean_mdef1_majority_pct=round(100.0 * k1j / max(n1j, 1), 2),
               S3_clean_mdefN_mean_pct=round(100.0 * kNm / max(nNm, 1), 2),
               S3_clean_mdefN_majority_pct=round(100.0 * kNj / max(nNj, 1), 2),
               S3_drop_pp=dict(mean=round(drop_mean, 2), majority=round(drop_maj, 2)),
               S3_rule='ONE-SIDED: drop <= 0.5pp (pre-registered E5 constraint 1)')
    res['S3_pass'] = bool(drop_mean <= CLEAN_DROP_PP and
                          drop_maj <= CLEAN_DROP_PP)
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
    ap.add_argument('--batch_size', type=int, default=32,
                    help='deployment/attack batch size.  Default 32; the '
                         'E5 driver passes 16 for the M_def=10 configs '
                         '(10-branch graph peak at bs=32, ~7.3GB, thrashes '
                         'the 8GB card ~15x; per-image results are '
                         'stream-shifted per the S1 RNG-stream finding -- '
                         'disclosed deviation, user-approved 2026-10-27).')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--mac_eps', type=float, default=8.0,
                    help='MAC defense budget in /255 units (published 8; '
                         'the matched-budget cell uses 4)')
    ap.add_argument('--mac_numsteps', type=int, default=4,
                    help='MAC PGD inner steps (published 4; the '
                         'matched-budget cell uses 2)')
    cli = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = cli.outdir or os.path.join('results', 'e4_aggregate', stamp)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, 'e4.log')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    sys.argv = [
        'nac_fair_experiment.py', '--evaluate', 'True', '--batch_size', str(cli.batch_size),
        '--test_set', cli.dataset, '--test_attack_type', 'pgd', '--test_eps', '4.0',
        '--test_numsteps', '10', '--test_stepsize', '1.0', '--method', 'null_patch',
        '--root', './data', '--dataset', cli.dataset, '--seed', '0', '--num_seeds', '1',
        '--tau_thres', '0.2', '--beta', '2.0', '--ttc_eps', '4.0',
        '--ttc_numsteps', '2', '--ttc_stepsize', '1.0', '--nac_momentum', '0.9',
        '--learnable_tau', '0.155', '--temperature', '75.0',
        '--counterattack', 'ttc', '--outdir', outdir,
        '--mac_eps', str(cli.mac_eps), '--mac_numsteps', str(cli.mac_numsteps),
        '--mac_num_views', '2',
        '--mac_tau_thres', '0.7', '--mac_tau_temp', '0.01',
    ]
    from nac_fair_experiment import parse_options
    args = parse_options()
    args.test_eps = args.test_eps / 255.
    args.test_stepsize = args.test_stepsize / 255.
    args.ttc_eps = args.ttc_eps / 255.
    args.ttc_stepsize = args.ttc_stepsize / 255.
    args.mac_eps = args.mac_eps / 255.   # MAC published budget (8/255)
    mac_stepsize = args.mac_eps / 4.0    # official mac.py: stepsize = eps/4
    args.imagenet_root = os.path.join(args.root, 'imagenet-100', 'imagenet_folder')
    args.tinyimagenet_root = os.path.join(args.root, 'tiny-imagenet-200', 'tiny-imagenet-200')

    m_def_max = max(cli.m_def_list)
    mac_note = (f" | MAC defense: eps={cli.mac_eps:g}/255, V=2 views, "
                f"K={cli.mac_numsteps}, stepsize=eps/4, tau_thres=0.7, "
                f"tau_temp=0.01"
                if 'mac' in cli.methods else '')
    logging.info("E4 aggregate pipeline v1.4 | M_att=%d M_def=%s steps=%d methods=%s seeds=%s",
                 cli.m_att, cli.m_def_list, cli.steps, cli.methods, cli.seeds)
    print(f"E4 aggregate pipeline v1.4 M_att={cli.m_att} M_def={cli.m_def_list}; log:", log_path)

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
    mac_aug = mac_build_aug() if 'mac' in cli.methods else None

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

    def mac_member(x, grad_out=True):
        """One full published-config MAC defense pass -> deployed logits
        (E3a member).  views = [clean view] + (V-1) augmented views (official
        build_views, no_grad); per-view guided-counterattack deltas detached
        (same disclosed truncation as the production attacker); defended
        views formed WITHOUT an outer clamp (faithful to official mac.py);
        logits aggregated by the official view-mean.  Attack-side gradient
        flows through the un-augmented view only (build_views is no_grad)
        while the loss keeps the full view-mean expectation -- disclosed in
        the plan pre-registration."""
        V = int(args.mac_num_views)
        views = mac_build_views(x, V, mac_aug)
        with torch.enable_grad():
            delta_v = mac_guided_counterattack(
                model, views.data, prompter, add_prompter,
                alpha=mac_stepsize, attack_iters=args.mac_numsteps,
                norm='l_inf', epsilon=args.mac_eps,
                tau_thres=args.mac_tau_thres, tau_temp=args.mac_tau_temp,
                clip_visual=None, aug=mac_aug).detach()
        def_im = views + delta_v
        if grad_out:
            out, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(def_im)),
                text_tokens=text_tokens, prompt_token=None,
                dataset_name=cli.dataset)
        else:
            with torch.no_grad():
                out, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(def_im)),
                    text_tokens=text_tokens, prompt_token=None,
                    dataset_name=cli.dataset)
        return mac_aggregate_views(out, V)

    members = {'mac': mac_member}
    results = {}
    out_json = os.path.join(outdir, 'e4_results.json')
    for seed in cli.seeds:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for name in cli.methods:
            fn = fns.get(name)
            mk_fn = lambda x, _n=name: mk(_n, x)
            member_fn = members.get(name)
            logging.info(f"=== seed {seed} | E4 aggregate vs {name} ===")
            # hard-acceptance preflight (S1/S2 single batch; S3 8 batches)
            chk = selfcheck(fn, mk_fn, model, val_loader, text_tokens,
                            prompter, cli.dataset, m_def_max, logging,
                            member_fn=member_fn)
            for md in cli.m_def_list:
                t0 = time.time()
                tallies = dict(mean=dict(n=0, k=0, vec=[]),
                               majority=dict(n=0, k=0, vec=[]))
                adv_tally = dict(n=0, k=0, vec=[])
                ps_l2, ps_linf, ps_index = [], [], []
                eval_time_s = 0.0
                for bi, (images, target) in enumerate(val_loader):
                    if cli.max_batches and bi >= cli.max_batches:
                        break
                    images, target = images.to(device), target.to(device)
                    delta_atk = eot_attack_aggregate(
                        fn, mk_fn, model, images, target, text_tokens,
                        prompter, cli.dataset, args.test_eps,
                        args.test_stepsize, cli.steps, cli.m_att, md,
                        member_fn=member_fn)
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
                        te = time.time()
                        ls = deploy_aggregate(fn, mk_fn, model, attacked,
                                                  text_tokens, prompter,
                                                  cli.dataset, md,
                                                  member_fn=member_fn)
                        eval_time_s += time.time() - te
                        pm, pj, _ = aggregate_predicts(ls)
                        okm = (pm == target).cpu().tolist()
                        okj = (pj == target).cpu().tolist()
                        tallies['mean']['n'] += len(okm)
                        tallies['mean']['k'] += sum(okm)
                        tallies['mean']['vec'].extend(okm)
                        tallies['majority']['n'] += len(okj)
                        tallies['majority']['k'] += sum(okj)
                        tallies['majority']['vec'].extend(okj)
                    if bi % 10 == 0:
                        msg = {a: round(tallies[a]['k'] / max(tallies[a]['n'], 1) * 100, 2)
                               for a in ('mean', 'majority')}
                        logging.info(f"  seed {seed} {name} Mdef{md} batch {bi}: {msg}")
                        print(f"  seed {seed} {name} Mdef{md} batch {bi}: {msg}", flush=True)
                        torch.cuda.empty_cache()
                key = f"{name}_seed{seed}_Mdef{md}"
                aggs = {}
                for agg in ('mean', 'majority'):
                    t = tallies[agg]
                    lo, hi = wilson_ci(t['k'], t['n'])
                    aggs[f"M{md}_{agg}"] = dict(
                        robust=round(t['k'] / max(t['n'], 1) * 100, 2),
                        k=t['k'], n=t['n'],
                        wilson95=[round(lo * 100, 2), round(hi * 100, 2)],
                        per_sample_defended_correct=t['vec'])
                entry = dict(
                    adv=round(adv_tally['k'] / max(adv_tally['n'], 1) * 100, 2),
                    per_sample=dict(
                        n=adv_tally['n'],
                        adv_correct=adv_tally['vec'],
                        l2_norm=[round(v, 4) for v in ps_l2],
                        linf_norm=[round(v, 6) for v in ps_linf],
                        image_index=ps_index),
                    selfcheck=chk,
                    deployment_eval_seconds=round(eval_time_s, 1),
                    attack_config=dict(
                        method=name, seed=seed, m_att=cli.m_att,
                        m_def_target=md, steps=cli.steps,
                        eps_att=round(args.test_eps, 6),
                        objective=('E_xi[CE(mean of M_def-branch aggregate)] '
                                   'estimated with M_att repetitions; '
                                   'M_def=1 reduces to mean-of-CE (production '
                                   'EOT attacker objective); detach-truncated '
                                   'disclosed'),
                        proxy_majority=('soft-vote per repetition '
                                        '(CE of mean branch logits); success '
                                        'always evaluated on the real vote')),
                    aggregations=aggs)
                results[key] = entry
                if name == 'mac':
                    results[key]['attack_config'].update(
                        mac_eps=f'{cli.mac_eps:g}/255'
                                + (' (published)' if cli.mac_eps == 8.0 else
                                   ' (matched-budget; published is 8/255)'),
                        mac_stepsize=f'{cli.mac_eps / 4:g}/255 (= eps/4, '
                                     f'official mac.py)',
                        mac_numsteps=int(args.mac_numsteps),
                        mac_num_views=int(args.mac_num_views),
                        mac_tau_thres=args.mac_tau_thres,
                        mac_tau_temp=args.mac_tau_temp,
                        defense_disclosure=(
                            'one member = one full published MAC pass '
                            '(view-mean logits, faithful mac_port.py); '
                            'per-view counterattack deltas detached (same '
                            'disclosed truncation as the production '
                            'attacker); official build_views runs under '
                            'no_grad so the attack gradient flows through '
                            'the un-augmented view only, while the loss '
                            'keeps the full view-mean expectation; defended '
                            'views formed without an outer clamp '
                            '(faithful to official mac.py)'))
                logging.info(f"{key}: adv={entry['adv']} "
                             + json.dumps({k: {kk: vv for kk, vv in v.items()
                                               if kk != 'per_sample_defended_correct'}
                                           for k, v in aggs.items()}))
                print(f"{key}: adv={entry['adv']} "
                      + json.dumps({k: {kk: vv for kk, vv in v.items()
                                        if kk != 'per_sample_defended_correct'}
                                    for k, v in aggs.items()}), flush=True)
                with open(out_json, 'w') as f:
                    json.dump(dict(
                        config=(f"{cli.dataset.upper()}, eps_att=4/255, PGD-{cli.steps} "
                                f"step=1/255, K=2, alpha=1/255, eps_def=4/255, bs={cli.batch_size}, "
                                f"M_att={cli.m_att} repetitions x full M_def-branch "
                                f"aggregate attack target (v1.4; M_def=1 -> "
                                f"mean-of-CE), deployment M_def={cli.m_def_list} "
                                f"fresh inits, majority tie-break=smallest label "
                                f"(disclosed), seeds={cli.seeds}" + mac_note),
                        results=results), f, indent=2)
                torch.cuda.empty_cache()
    logging.info("DONE")
    print("DONE:", flush=True)


if __name__ == '__main__':
    main()
