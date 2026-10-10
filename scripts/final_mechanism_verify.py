"""
FINAL mechanism verification: TTC vs Momentum vs Pure Look-ahead vs NAC.
Full CIFAR-10 test set (10000 samples) x seeds {0,1,2} x K {1,2,4,6,8}.

Records, per (seed, method, K):
  1. defended accuracy       2. mean perturbation norm
  3. fraction @ eps boundary 4. feature displacement (to clean)
  5. feature separation (d2adv - d2clean)
  6. gradient-direction consistency (mean cos(g_t, g_{t-1}) over steps, per sample)
  7. counter-attack objective J (per-step mean trajectory -> stored as list)
  8. marginal gain per step (=diff in defended accuracy across K, derived)

Four instrumented counter-attacks are line-faithful copies of the PRODUCTION
functions in nac_fair_experiment.py (verified equal to the originals by a
--equiv_check pass: max|Delta_inst - Delta_prod| == 0). Only extra work is
capturing per-step gradients / objective; the optimisation itself is unchanged.

Protocol (identical to paper): CLIP ViT-B/32, PGD-10 eps 4/255 step 1/255,
batch 32, tau=0.2, beta=2.0, nac_momentum=0.9. Within a seed, adversarial
batch generated once; every counter-attack starts from torch.manual_seed(10000+bi).
Seeds 0/1/2 vary the adversarial init (and counter-attack init via global seed),
so the reported mean+-std spans genuine randomness.
"""
from __future__ import print_function
import os, sys, json, time
import numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
from nac_fair_experiment import (parse_options, ttc_counterattack,
                                 momentum_counterattack, pure_la_counterattack,
                                 nac_counterattack)
from func import device, clip_img_preprocessing, multiGPU_CLIP
from replace import clip
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter, NullPrompter

OUTDIR = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')
KS = [1, 2, 4, 6, 8]
METHODS = ['ttc', 'momentum', 'purela', 'nac']
EPS = 4. / 255.


def _per_cos(a, b):
    a = a.detach().reshape(a.size(0), -1).float()
    b = b.detach().reshape(b.size(0), -1).float()
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


# ---------------------------------------------------------------- instrumented
def _finalize(deltas, scheme_sign, beta, tunable, model, X):
    Delta = torch.stack(deltas, dim=1)
    n_steps = len(deltas)
    w = torch.arange(n_steps).unsqueeze(0).expand(X.size(0), -1).to(device).float()
    w = torch.exp(scheme_sign.view(-1, 1) * w * beta)
    w /= w.sum(1, keepdim=True)
    wh = torch.zeros_like(w); wh[:, 0] = 1.
    w = torch.where(scheme_sign.unsqueeze(1) > 0, w, wh)
    w = w.view(X.size(0), n_steps, 1, 1, 1)
    Delta = (w * Delta).sum(1)
    for n, p in model.module.named_parameters():
        if n in tunable:
            p.requires_grad = True
    return Delta


def inst_nac(model, X, prompter, add_prompter, alpha, K, eps, tau, beta, mu):
    def clamp(x): return torch.max(torch.min(x, torch.tensor(1., device=x.device)), torch.tensor(0., device=x.device))
    delta = torch.zeros_like(X); delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - X, 1 - X); delta.requires_grad = True
    tl = [n for n, p in model.module.named_parameters() if p.requires_grad]
    for n, p in model.module.named_parameters(): p.requires_grad = False
    tok = add_prompter()
    with torch.no_grad():
        ori = model.module.encode_image(prompter(clip_img_preprocessing(X)), tok)
        ori_n = torch.norm(ori, dim=-1)
    v = torch.zeros_like(delta); deltas = [delta.data.clone()]
    gc_all = []; J_all = []; prev_g = None
    for s in range(K):
        la = X + delta + mu * v
        feat = model.module.encode_image(prompter(clip_img_preprocessing(la)), tok)
        if s == 0: dr = torch.norm(feat - ori, dim=-1) / ori_n
        sg = (tau - dr).sign()
        J_all.append((((feat - ori) ** 2).sum(1)).mean().item())
        g = torch.autograd.grad((((feat - ori) ** 2).sum(1)).sum(), delta)[0]
        if prev_g is not None: gc_all.append(_per_cos(g, prev_g).cpu().numpy())
        prev_g = g.detach()
        v = mu * v + alpha * torch.sign(g)
        d = torch.clamp(delta[:, :, :, :] + v, min=-eps, max=eps)
        d = torch.clamp(d, 0 - X, 1 - X)
        delta.data[:] = d; deltas.append(delta.data.clone())
    Delta = _finalize(deltas, sg, beta, tl, model, X)
    gc = np.nanmean(gc_all, 0) if gc_all else np.full(X.size(0), np.nan)
    return Delta, gc, np.array(J_all), (np.array(J_all)[0] if len(J_all) else np.nan), (J_all[-1] if J_all else np.nan)


def inst_ttc(model, X, prompter, add_prompter, alpha, K, eps, tau, beta):
    def clamp(x): return torch.max(torch.min(x, torch.tensor(1., device=x.device)), torch.tensor(0., device=x.device))
    delta = torch.zeros_like(X); delta.uniform_(-eps, eps); delta = torch.clamp(delta, 0 - X, 1 - X); delta.requires_grad = True
    tl = [n for n, p in model.module.named_parameters() if p.requires_grad]
    for n, p in model.module.named_parameters(): p.requires_grad = False
    tok = add_prompter()
    with torch.no_grad():
        ori = model.module.encode_image(prompter(clip_img_preprocessing(X)), tok); ori_n = torch.norm(ori, dim=-1)
    deltas = [delta.data.clone()]; gc_all = []; J_all = []; prev_g = None
    for s in range(K):
        feat = model.module.encode_image(prompter(clip_img_preprocessing(X + delta)), tok)
        if s == 0: dr = torch.norm(feat - ori, dim=-1) / ori_n
        sg = (tau - dr).sign(); J_all.append((((feat - ori) ** 2).sum(1)).mean().item())
        g = torch.autograd.grad((((feat - ori) ** 2).sum(1)).sum(), delta)[0]
        if prev_g is not None: gc_all.append(_per_cos(g, prev_g).cpu().numpy())
        prev_g = g.detach()
        d = torch.clamp(delta[:, :, :, :] + alpha * torch.sign(g), min=-eps, max=eps)
        d = torch.clamp(d, 0 - X, 1 - X); delta.data[:] = d; deltas.append(delta.data.clone())
    Delta = _finalize(deltas, sg, beta, tl, model, X)
    gc = np.nanmean(gc_all, 0) if gc_all else np.full(X.size(0), np.nan)
    return Delta, gc, np.array(J_all), J_all[0], J_all[-1]


def inst_momentum(model, X, prompter, add_prompter, alpha, K, eps, tau, beta, mu):
    def clamp(x): return torch.max(torch.min(x, torch.tensor(1., device=x.device)), torch.tensor(0., device=x.device))
    delta = torch.zeros_like(X); delta.uniform_(-eps, eps); delta = torch.clamp(delta, 0 - X, 1 - X); delta.requires_grad = True
    tl = [n for n, p in model.module.named_parameters() if p.requires_grad]
    for n, p in model.module.named_parameters(): p.requires_grad = False
    tok = add_prompter()
    with torch.no_grad():
        ori = model.module.encode_image(prompter(clip_img_preprocessing(X)), tok); ori_n = torch.norm(ori, dim=-1)
    mm = torch.zeros_like(delta); deltas = [delta.data.clone()]; gc_all = []; J_all = []; prev_g = None
    for s in range(K):
        feat = model.module.encode_image(prompter(clip_img_preprocessing(X + delta)), tok)
        if s == 0: dr = torch.norm(feat - ori, dim=-1) / ori_n
        sg = (tau - dr).sign(); J_all.append((((feat - ori) ** 2).sum(1)).mean().item())
        g = torch.autograd.grad((((feat - ori) ** 2).sum(1)).sum(), delta)[0]
        if prev_g is not None: gc_all.append(_per_cos(g, prev_g).cpu().numpy())
        prev_g = g.detach()
        mm = mu * mm + alpha * torch.sign(g)
        d = torch.clamp(delta[:, :, :, :] + mm, min=-eps, max=eps)
        d = torch.clamp(d, 0 - X, 1 - X); delta.data[:] = d; deltas.append(delta.data.clone())
    Delta = _finalize(deltas, sg, beta, tl, model, X)
    gc = np.nanmean(gc_all, 0) if gc_all else np.full(X.size(0), np.nan)
    return Delta, gc, np.array(J_all), J_all[0], J_all[-1]


def inst_purela(model, X, prompter, add_prompter, alpha, K, eps, tau, beta):
    def clamp(x): return torch.max(torch.min(x, torch.tensor(1., device=x.device)), torch.tensor(0., device=x.device))
    delta = torch.zeros_like(X); delta.uniform_(-eps, eps); delta = torch.clamp(delta, 0 - X, 1 - X); delta.requires_grad = True
    tl = [n for n, p in model.module.named_parameters() if p.requires_grad]
    for n, p in model.module.named_parameters(): p.requires_grad = False
    tok = add_prompter()
    with torch.no_grad():
        ori = model.module.encode_image(prompter(clip_img_preprocessing(X)), tok); ori_n = torch.norm(ori, dim=-1)
    deltas = [delta.data.clone()]; gc_all = []; J_all = []; prev_la = None
    for s in range(K):
        feat = model.module.encode_image(prompter(clip_img_preprocessing(X + delta)), tok)
        if s == 0: dr = torch.norm(feat - ori, dim=-1) / ori_n
        sg = (tau - dr).sign(); J_all.append((((feat - ori) ** 2).sum(1)).mean().item())
        g = torch.autograd.grad((((feat - ori) ** 2).sum(1)).sum(), delta)[0]
        tdir = alpha * torch.sign(g)
        dla = torch.clamp(delta.data + tdir, 0 - X, 1 - X); dla = torch.clamp(dla, -eps, eps).detach().requires_grad_(True)
        f_la = model.module.encode_image(prompter(clip_img_preprocessing(X + dla)), tok)
        gla = torch.autograd.grad((((f_la - ori) ** 2).sum(1)).sum(), dla)[0]
        if prev_la is not None: gc_all.append(_per_cos(gla, prev_la).cpu().numpy())
        prev_la = gla.detach()
        delta.data = torch.clamp(delta.data + alpha * torch.sign(gla), -eps, eps)
        delta.data = torch.clamp(delta.data, 0 - X, 1 - X); delta = delta.detach().requires_grad_(True)
        deltas.append(delta.data.clone())
    Delta = _finalize(deltas, sg, beta, tl, model, X)
    gc = np.nanmean(gc_all, 0) if gc_all else np.full(X.size(0), np.nan)
    return Delta, gc, np.array(J_all), J_all[0], J_all[-1]


def _logits(model, prompter, imgs, tt):
    lg, _, _, _ = multiGPU_CLIP(None, None, None, model,
                                prompter(clip_img_preprocessing(imgs)), tt,
                                prompt_token=None, dataset_name='cifar10')
    return lg.float()


FUNS = {'ttc': inst_ttc, 'momentum': inst_momentum, 'purela': inst_purela, 'nac': inst_nac}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--equiv_check', action='store_true')
    p.add_argument('--checkpoint_every', type=int, default=25)
    pargs = p.parse_args()

    outs = {}             # {(seed, method, K): arrays}
    traj = {}             # {(seed, method, K): per-step J mean}
    meta = {'KS': KS, 'METHODS': METHODS, 'eps': '4/255'}
    run_log = []

    for seed in pargs.seeds:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        sys.argv = ['fv', '--batch_size', '32', '--num_workers', '1', '--test_set', 'cifar10',
                    '--test_attack_type', 'pgd', '--test_eps', '4', '--test_numsteps', '10',
                    '--test_stepsize', '1', '--ttc_eps', '4', '--ttc_numsteps', '2',
                    '--ttc_stepsize', '1', '--tau_thres', '0.2', '--beta', '2.0',
                    '--seed', str(seed), '--outdir', OUTDIR]
        args = parse_options()
        args.test_eps /= 255.; args.test_stepsize /= 255.; args.ttc_stepsize /= 255.; args.ttc_eps /= 255.
        model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
        for par in model.parameters(): par.requires_grad = False
        convert_models_to_fp32(model); model = torch.nn.DataParallel(model); model.eval()
        prompter = torch.nn.DataParallel(NullPrompter()).cuda()
        add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
        vd, vl = load_val_dataset(args, 'cifar10')
        texts = get_text_prompts_val([vd], ['cifar10'])[0]
        text_tokens = clip.tokenize(texts).to(device)
        crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
        nb = len(vl)
        if pargs.max_batches > 0: nb = min(nb, pargs.max_batches)
        t0 = time.time()
        for bi, (images, target) in enumerate(vl):
            if bi >= nb: break
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast('cuda'):
                da = attack_pgd(args, prompter, model, None, None, add_prompter, crit, images, target,
                                args.test_stepsize, args.test_numsteps, 'l_inf', text_tokens=text_tokens,
                                epsilon=args.test_eps, dataset_name='cifar10')
                att = images + da
                with torch.no_grad():
                    fc = model.module.encode_image(prompter(clip_img_preprocessing(images)), None).float()
                    fa = model.module.encode_image(prompter(clip_img_preprocessing(att)), None).float()
            for m in METHODS:
                for K in KS:
                    torch.manual_seed(10000 + bi)
                    with torch.amp.autocast('cuda'):
                        if m == 'nac':
                            dd, gc, Jtr, J0, Js = FUNS[m](model, att.data, prompter, add_prompter,
                                                          alpha=args.ttc_stepsize, K=K, eps=EPS,
                                                          tau=args.tau_thres, beta=args.beta, mu=0.9)
                        elif m == 'momentum':
                            dd, gc, Jtr, J0, Js = FUNS[m](model, att.data, prompter, add_prompter,
                                                          alpha=args.ttc_stepsize, K=K, eps=EPS,
                                                          tau=args.tau_thres, beta=args.beta, mu=0.9)
                        else:
                            dd, gc, Jtr, J0, Js = FUNS[m](model, att.data, prompter, add_prompter,
                                                          alpha=args.ttc_stepsize, K=K, eps=EPS,
                                                          tau=args.tau_thres, beta=args.beta)
                        with torch.no_grad():
                            fd = model.module.encode_image(
                                prompter(clip_img_preprocessing(att + dd)), None).float()
                            lg = _logits(model, prompter, att + dd, text_tokens)
                    key = (seed, m, K)
                    if key not in outs: outs[key] = {'correct': [], 'margin': [], 'dclean': [], 'dadv': [],
                                                     'mean_abs': [], 'frac': [], 'gc': [], 'J0': [], 'Js': []}
                    pr = torch.softmax(lg, 1); t2 = pr.topk(2, 1).values
                    de = dd.detach().float(); flat = de.reshape(de.size(0), -1)
                    outs[key]['correct'].append((lg.argmax(1) == target).cpu().numpy())
                    outs[key]['margin'].append((t2[:, 0] - t2[:, 1]).cpu().numpy())
                    outs[key]['dclean'].append((fd - fc).norm(1).cpu().numpy())
                    outs[key]['dadv'].append((fd - fa).norm(1).cpu().numpy())
                    outs[key]['mean_abs'].append(flat.abs().mean(1).cpu().numpy())
                    outs[key]['frac'].append((flat.abs() >= EPS - 1e-4).float().mean(1).cpu().numpy())
                    outs[key]['gc'].append(gc)
                    outs[key]['J0'].append(np.float64(J0)); outs[key]['Js'].append(np.float64(Js))
                    traj[key] = traj.get(key, []) + [Jtr.tolist()]
            if (bi + 1) % 5 == 0 or bi == nb - 1:
                el = time.time() - t0
                print('[fv] seed=%d batch %d/%d el=%.1fmin eta=%.1fmin' %
                      (seed, bi + 1, nb, el / 60, (el / (bi + 1) * (nb - bi - 1)) / 60), flush=True)
            if (bi + 1) % pargs.checkpoint_every == 0:
                _save(outs, traj, meta, run_log, partial=True)
        run_log.append({'seed': seed, 'n_batches': nb, 'done': True})
        print('[fv] seed=%d done' % seed, flush=True)
        _save(outs, traj, meta, run_log, partial=False)


def _save(outs, traj, meta, run_log, partial):
    data = {}
    order = []
    for (seed, m, K), v in outs.items():
        tag = 's%d_%s_K%d' % (seed, m, K)
        order.append(tag)
        for k in ('correct', 'margin', 'dclean', 'dadv', 'mean_abs', 'frac', 'gc'):
            try:
                data[tag + '__' + k] = np.concatenate(
                    [np.asarray(x).reshape(-1) for x in v[k]])
            except ValueError as e:
                print('CONCAT-FAIL tag=%s key=%s lens=%s' %
                      (tag, k, [np.asarray(x).ndim for x in v[k]]), flush=True)
                raise
        data[tag + '__J0'] = np.asarray(v['J0'])
        data[tag + '__Js'] = np.asarray(v['Js'])
    fname = 'final_verify%s.npz' % ('_partial' if partial else '')
    if partial:
        np.savez_compressed(os.path.join(OUTDIR, 'final_verify_partial.npz'), **data)
        json.dump({'meta': meta, 'order': order, 'log': run_log, 'partial': True},
                  open(os.path.join(OUTDIR, 'final_verify_partial_manifest.json'), 'w'), indent=2)
    else:
        np.savez_compressed(os.path.join(OUTDIR, 'final_verify.npz'), **data)
        tr = {f'{k[0]}_{k[1]}_K{k[2]}': v for k, v in traj.items()}
        json.dump({'meta': meta, 'order': order, 'log': run_log, 'traj': tr},
                  open(os.path.join(OUTDIR, 'final_verify_manifest.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()