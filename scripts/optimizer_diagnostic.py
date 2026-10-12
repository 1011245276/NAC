"""
Optimizer mechanism diagnostic: WHY does NAG outperform other optimizers?

Experiment A: Per-step gradient trajectory for TTC / NAC / Adam / RMSProp
  - loss (progress), cos_sim_true (alignment with ideal correction),
    cos_sim_prev (direction consistency)

Experiment B: Look-ahead vs current-position gradient quality (NAC only)
  - loss and gradient at BOTH delta_t and delta_t + mu*v_t
  - which position provides better alignment with true correction?

Usage:
  python optimizer_diagnostic.py                          # full run (~500 samples)
  python optimizer_diagnostic.py --smoke                  # 2-batch smoke test
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
os.chdir(_NAC_ROOT)

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import *
from attacks import attack_pgd
from func import clip_img_preprocessing
from nac_fair_experiment import parse_options

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTDIR = os.path.join(_NAC_ROOT, 'results', 'optimizer_diagnostic')
os.makedirs(OUTDIR, exist_ok=True)

EPS_DEF = 4. / 255.   # defense epsilon
EPS_ATK = 4. / 255.   # attack epsilon
ALPHA  = 1. / 255.    # step size
MU     = 0.9          # NAC momentum
K      = 2             # iteration count
TAU    = 0.2
BETA   = 2.0
DECAY  = 0.9           # RMSProp / Adam decay
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def _clamp(x, lo, hi):
    if not torch.is_tensor(lo):
        lo = x.new_tensor(lo)
    if not torch.is_tensor(hi):
        hi = x.new_tensor(hi)
    return torch.max(torch.min(x, hi), lo)


def _cos_sim_batch(a, b):
    """Per-sample cosine similarity, a/b shape [B, C, H, W]."""
    af = a.reshape(a.size(0), -1)
    bf = b.reshape(b.size(0), -1)
    return F.cosine_similarity(af, bf, dim=1)


def _diagnostic_counterattack(model, X_adv, prompter, add_prompter,
                               clip_visual, method, clean_dir,
                               n_steps=K):
    """
    Single counterattack run with per-step diagnostics.

    clean_dir: [B, C, H, W] tensor, the ideal correction direction
               (x_clean - x_adv), normalized per-sample for cos_sim.

    Returns dict of lists (per-step arrays).
    """
    B = X_adv.size(0)
    lower, upper = 0., 1.

    delta = torch.zeros_like(X_adv).uniform_(-EPS_DEF, EPS_DEF)
    delta = _clamp(delta, lower - X_adv, upper - X_adv).detach()
    delta.requires_grad = True

    prompt_token = add_prompter()
    with torch.no_grad():
        X_adv_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X_adv)), prompt_token)
        X_adv_norm = torch.norm(X_adv_reps, dim=-1)

    diff_ratio = None
    velocity = torch.zeros_like(delta)
    # Adam state
    m_adam = torch.zeros_like(delta)
    v_adam = torch.zeros_like(delta)
    # RMSProp state
    v_rms = torch.zeros_like(delta)

    diag = {f'step{t}': {} for t in range(n_steps)}

    for t in range(n_steps):
        # ---- determine position ----
        if method == 'nac':
            pos = X_adv + delta + MU * velocity   # look-ahead
        else:
            pos = X_adv + delta                    # current

        # ---- forward + gradient ----
        img = prompter(clip_img_preprocessing(pos))
        reps = model.module.encode_image(img, prompt_token)
        if t == 0 and diff_ratio is None:
            feat_diff = reps - X_adv_reps
            diff_ratio = torch.norm(feat_diff, dim=-1) / X_adv_norm
        scheme_sign = (TAU - diff_ratio).sign()
        l2_loss = (((reps - X_adv_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta, retain_graph=False)[0]

        # ---- diagnostics ----
        grad_flat = grad.reshape(B, -1)
        grad_norm = grad_flat.norm(dim=1, keepdim=True) + 1e-8
        grad_unit = grad_flat / grad_norm
        cd = clean_dir.reshape(B, -1)
        cd_norm = cd.norm(dim=1, keepdim=True) + 1e-8
        cd_unit = cd / cd_norm
        cos_true = (grad_unit * cd_unit).sum(dim=1)
        delta_norm = delta.reshape(B, -1).norm(dim=1)

        d = diag[f'step{t}']
        d['loss'] = l2_loss.item()
        d['cos_true'] = cos_true.detach().cpu().numpy()   # [B]
        d['grad_unit'] = grad_unit.detach().cpu().numpy()  # [B, D]
        d['delta_norm'] = delta_norm.detach().cpu().numpy()

        if t > 0:
            prev_unit = diag[f'step{t-1}']['grad_unit']
            cos_prev = (grad_unit.cpu() * torch.from_numpy(prev_unit)).sum(dim=1).numpy()
            d['cos_prev'] = cos_prev
        else:
            d['cos_prev'] = np.zeros(B)

        # ---- Experiment B: NAC extra gradient at current position ----
        if method == 'nac' and t < n_steps:
            # gradient at CURRENT position (what TTC would see)
            pos_cur = X_adv + delta
            img_cur = prompter(clip_img_preprocessing(pos_cur))
            reps_cur = model.module.encode_image(img_cur, prompt_token)
            l2_cur = (((reps_cur - X_adv_reps) ** 2).sum(1)).sum()
            grad_cur = torch.autograd.grad(l2_cur, delta, retain_graph=False)[0]
            grad_cur_flat = grad_cur.reshape(B, -1)
            gcn = grad_cur_flat.norm(dim=1, keepdim=True) + 1e-8
            grad_cur_unit = grad_cur_flat / gcn
            cos_true_cur = (grad_cur_unit * cd_unit).sum(dim=1)
            cos_la_vs_cur = (grad_unit * grad_cur_unit).sum(dim=1)
            d['loss_current'] = l2_cur.item()
            d['cos_true_current'] = cos_true_cur.detach().cpu().numpy()
            d['cos_la_vs_cur'] = cos_la_vs_cur.detach().cpu().numpy()

        # ---- update ----
        if method == 'ttc':
            delta.data = delta.data + ALPHA * torch.sign(grad)
        elif method == 'nac':
            velocity = MU * velocity + ALPHA * torch.sign(grad)
            delta.data = delta.data + velocity
        elif method == 'adam':
            m_adam = ADAM_B1 * m_adam + (1 - ADAM_B1) * grad
            v_adam = ADAM_B2 * v_adam + (1 - ADAM_B2) * grad ** 2
            m_hat = m_adam / (1 - ADAM_B1 ** (t + 1))
            v_hat = v_adam / (1 - ADAM_B2 ** (t + 1))
            delta.data = delta.data + ALPHA * m_hat / (v_hat.sqrt() + ADAM_EPS)
        elif method == 'rmsprop':
            v_rms = DECAY * v_rms + (1 - DECAY) * grad ** 2
            delta.data = delta.data + ALPHA * grad / (v_rms.sqrt() + ADAM_EPS)

        delta.data = torch.clamp(delta.data, -EPS_DEF, EPS_DEF)
        delta.data = _clamp(delta.data, lower - X_adv, upper - X_adv)

    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='2-batch smoke test')
    parser.add_argument('--n_batches', type=int, default=16,
                        help='number of batches (default 16 = 512 samples)')
    args = parser.parse_args()

    n_batches = 2 if args.smoke else args.n_batches

    # ---- setup ----
    sys.argv = ['diag', '--batch_size', '32', '--num_workers', '2',
                '--test_set', 'cifar10', '--test_attack_type', 'pgd',
                '--test_eps', '4', '--test_numsteps', '10',
                '--test_stepsize', '1', '--seed', '0']
    opt = parse_options()
    opt.test_eps /= 255.; opt.test_stepsize /= 255.
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters(): p.requires_grad = False
    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model); model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()

    vd, vl = load_val_dataset(opt, 'cifar10')
    texts = get_text_prompts_val([vd], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    methods = ['ttc', 'nac', 'adam', 'rmsprop']
    all_data = {m: [] for m in methods}

    print(f'[diag] K={K} eps_def={EPS_DEF:.4f} alpha={ALPHA:.4f} mu={MU}'
          f' n_batches={n_batches} bs=32', flush=True)
    t0 = time.time()

    for bi, (images, target) in enumerate(vl):
        if bi >= n_batches: break
        images, target = images.to(device), target.to(device)

        # PGD attack
        with torch.amp.autocast('cuda'):
            da = attack_pgd(opt, prompter, model, None, None, add_prompter,
                            crit, images, target, opt.test_stepsize,
                            opt.test_numsteps, 'l_inf',
                            text_tokens=text_tokens,
                            epsilon=opt.test_eps, dataset_name='cifar10')
        adv = images + da
        # clean direction (ideal correction)
        clean_dir = (images - adv).detach()  # [B, C, H, W]

        for method in methods:
            torch.manual_seed(10000 + bi)  # same delta init for fair comparison
            with torch.amp.autocast('cuda'):
                d = _diagnostic_counterattack(
                    model, adv, prompter, add_prompter,
                    clip_visual=None, method=method,
                    clean_dir=clean_dir, n_steps=K)
            all_data[method].append(d)

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            el = (time.time() - t0) / 60
            eta = el / (bi + 1) * (n_batches - bi - 1)
            print(f'[diag] batch {bi+1}/{n_batches} el={el:.1f}min eta={eta:.1f}min',
                  flush=True)

    # ---- aggregate ----
    def agg(method_list, key, step):
        return np.concatenate([d[f'step{step}'][key] for d in method_list])

    print('\n===== Experiment A: Gradient Trajectory Analysis =====', flush=True)
    print(f'{"Method":<10} {"Step":<5} {"Loss":>10} {"cos_true":>10} '
          f'{"cos_prev":>10} {"delta_norm":>12}', flush=True)
    for m in methods:
        for t in range(K):
            losses = [d[f'step{t}']['loss'] for d in all_data[m]]
            ct = agg(all_data[m], 'cos_true', t)
            cp = agg(all_data[m], 'cos_prev', t)
            dn = agg(all_data[m], 'delta_norm', t)
            print(f'{m:<10} {t:<5} {np.mean(losses):>10.1f} '
                  f'{np.mean(ct):>10.4f} {np.mean(cp):>10.4f} '
                  f'{np.mean(dn):>12.6f}', flush=True)

    print('\n===== Experiment B: Look-ahead vs Current (NAC only) =====',
          flush=True)
    print(f'{"Step":<5} {"Loss_LA":>10} {"Loss_cur":>10} '
          f'{"cosT_LA":>10} {"cosT_cur":>10} {"cos_LA_vs_cur":>14}',
          flush=True)
    for t in range(K):
        la = [d[f'step{t}']['loss'] for d in all_data['nac']]
        cu = [d[f'step{t}']['loss_current'] for d in all_data['nac']]
        ct_la = agg(all_data['nac'], 'cos_true', t)
        ct_cu = agg(all_data['nac'], 'cos_true_current', t)
        lc = agg(all_data['nac'], 'cos_la_vs_cur', t)
        print(f'{t:<5} {np.mean(la):>10.1f} {np.mean(cu):>10.1f} '
              f'{np.mean(ct_la):>10.4f} {np.mean(ct_cu):>10.4f} '
              f'{np.mean(lc):>14.4f}', flush=True)

    # ---- save npz ----
    save = {}
    for m in methods:
        for t in range(K):
            for key in ['cos_true', 'cos_prev', 'delta_norm']:
                save[f'{m}_step{t}_{key}'] = agg(all_data[m], key, t)
            save[f'{m}_step{t}_loss'] = np.array(
                [d[f'step{t}']['loss'] for d in all_data[m]])
        if m == 'nac':
            for t in range(K):
                for key in ['cos_true_current', 'cos_la_vs_cur']:
                    save[f'nac_step{t}_{key}'] = agg(all_data['nac'], key, t)
                save[f'nac_step{t}_loss_current'] = np.array(
                    [d[f'step{t}']['loss_current'] for d in all_data['nac']])

    tag = 'smoke' if args.smoke else 'full'
    outpath = os.path.join(OUTDIR, f'diag_{tag}.npz')
    np.savez_compressed(outpath, **save)
    print(f'\n[diag] saved to {outpath}', flush=True)


if __name__ == '__main__':
    main()
