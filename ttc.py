"""
TTC: Test-Time Counterattack (CVPR 2025).

Original implementation by Xing et al.
https://github.com/Sxing2/CLIP-Test-time-Counterattacks

Only modification: extracted as standalone module for fair comparison.
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_tau(clip_visual, images, noise):
    """Compute feature difference ratio for tau-threshold gating."""
    orig_feat = clip_visual(images, None)
    noisy_feat = clip_visual(images + noise, None)
    return (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)


def ttc_counterattack(model, X, prompter, add_prompter, alpha, attack_iters,
                       norm="l_inf", epsilon=0, tau_thres=None, beta=None,
                       clip_visual=None, clip_preprocess=None):
    """
    TTC: Test-Time Counterattack (original implementation).

    Minimizes L2 distance between adversarial and clean image embeddings.

    Args:
        model: CLIP model (DataParallel wrapped)
        X: input images [B, 3, H, W] (adversarial)
        prompter: visual prompter (NullPrompter for standard)
        add_prompter: additional token prompter
        alpha: step size
        attack_iters: number of PGD steps
        norm: 'l_inf' or 'l_2'
        epsilon: perturbation budget
        tau_thres: threshold for step weighting (default: 0.2)
        beta: weighting coefficient (default: 2.0)
        clip_visual: original visual encoder for AFT models (optional)
        clip_preprocess: image preprocessing function

    Returns:
        Delta: counterattack perturbation [B, 3, H, W]
    """
    lower_limit, upper_limit = 0, 1

    def clamp(X, lo, hi):
        return torch.max(torch.min(X, hi), lo)

    delta = torch.zeros_like(X)
    if epsilon <= 0.:
        return delta

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    if attack_iters == 0:
        return delta.data

    diff_ratio = compute_tau(clip_visual, X, delta.data) if clip_visual is not None else None

    # Freeze model
    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_clean = clip_preprocess(X) if clip_preprocess else X
        X_ori_reps = model.module.encode_image(prompter(X_clean), prompt_token)
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    deltas_per_step = [delta.data.clone()]

    for step_id in range(attack_iters):
        X_input = clip_preprocess(X + delta) if clip_preprocess else (X + delta)
        prompted_images = prompter(X_input)
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_sign = (tau_thres - diff_ratio).sign()

        l2_loss = (((X_att_reps - X_ori_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    # Weighted averaging across steps
    Delta = torch.stack(deltas_per_step, dim=1)
    weights = torch.arange(attack_iters + 1).unsqueeze(0).expand(X.size(0), -1).to(device)
    weights = torch.exp(scheme_sign.view(-1, 1) * weights * beta)
    weights /= weights.sum(dim=1, keepdim=True)

    weights_hard = torch.zeros_like(weights)
    weights_hard[:, 0] = 1.
    weights = torch.where(scheme_sign.unsqueeze(1) > 0, weights, weights_hard)
    weights = weights.view(X.size(0), attack_iters + 1, 1, 1, 1)
    Delta = (weights * Delta).sum(dim=1)

    # Unfreeze
    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return Delta
