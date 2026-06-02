"""
NAC: Nesterov Accelerated Counterattack (Ours).

Same optimization problem as TTC, but uses Nesterov accelerated gradient
instead of standard PGD. Only two lines differ from TTC:
  1. Gradient computed at look-ahead position (x + delta + mu * velocity)
  2. Nesterov momentum update (velocity = mu * velocity + alpha * sign(grad))

Theoretical convergence: O(1/k^2) vs PGD's O(1/k).
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_tau(clip_visual, images, noise):
    """Compute feature difference ratio for tau-threshold gating."""
    orig_feat = clip_visual(images, None)
    noisy_feat = clip_visual(images + noise, None)
    return (noisy_feat - orig_feat).norm(dim=-1) / orig_feat.norm(dim=-1)


def nac_counterattack(model, X, prompter, add_prompter, alpha, attack_iters,
                       norm="l_inf", epsilon=0, tau_thres=None, beta=None,
                       clip_visual=None, clip_preprocess=None,
                       nac_momentum=0.9):
    """
    NAC: Nesterov Accelerated Counterattack (our method).

    Key difference from TTC:
      TTC: grad = ∇L(x + delta), delta = delta + alpha * sign(grad)
      NAC: grad = ∇L(x + delta + mu * velocity)
           velocity = mu * velocity + alpha * sign(grad)
           delta = delta + velocity

    Args:
        model: CLIP model (DataParallel wrapped)
        X: input images [B, 3, H, W] (adversarial)
        prompter: visual prompter
        add_prompter: additional token prompter
        alpha: step size
        attack_iters: number of steps
        norm: 'l_inf' or 'l_2'
        epsilon: perturbation budget
        tau_thres: threshold for step weighting
        beta: weighting coefficient
        clip_visual: original visual encoder (optional)
        clip_preprocess: image preprocessing function
        nac_momentum: Nesterov momentum coefficient (default: 0.9)

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

    # === NAC CHANGE 1: Initialize Nesterov velocity ===
    velocity = torch.zeros_like(delta)

    for step_id in range(attack_iters):
        # === NAC CHANGE 2: Compute gradient at look-ahead position ===
        # TTC: X + delta
        # NAC: X + delta + mu * velocity
        look_ahead = X + delta + nac_momentum * velocity
        X_input = clip_preprocess(look_ahead) if clip_preprocess else look_ahead
        prompted_images = prompter(X_input)
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        if step_id == 0 and diff_ratio is None:
            feature_diff = X_att_reps - X_ori_reps
            diff_ratio = torch.norm(feature_diff, dim=-1) / X_ori_norm

        scheme_sign = (tau_thres - diff_ratio).sign()

        l2_loss = (((X_att_reps - X_ori_reps) ** 2).sum(1)).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]

        # === NAC CHANGE 3: Nesterov momentum update ===
        # TTC: d = d + alpha * sign(g)
        # NAC: v = mu * v + alpha * sign(g), d = d + v
        velocity = nac_momentum * velocity + alpha * torch.sign(grad)
        d = delta[:, :, :, :] + velocity
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d, min=-epsilon, max=epsilon)
        elif norm == "l_2":
            d = d.view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        deltas_per_step.append(delta.data.clone())

    # Weighted averaging across steps (same as TTC)
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
