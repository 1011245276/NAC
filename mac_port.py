"""
Faithful, line-by-line port of the MAC multi-view guided counterattack into the
NAC experiment harness.

Every function in this module is copied verbatim from the official MAC
repository (external/MAC/code/mac.py, commit as cloned into external/MAC) so
that the ported version is behaviourally identical to the published method.
The ONLY changes are:

  1. import paths -- the harness lives in nac_project/, so clip_img_preprocessing
     is imported from the harness's func.py (verified numerically equivalent to
     external/MAC/code/func.py: same CLIP mean/std normalization).
  2. `lower_limit`, `upper_limit` and `clamp` are defined locally; in the
     official repo they are module-level globals in attacks.py and pulled in by
     `from attacks import *`. Values are identical (0, 1).

Nothing else is altered: same view construction, same self-anchor (encoding of
the un-augmented input), same corruption-degree soft weighting, same PGD inner
loop, same soft_weight applied to the final delta, same mean-logit aggregation.

Reference (official): external/MAC/code/mac.py lines 72-225.
"""
import torch
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    INTERP_BILINEAR = InterpolationMode.BILINEAR
except Exception:
    from PIL import Image
    INTERP_BILINEAR = Image.BILINEAR

from func import clip_img_preprocessing

# In external/MAC/code/attacks.py these are module-level globals.
lower_limit, upper_limit = 0, 1


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class AdditiveNoise(torch.nn.Module):
    def __init__(self, sigma_min=0.0, sigma_max=0.02):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x):
        sigma = torch.empty(1, device=x.device).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0, 1)


def build_aug(
    rotation=5.0,
    translate=0.04,
    scale_var=0.08,
    brightness=0.08,
    contrast=0.08,
    saturation=0.08,
    hue=0.02,
    blur_sigma=(0.5, 1.2),
    noise_sigma=0.02,
):
    affine = transforms.RandomAffine(
        degrees=rotation,
        translate=(translate, translate),
        scale=(1.0 - scale_var, 1.0 + scale_var),
        shear=0.0,
        interpolation=INTERP_BILINEAR
    )

    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )

    aug = transforms.Compose([
        transforms.RandomApply([affine], p=1.0),
        transforms.RandomApply([color_jitter], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=blur_sigma)], p=0.5),
        transforms.RandomApply([AdditiveNoise(0.0, noise_sigma)], p=0.5),
    ])
    return aug


def build_views(images, num_views, aug):
    if num_views <= 1 or aug is None:
        return images

    views = [images]

    with torch.no_grad():
        for _ in range(num_views - 1):
            imgs_aug = torch.stack(
                [aug(img) for img in images],
                dim=0
            )
            views.append(imgs_aug)

    return torch.cat(views, dim=0)


def aggregate_views(logits, num_views):
    if num_views <= 1:
        return logits
    V = num_views
    Bv, C = logits.shape
    assert Bv % V == 0, f"Logits batch {Bv} not divisible by num_views {V}"
    N = Bv // V
    return logits.view(V, N, C).mean(dim=0)


def multiview_guided_counterattack(model, X, prompter, add_prompter, alpha, attack_iters,
                                   norm="l_inf", epsilon=0, visual_model_orig=None,
                                   tau_thres: float = None, tau_temp: float = None,
                                   clip_visual=None, aug=None):
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

    # Freeze model parameters
    tunable_param_names = []
    for n, p in model.module.named_parameters():
        if p.requires_grad:
            tunable_param_names.append(n)
            p.requires_grad = False

    prompt_token = add_prompter()
    with torch.no_grad():
        X_ori_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X)), prompt_token
        )
        X_ori_norm = torch.norm(X_ori_reps, dim=-1)

    if aug is None:
        aug = build_aug()

    with torch.no_grad():
        X_base = X.detach()

        X_aug = torch.stack(
            [aug(img) for img in X_base],
            dim=0
        )

        X_aug_reps = model.module.encode_image(
            prompter(clip_img_preprocessing(X_aug)),
            prompt_token
        )
        corpt_degree = (X_aug_reps - X_ori_reps).norm(dim=-1) / (X_ori_reps.norm(dim=-1) + 1e-10)

    soft_weight = torch.sigmoid((corpt_degree - tau_thres) / tau_temp).view(-1, 1, 1, 1)

    for _step_id in range(attack_iters):

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        X_att_reps = model.module.encode_image(prompted_images, prompt_token)

        l2_loss = ((((X_att_reps - X_ori_reps) ** 2).sum(1))).sum()
        grad = torch.autograd.grad(l2_loss, delta)[0]
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d

    delta = delta * soft_weight

    # Unfreeze model parameters. Only for completeness of code
    for n, p in model.module.named_parameters():
        if n in tunable_param_names:
            p.requires_grad = True

    return delta.detach()