"""
Clean-side FPT two-level probe τ, MIRRORING fpt_signal_extract.py EXACTLY.

Purpose: control experiment to check whether FPT-Noise's clean/adv discrimination
actually reproduces on our setup. We already have τ_adv (adversarial side) in
fpt_signals.npz. Here we compute τ_clean on the CLEAN image X using the IDENTICAL
per-sample noise draws (same Generator seeds 10000+bi*1000+j / 20000+bi*1000+j)
and the SAME RNG-stream replication of the counterattack init, so that:
  * clean and adv τ use pointwise-identical noise -> cleanest possible comparison;
  * the middle-loop RNG state matches final_verify exactly (alignment preserved).
Output: results/adaptive_pilot/fpt_clean.npz with fpt_tau_clean (10000,),
  f_clean (10000,512) (for alignment sanity vs geometry.f_clean),
  df0/df1/fnorm for the clean side.
Crash-safe: incremental save each batch.
"""
from __future__ import print_function
import os, sys, time
import numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
from nac_fair_experiment import parse_options
from func import device, clip_img_preprocessing
from replace import clip
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter, NullPrompter

OUTDIR = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')
GEO = np.load(os.path.join(OUTDIR, 'geometry.npz'))  # alignment reference (f_clean)
NBT_CAP = int(os.environ.get('FPT_NBT', 10 ** 9))
EPS0, EPS1 = 4. / 255., 32. / 255.
EPS_CA = 4. / 255.
PROG_EVERY = 25


def main():
    sys.argv = ['fptc', '--batch_size', '32', '--num_workers', '2', '--test_set', 'cifar10',
                '--test_attack_type', 'pgd', '--test_eps', '4', '--test_numsteps', '10',
                '--test_stepsize', '1', '--seed', '0', '--outdir', OUTDIR]
    args = parse_options()
    args.test_eps /= 255.; args.test_stepsize /= 255.
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=0)
    for p in model.parameters(): p.requires_grad = False
    convert_models_to_fp32(model); model = torch.nn.DataParallel(model); model.eval()
    prompter = torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter = torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd, vl = load_val_dataset(args, 'cifar10')
    texts = get_text_prompts_val([vd], ['cifar10'])[0]
    text_tokens = clip.tokenize(texts).to(device)
    crit = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    def feat(x):
        with torch.no_grad():
            return model.module.encode_image(
                prompter(clip_img_preprocessing(x)), None).float().cpu().numpy()

    tau_all, df0_all, df1_all, fnorm_all, clean_all = [], [], [], [], []
    t0 = time.time(); nb = len(vl); nbt = min(NBT_CAP, nb)
    for bi, (images, target) in enumerate(vl):
        if bi >= nbt: break
        images, target = images.to(device), target.to(device)
        with torch.amp.autocast('cuda'):
            da = attack_pgd(args, prompter, model, None, None, add_prompter, crit, images, target,
                            args.test_stepsize, args.test_numsteps, 'l_inf', text_tokens=text_tokens,
                            epsilon=args.test_eps, dataset_name='cifar10')
            adv = images + da
        fc = feat(images)                       # CLEAN feature (X)
        # ---- two-level probe on the CLEAN sample, same noise draws as adv side ----
        B = images.shape[0]
        d0s, d1s = [], []
        for j in range(B):
            g = torch.Generator().manual_seed(10000 + bi * 1000 + j)
            d0 = torch.randn(3, 224, 224, generator=g) * (EPS0 * 255.0)
            g = torch.Generator().manual_seed(20000 + bi * 1000 + j)
            d1 = torch.randn(3, 224, 224, generator=g) * (EPS1 * 255.0)
            d0 = torch.clamp(d0, -EPS0 * 255.0, EPS0 * 255.0) / 255.0
            d1 = torch.clamp(d1, -EPS1 * 255.0, EPS1 * 255.0) / 255.0
            d0s.append(d0.unsqueeze(0)); d1s.append(d1.unsqueeze(0))
        d0 = torch.cat(d0s).cuda().to(images.dtype)
        d1 = torch.cat(d1s).cuda().to(images.dtype)
        fd0 = feat((images + d0).clamp(-1, 1))
        fd1 = feat((images + d1).clamp(-1, 1))
        fnorm = np.linalg.norm(fc, axis=1)
        df0 = np.linalg.norm(fd0 - fc, axis=1)
        df1 = np.linalg.norm(fd1 - fc, axis=1)
        tau = (df1 - df0) / fnorm
        # RNG-stream replication (identical role as adv-side script; keeps state aligned)
        torch.manual_seed(10000 + bi)
        torch.zeros_like(adv).uniform_(-EPS_CA, EPS_CA)
        tau_all.append(tau); df0_all.append(df0); df1_all.append(df1)
        fnorm_all.append(fnorm); clean_all.append(fc)
        if (bi + 1) % PROG_EVERY == 0 or bi == nbt - 1:
            print('[fptc] batch %d/%d mean_tau=%.4f el=%.1fmin eta=%.1fmin' %
                  (bi + 1, nbt, np.mean(tau),
                   (time.time() - t0) / 60,
                   (time.time() - t0) / (bi + 1) * (nbt - bi - 1) / 60), flush=True)
        np.savez_compressed(os.path.join(OUTDIR, 'fpt_clean.npz'),
                            fpt_tau_clean=np.concatenate(tau_all),
                            df0=np.concatenate(df0_all), df1=np.concatenate(df1_all),
                            fnorm=np.concatenate(fnorm_all),
                            f_clean=np.concatenate(clean_all))

    tau = np.concatenate(tau_all); fc = np.concatenate(clean_all)
    ref = np.asarray(GEO['f_clean'])[:len(tau)]
    al = np.linalg.norm(fc - ref, axis=1)
    print('\n[align] f_clean vs geometry max_L2=%.4f mean_L2=%.4f  (expect ~1e-5)' %
          (al.max(), al.mean()), flush=True)
    print('[fptc] ALL DONE  n=%d  fnorm_clean=%.3f  df0=%.3f df1=%.3f fpt_tau_clean=%.4f' % (
        len(tau), fnorm.mean(), np.mean(np.concatenate(df0_all)),
        np.mean(np.concatenate(df1_all)), tau.mean()), flush=True)

if __name__ == '__main__':
    main()