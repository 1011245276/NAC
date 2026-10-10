"""
Active-probe signal: FPT-Noise two-level feature-perception-threshold.

FPT (FPT-Noise, IEEE TCSVT) distinguishes clean/adversarial by the DIFFERENCE in
latent drift under two random-noise levels, instead of a single level:
    tau_FPT = ( ||f(X+d1)-f(X)|| - ||f(X+d0)-f(X)|| ) / ||f(X)||
where d0 ~ small (eps=4/255), d1 ~ large (eps=32/255). Its claim: adversarial
samples show a distinct differential response vs clean, giving a stronger
clean/adv - or here, "need-to-switch-solver" - routing signal than the static
signals (all ~0.50 AUROC) already tested.

Purpose here is NOT to defend clean images but to test whether this ACTIVE
probe signal can route "switch away from NAC to Momentum" (y=1 iff NAC_wrong &
Momentum_ok) better than chance — the same target as routing_signal_test.py.

Protocol (identical to final_verify / geometry so index alignment is exact):
  ViT-B/32 x cifar10, seed0, batch 32, PGD-10 eps=4/255 step 1/255,
  X here = the ADVERSARIAL sample x_adv (= X+da). d0/d1 sampled from N(0,I)
  projected to L-inf balls of 4/255 and 32/255; per-sample fixed seed (10000+bi
   + noise idx) so the probe is deterministic. Features via encode_image.
Output: results/adaptive_pilot/fpt_signals.npz with
  fpt_tau (10000,) df0 (10000,) df1 (10000,) fnorm (10000,), f_adv (10000,512)
  f_adv written for alignment-sanity vs geometry.npz.
Crash-safe: saved after every batch with cumulative concat.
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
GEO = np.load(os.path.join(OUTDIR, 'geometry.npz'))  # alignment reference
NBT_CAP = int(os.environ.get('FPT_NBT', 10 ** 9))  # env-overridable for smoke; default ALL batches
EPS0, EPS1 = 4. / 255., 32. / 255.   # probe noise levels
EPS_CA = 4. / 255.                    # counterattack eps (for RNG-stream replication)
PROG_EVERY = 25


def main():
    sys.argv = ['fpt', '--batch_size', '32', '--num_workers', '2', '--test_set', 'cifar10',
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

    tau_all, df0_all, df1_all, fnorm_all, adv_all = [], [], [], [], []
    t0 = time.time(); nb = len(vl); nbt = min(NBT_CAP, nb)
    for bi, (images, target) in enumerate(vl):
        if bi >= nbt: break
        images, target = images.to(device), target.to(device)
        with torch.amp.autocast('cuda'):
            da = attack_pgd(args, prompter, model, None, None, add_prompter, crit, images, target,
                            args.test_stepsize, args.test_numsteps, 'l_inf', text_tokens=text_tokens,
                            epsilon=args.test_eps, dataset_name='cifar10')
            adv = images + da
        fa = feat(adv)
        # ---- active two-level probe on the ADVERSARIAL sample ----
        B = images.shape[0]
        d0s, d1s = [], []
        for j in range(B):
            g = torch.Generator().manual_seed(10000 + bi*1000 + j)
            d0 = torch.randn(3, 224, 224, generator=g) * (EPS0 * 255.0)
            g = torch.Generator().manual_seed(20000 + bi*1000 + j)
            d1 = torch.randn(3, 224, 224, generator=g) * (EPS1 * 255.0)
            # project to L-inf balls (keep sign*direction, clamp magnitude)
            d0 = torch.clamp(d0, -EPS0*255.0, EPS0*255.0) / 255.0
            d1 = torch.clamp(d1, -EPS1*255.0, EPS1*255.0) / 255.0
            d0s.append(d0.unsqueeze(0)); d1s.append(d1.unsqueeze(0))
        d0 = torch.cat(d0s).cuda().to(images.dtype)
        d1 = torch.cat(d1s).cuda().to(images.dtype)
        fd0 = feat((adv + d0).clamp(-1, 1))
        fd1 = feat((adv + d1).clamp(-1, 1))
        fnorm = np.linalg.norm(fa, axis=1)
        df0 = np.linalg.norm(fd0 - fa, axis=1)
        df1 = np.linalg.norm(fd1 - fa, axis=1)
        tau = (df1 - df0) / fnorm
        # ---- RNG-stream replication (CRITICAL for final_verify alignment) ----
        # final_mechanism_verify.py per batch: attack -> 20x[manual_seed(10000+bi);
        # counterattack]. Each counterattack's ONLY RNG op is
        #   delta = torch.zeros_like(X); delta.uniform_(-eps, eps)   (inst_nac L69-70)
        # so the state entering batch bi+1's attack == state after
        # [manual_seed(10000+bi) + one uniform_ of att's shape/dtype].
        # Replicate exactly that (bounds identical to inst_nac's eps=4/255):
        torch.manual_seed(10000 + bi)
        torch.zeros_like(adv).uniform_(-EPS_CA, EPS_CA)
        tau_all.append(tau); df0_all.append(df0); df1_all.append(df1)
        fnorm_all.append(fnorm); adv_all.append(fa)
        if (bi + 1) % PROG_EVERY == 0 or bi == nbt - 1:
            print('[fpt] batch %d/%d mean_tau=%.4f el=%.1fmin eta=%.1fmin' %
                  (bi + 1, nbt, np.mean(tau),
                   (time.time() - t0) / 60,
                   (time.time() - t0) / (bi + 1) * (nbt - bi - 1) / 60), flush=True)
        np.savez_compressed(os.path.join(OUTDIR, 'fpt_signals.npz'),
                            fpt_tau=np.concatenate(tau_all),
                            df0=np.concatenate(df0_all),
                            df1=np.concatenate(df1_all),
                            fnorm=np.concatenate(fnorm_all),
                            f_adv=np.concatenate(adv_all))

    tau = np.concatenate(tau_all); fnorm = np.concatenate(fnorm_all)
    fa = np.concatenate(adv_all)
    # ---- alignment sanity vs geometry.npz (should be ~0) ----
    ref = np.asarray(GEO['f_adv'])[:len(tau)]
    al = np.linalg.norm(fa - ref, axis=1)
    print('\n[align] f_adv vs geometry max_L2=%.4f mean_L2=%.4f  (expect ~1e-5 or below)' %
          (al.max(), al.mean()), flush=True)
    # ---- clean vs adv separation sanity (descriptive only, not routing) ----
    print('[fpt] ALL DONE  n=%d  fnorm_adv=%.3f  df0=%.3f df1=%.3f fpt_tau=%.4f' % (
        len(tau), fnorm.mean(), np.mean(np.concatenate(df0_all)),
        np.mean(np.concatenate(df1_all)), tau.mean()), flush=True)

if __name__ == '__main__':
    main()