"""
Geometry mechanism pilot: extract CLIP feature vectors for geometry analysis.

Existing data only stored scalars; this run recovers the actual embedding
vectors needed for angular/directional analysis:
  f_clean (10000,512), f_adv, f_def(=NAC K=4 counterattacked),
  labels, nac_correct, and the 10 class text-prototypes.

Protocol identical to prior runs: CIFAR-10 test (10000), seed 0, ViT-B/32,
PGD-10 eps 4/255 step 1/255, NAC counterattack K=4 (weights/tau/beta default).
Deterministic -> directly comparable to final_verify.npz NAC@K4 = 37.51.
Output: results/adaptive_pilot/geometry.npz (+ accuracy sanity in log)
"""
from __future__ import print_function
import os, sys, time, json
import numpy as np, torch
_NAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NAC_ROOT)
from nac_fair_experiment import parse_options, nac_counterattack
from func import device, clip_img_preprocessing
from replace import clip
from utils import load_val_dataset, get_text_prompts_val, convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter, NullPrompter

OUTDIR = os.path.join(_NAC_ROOT, 'results', 'adaptive_pilot')
K = 4
EPS = 4. / 255.

def main():
    sys.argv = ['geo', '--batch_size', '32', '--num_workers', '2', '--test_set', 'cifar10',
                '--test_attack_type', 'pgd', '--test_eps', '4', '--test_numsteps', '10',
                '--test_stepsize', '1', '--ttc_eps', '4', '--ttc_numsteps', '2',
                '--ttc_stepsize', '1', '--tau_thres', '0.2', '--beta', '2.0',
                '--seed', '0', '--outdir', OUTDIR]
    args = parse_options()
    args.test_eps /= 255.; args.test_stepsize /= 255.; args.ttc_stepsize /= 255.; args.ttc_eps /= 255.
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

    # text prototypes (class-conditional) via encode_text
    with torch.no_grad():
        protos = model.module.encode_text(text_tokens).float().cpu().numpy()  # [10,512]

    def feat(x):
        with torch.no_grad():
            return model.module.encode_image(
                prompter(clip_img_preprocessing(x)), None).float().cpu().numpy()

    fc_all, fa_all, fd_all, lab_all, ok_all = [], [], [], [], []
    t0 = time.time(); nb = len(vl)
    for bi, (images, target) in enumerate(vl):
        images, target = images.to(device), target.to(device)
        with torch.amp.autocast('cuda'):
            da = attack_pgd(args, prompter, model, None, None, add_prompter, crit, images, target,
                            args.test_stepsize, args.test_numsteps, 'l_inf', text_tokens=text_tokens,
                            epsilon=args.test_eps, dataset_name='cifar10')
            att = images + da
        fc_all.append(feat(images))
        fa_all.append(feat(att))
        torch.manual_seed(10000 + bi)
        with torch.amp.autocast('cuda'):
            dd = nac_counterattack(model, att.data, prompter, add_prompter, alpha=args.ttc_stepsize,
                                   attack_iters=K, norm='l_inf', epsilon=EPS,
                                   visual_model_orig=None, tau_thres=args.tau_thres,
                                   beta=args.beta, clip_visual=None, nac_momentum=0.9)
        fd_all.append(feat(att + dd))
        lab_all.append(target.cpu().numpy())
        # correctness: cosine-vs-text via prototype (skip; final uses encode_image+text)
        if (bi + 1) % 10 == 0 or bi == nb - 1:
            print('[geo] batch %d/%d el=%.1fmin eta=%.1fmin' %
                  (bi + 1, nb, (time.time() - t0) / 60,
                   ((time.time() - t0) / (bi + 1) * (nb - bi - 1)) / 60), flush=True)

    fc = np.concatenate(fc_all); fa = np.concatenate(fa_all); fd = np.concatenate(fd_all)
    lab = np.concatenate(lab_all)
    # correctness by cosine logits (image feat . text feat)
    proto = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    clean_logits = fc @ proto.T
    adv_logits = fa @ proto.T
    def_logits = fd @ proto.T
    clean_ok = (clean_logits.argmax(1) == lab)
    adv_ok = (adv_logits.argmax(1) == lab)
    nac_ok = (def_logits.argmax(1) == lab)
    print('[geo] clean=%.2f adv(no def)=%.2f NAC@K%d=%.2f' % (
        clean_ok.mean() * 100, adv_ok.mean() * 100, K, nac_ok.mean() * 100), flush=True)
    np.savez_compressed(os.path.join(OUTDIR, 'geometry.npz'),
                        f_clean=fc, f_adv=fa, f_def=fd, label=lab,
                        protos=protos, clean_ok=clean_ok, adv_ok=adv_ok, nac_ok=nac_ok)
    json.dump({'K': K, 'eps': '4/255', 'n': int(len(lab)),
               'clean_acc': float(clean_ok.mean() * 100),
               'adv_acc': float(adv_ok.mean() * 100),
               'nac_acc': float(nac_ok.mean() * 100)},
              open(os.path.join(OUTDIR, 'geometry_manifest.json'), 'w'), indent=2)
    print('[geo] ALL DONE')

if __name__ == '__main__':
    main()