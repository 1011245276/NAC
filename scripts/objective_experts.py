"""
Objective-level MoE oracle: 4 CLIP-specific counterattack OBJECTIVES, SAME PGD
optimizer (objective is the ONLY difference). Per-sample success of each
objective + objective-level oracle - NAC.

Experts (applied to attacked image X, restored via delta):
  A repel-own      : MAX ||f(X+d)-f(X)||^2        (== current TTC/NAC objective)
  B clean-attract  : MIN ||f(X+d)-f(clean)||^2    (needs clean image; oracle-informative)
  C proto-attract  : MIN 1-cos(f(X+d), text[label]) (needs correct label; oracle-informative)
  D confidence     : MAX image-text logit margin of f(X+d) (label-free, CLIP-semantic)

Optimizer plain-PGD (sign), K=4, eps=4/255, same tau/beta weighting. seed 0.
Correctness = cosine-logit argmax (== geometry pipeline). NAC@K4 reference = 37.56.
"""
from __future__ import print_function
import os, sys, time, json
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
K = 4; EPS = 4./255.; STEP = 1./255.; BETA = 2.0; TAU = 0.2

def main():
    sys.argv = ['obj','--batch_size','32','--num_workers','2','--test_set','cifar10',
                '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
                '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2',
                '--ttc_stepsize','1','--tau_thres','0.2','--beta','2.0',
                '--seed','0','--outdir',OUTDIR]
    args = parse_options()
    args.test_eps/=255.; args.test_stepsize/=255.; args.ttc_stepsize/=255.; args.ttc_eps/=255.
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda()
    add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd,vl=load_val_dataset(args,'cifar10'); texts=get_text_prompts_val([vd],['cifar10'])[0]
    text_tokens=clip.tokenize(texts).to(device); crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    with torch.no_grad():
        protos=model.module.encode_text(text_tokens).float()
    pn=protos/protos.norm(dim=1,keepdim=True)

    def enc(x):
        with torch.no_grad():
            return model.module.encode_image(prompter(clip_img_preprocessing(x)),None).float()

    def run_obj(X, lab, clean_ref, variant):
        lower,upper=0.0,1.0
        d=torch.zeros_like(X); d.uniform_(-EPS,EPS); d=torch.clamp(d,lower-X,upper-X); d.requires_grad_()
        with torch.no_grad():
            Fx=enc(X)
        deltas=[d.data.clone()]; diff_ratio=None
        for _ in range(K):
            feat = enc_graph(X+d)   # grad-enabled forward through d
            if diff_ratio is None:
                diff_ratio=torch.norm(feat-Fx,dim=-1)/(torch.norm(Fx,dim=-1)+1e-12)
            sg=(TAU-diff_ratio).sign()
            if variant=='A':
                l2=(((feat-Fx)**2).sum(1)).sum(); g=torch.autograd.grad(l2,d)[0]
            elif variant=='B':
                l2=(((feat-clean_ref)**2).sum(1)).sum(); g=torch.autograd.grad(l2,d)[0]
            elif variant=='C':
                l2=-(feat*pn[lab]).sum(1).sum(); g=torch.autograd.grad(l2,d)[0]
            elif variant=='D':
                cls=feat@pn.T; sc=torch.softmax(cls*10.0,1); top2=sc.topk(2,1).values
                l2=-(top2[:,0]-top2[:,1]).sum(); g=torch.autograd.grad(l2,d)[0]
            gdir=g.detach().sign()
            d3 = d + gdir*STEP if variant=='A' else d - gdir*STEP
            d3=torch.clamp(d3,-EPS,EPS); d3=torch.clamp(d3,lower-X,upper-X)
            d.data[:]=d3; deltas.append(d.data.clone())
        dstack=torch.stack(deltas,1)
        w=torch.arange(K+1).unsqueeze(0).expand(X.size(0),-1).to(device).float()
        w=torch.exp(sg.view(-1,1)*w*BETA); w/=w.sum(1,keepdim=True); wh=torch.zeros_like(w); wh[:,0]=1.
        w=torch.where(sg.unsqueeze(1)>0,w,wh); w=w.view(X.size(0),K+1,1,1,1)
        Delta=(w*dstack).sum(1)
        return X+Delta

    def enc_graph(x):
        # forward WITH grad (dropout off; batchnorm/attention determinism ok under eval)
        return model.module.encode_image(prompter(clip_img_preprocessing(x)),None).float()

    ok={v:[] for v in 'ABCD'}
    t0=time.time(); nb=len(vl)
    for bi,(images,target) in enumerate(vl):
        images,target=images.to(device),target.to(device)
        with torch.amp.autocast('cuda'):
            da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                          args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                          epsilon=args.test_eps,dataset_name='cifar10')
            att=images+da
        clean_ref=enc(images)
        for v in 'ABCD':
            torch.manual_seed(10000+bi)
            with torch.amp.autocast('cuda'):
                def_img=run_obj(att.data.clone(),target,clean_ref,v)
                fdef=enc(def_img.detach())
            ok[v].append((fdef@pn.T).argmax(1).eq(target).cpu().numpy())
        if (bi+1)%10==0 or bi==nb-1:
            es=[round(float(np.concatenate(ok[x]).mean()*100),2) for x in 'ABCD']
            print('[objexp] batch %d/%d el=%.1fmin  A%s B%s C%s D%s'%
                  (bi+1,nb,(time.time()-t0)/60,*es),flush=True)
        if (bi+1)%25==0 or bi==nb-1:
            _save(ok)
    _save(ok)
    out={f'obj_{v}__correct':np.concatenate(ok[v]) for v in 'ABCD'}
    np.savez_compressed(os.path.join(OUTDIR,'objective_experts.npz'),**out)

def _save(ok):
    import os as _os
    data={f'obj_{v}__correct':np.concatenate(ok[v]) for v in 'ABCD'} if all(ok[v] for v in 'ABCD') else {}
    np.savez_compressed(_os.path.join(OUTDIR,'objective_experts_partial.npz'),**data)
    acc={v:float(np.concatenate(ok[v]).mean()*100) for v in 'ABCD'}
    json.dump({'K':K,'optimizer':'plain-PGD','eps':'4/255','objA_repel_own':acc['A'],
               'objB_clean_attract':acc['B'],'objC_proto_attract':acc['C'],
               'objD_confidence':acc['D']},
              open(os.path.join(OUTDIR,'objective_experts_manifest.json'),'w'),indent=2)
    print('[objexp] final acc', {v:round(a,2) for v,a in acc.items()},flush=True)

if __name__=='__main__':
    main()