"""
Semantic-anchor pilot: label-free prototype anchors from the ADVERSARIAL image
itself (no clean, no true label). Counterattack objective = attract f(X+d) to a
constructed semantic anchor. Compare with NAC (37.51).

Anchors (per-sample, from adversarial-image vs 10 class-text sim):
  A top1 proto
  B top3-uniform proto
  C softmax-weighted proto (temp=10)
  D temperature-scaled soft proto (temp=3)

Optimizer plain-PGD, K=4, eps=4/255, seed 0 (== objective_experts/NAC protocol).
Also report pseudo-label-correct vs wrong subsets.
Output: results/adaptive_pilot/semantic_anchor.npz
"""
from __future__ import print_function
import os, sys, time, json
import numpy as np, torch
_NAC_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,_NAC_ROOT)
from nac_fair_experiment import parse_options
from func import device, clip_img_preprocessing
from replace import clip
from utils import load_val_dataset,get_text_prompts_val,convert_models_to_fp32
from attacks import attack_pgd
from models.prompters import TokenPrompter,NullPrompter
OUTDIR=os.path.join(_NAC_ROOT,'results','adaptive_pilot'); K=4; EPS=4./255.; STEP=1./255.; TAU=0.2; BETA=2.0

def main():
    sys.argv=['sa','--batch_size','32','--num_workers','2','--test_set','cifar10',
              '--test_attack_type','pgd','--test_eps','4','--test_numsteps','10',
              '--test_stepsize','1','--ttc_eps','4','--ttc_numsteps','2','--ttc_stepsize','1',
              '--tau_thres','0.2','--beta','2.0','--seed','0','--outdir',OUTDIR]
    args=parse_options()
    for a in ('test_eps','test_stepsize','ttc_stepsize','ttc_eps'): setattr(args,a,getattr(args,a)/255.)
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    model,_=clip.load('ViT-B/32',device,jit=False,prompt_len=0)
    for p in model.parameters(): p.requires_grad=False
    convert_models_to_fp32(model); model=torch.nn.DataParallel(model); model.eval()
    prompter=torch.nn.DataParallel(NullPrompter()).cuda(); add_prompter=torch.nn.DataParallel(TokenPrompter(0)).cuda()
    vd,vl=load_val_dataset(args,'cifar10'); texts=get_text_prompts_val([vd],['cifar10'])[0]
    text_tokens=clip.tokenize(texts).to(device); crit=torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    with torch.no_grad(): protos=model.module.encode_text(text_tokens).float()
    pn=(protos/protos.norm(dim=1,keepdim=True))

    def enc_nograd(x):
        with torch.no_grad(): return model.module.encode_image(prompter(clip_img_preprocessing(x)),None).float()
    def enc_graph(x):
        return model.module.encode_image(prompter(clip_img_preprocessing(x)),None).float()

    def run_anchor(X, anchor, var):
        lower,upper=0.0,1.0
        d=torch.zeros_like(X); d.uniform_(-EPS,EPS); d=torch.clamp(d,lower-X,upper-X); d.requires_grad_()
        with torch.no_grad(): Fx=enc_nograd(X)
        deltas=[d.data.clone()]; diff=None
        for _ in range(K):
            feat=enc_graph(X+d)
            if diff is None: diff=torch.norm(feat-Fx,dim=-1)/(torch.norm(Fx,dim=-1)+1e-12)
            sg=(TAU-diff).sign()
            l2=(((feat-anchor)**2).sum(1)).sum(); g=torch.autograd.grad(l2,d)[0]
            d3=d-g.detach().sign()*STEP; d3=torch.clamp(d3,-EPS,EPS); d3=torch.clamp(d3,lower-X,upper-X)
            d.data[:]=d3; deltas.append(d.data.clone())
        ds=torch.stack(deltas,1)
        w=torch.arange(K+1).unsqueeze(0).expand(X.size(0),-1).to(device).float()
        w=torch.exp(sg.view(-1,1)*w*BETA); w/=w.sum(1,keepdim=True); wh=torch.zeros_like(w); wh[:,0]=1.
        w=torch.where(sg.unsqueeze(1)>0,w,wh); w=w.view(X.size(0),K+1,1,1,1)
        Delta=(w*ds).sum(1)
        return X+Delta

    ok={v:[] for v in 'ABCD'}; pseudo_ok={v:[] for v in 'ABCD'}  # per-sample (correct,v pseudo-ok)
    # pseudo-label arrays collected per batch
    top1_all=[]; top3_all=[]; lab_all=[]
    t0=time.time(); nb=len(vl)
    for bi,(images,target) in enumerate(vl):
        images,target=images.to(device),target.to(device)
        with torch.amp.autocast('cuda'):
            da=attack_pgd(args,prompter,model,None,None,add_prompter,crit,images,target,
                          args.test_stepsize,args.test_numsteps,'l_inf',text_tokens=text_tokens,
                          epsilon=args.test_eps,dataset_name='cifar10')
            att=images+da
        fa=enc_nograd(att); fa_n=fa/fa.norm(dim=1,keepdim=True)
        sim=fa_n@pn.T; T=torch.argsort(-sim,dim=1); top1=T[:,0].cpu().numpy(); top3=T[:,:3].cpu().numpy()
        lab_arr=target.cpu().numpy()
        top1_all.append(top1); top3_all.append(top3); lab_all.append(lab_arr)
        # compute softmax probs once for anchors
        pr10=torch.softmax(sim*10.0,1)
        anchors={}
        anchors['A']=pn[sim.argmax(1)]
        anchors['B']=(pn[top3[:,0]]+pn[top3[:,1]]+pn[top3[:,2]])
        anchors['B']=anchors['B']/anchors['B'].norm(dim=1,keepdim=True)
        anchors['C']=(pr10.unsqueeze(2)*pn.unsqueeze(0)).sum(1); anchors['C']=anchors['C']/anchors['C'].norm(dim=1,keepdim=True)
        pr3=torch.softmax(sim*3.0,1)
        anchors['D']=(pr3.unsqueeze(2)*pn.unsqueeze(0)).sum(1); anchors['D']=anchors['D']/anchors['D'].norm(dim=1,keepdim=True)
        for var in 'ABCD':
            anchor=anchors[var]
            torch.manual_seed(10000+bi)
            with torch.amp.autocast('cuda'):
                def_img=run_anchor(att.data.clone(),anchor,var)
                fdef=enc_nograd(def_img.detach())
            pred=(fdef@pn.T).argmax(1)
            ok[var].append(pred.eq(target).cpu().numpy())
            pseudo_ok[var].append((pred.eq(torch.from_numpy(top1).to(device))).cpu().numpy())
        if (bi+1)%10==0 or bi==nb-1:
            es=[round(float(np.concatenate(ok[x]).mean()*100),2) for x in 'ABCD']
            print('[sa] batch %d/%d el=%.1fmin A%s B%s C%s D%s'%(bi+1,nb,(time.time()-t0)/60,*es),flush=True)
        if (bi+1)%25==0 or bi==nb-1:
            _save(ok,top1_all,top3_all,lab_all)
    _save(ok,top1_all,top3_all,lab_all)
    print('[sa] final', {v:round(float(np.concatenate(ok[v]).mean()*100),2) for v in 'ABCD'},flush=True)

def _save(ok,top1_all,top3_all,lab_all):
    top1=np.concatenate(top1_all); top3=np.concatenate(top3_all); lab=np.concatenate(lab_all)
    data={f'anchor_{v}__correct':np.concatenate(ok[v]) for v in 'ABCD'}
    data['top1']=top1; data['top3']=top3; data['label']=lab
    np.savez_compressed(os.path.join(OUTDIR,'semantic_anchor.npz'),**data)
    acc=' '.join('%s=%.2f'%(v,np.concatenate(ok[v]).mean()*100) for v in 'ABCD')
    print('[sa save] acc: %s | pseudo-top1 acc=%.2f | true-in-top3=%.2f'%(
        acc,(top1==lab).mean()*100,(top3==lab[:,None]).any(1).mean()*100),flush=True)

if __name__=='__main__': main()