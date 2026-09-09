import sys, os, time, csv
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
import wandb
EXP_ROOT=Path(__file__).resolve().parents[1]; SRESNET_ROOT=EXP_ROOT.parents[1]
sys.path.insert(0,str(SRESNET_ROOT)); sys.path.insert(0,str(EXP_ROOT/"common"))
from src.model_tdbn_fpp import SResNestFPP
from src.reproducibility import seed_from_environment
from mad_fall_dataset import MADFallDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
SEED=seed_from_environment()
T=int(os.environ.get("T","50")); SEQ_LEN=5
BS=int(os.environ.get("BS","3")); EPOCHS=int(os.environ.get("EPOCHS","40"))
SPP=int(os.environ.get("SPP","40")); SPP_VAL=int(os.environ.get("SPP_VAL","30"))
P_FALL=float(os.environ.get("P_FALL","0.5")); FALL_BOOST=float(os.environ.get("FALL_BOOST","2.0"))
SEL_K=int(os.environ.get("SEL_K","3")); KS=(1,2,3,4,5)
LR,WD=1e-4,0.01; ACT=(1,2,3,4,5,6,7,8,9); FALL=9
TRAIN_P=list(range(1,71)); VAL_P=list(range(71,86))
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT=EXP_ROOT/"06_fall_detect"; OUT.mkdir(parents=True,exist_ok=True)
TAG=f"T{T}_fb{FALL_BOOST}_kcurve"
LAST=OUT/f"last_{TAG}.pth"; BEST=OUT/f"best_{TAG}.pth"; CSVP=OUT/f"metrics_{TAG}.csv"
wandb.init(project="Article5-MAD-Fall", name=f"fallKC_T{T}_bs{BS}_ep{EPOCHS}_fb{FALL_BOOST}",
           config=dict(T=T,bs=BS,epochs=EPOCHS,p_fall=P_FALL,fall_boost=FALL_BOOST,
                       sel_k=SEL_K,seed=SEED))
tr=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=TRAIN_P,activities=ACT,fall_activity=FALL,
   seq_len=SEQ_LEN,T=T,samples_per_participant=SPP,p_fall=P_FALL,seed=0,split_name="train")
va=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=VAL_P,activities=ACT,fall_activity=FALL,
   seq_len=SEQ_LEN,T=T,samples_per_participant=SPP_VAL,p_fall=P_FALL,seed=1,split_name="val")
trl=DataLoader(tr,batch_size=BS,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
val=DataLoader(va,batch_size=BS,shuffle=False,num_workers=4,pin_memory=True)
K=len(ACT); fi=tr.fall_idx
w=torch.ones(K,device=dev); w[fi]=FALL_BOOST
model=SResNestFPP(num_steps=T,num_classes=K).to(dev)
crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
print(f"Train {len(tr)} | Val {len(va)} | fall_idx={fi} | sel_K={SEL_K}",flush=True)
if not CSVP.exists():
    with open(CSVP,"w",newline="") as f:
        c=["epoch","train_loss","frameacc","loc"]
        for k in KS: c+=[f"recall_k{k}",f"prec_k{k}",f"f1_k{k}",f"far_k{k}"]
        csv.writer(f).writerow(c)
def evaluate():
    model.eval(); cn=[]; tt=[]; fc=ft=ftp=ftt=0
    with torch.no_grad():
        for x,y in val:
            functional.reset_net(model)
            x=x.to(dev,dtype=torch.float32); y=y.to(dev,dtype=torch.long)
            p=model(x).argmax(-1)
            fc+=(p==y).sum().item(); ft+=y.numel()
            fm=(y==fi); ftp+=((p==fi)&fm).sum().item(); ftt+=fm.sum().item()
            cn.append((p==fi).sum(1).cpu()); tt.append((y==fi).any(1).cpu())
    functional.reset_net(model)
    cnt=torch.cat(cn); tru=torch.cat(tt); r={"facc":fc/max(ft,1),"loc":ftp/max(ftt,1)}
    for k in KS:
        pf=cnt>=k; TP=(tru&pf).sum().item(); FP=((~tru)&pf).sum().item()
        FN=(tru&~pf).sum().item(); TN=((~tru)&~pf).sum().item()
        rec=TP/max(TP+FN,1); pre=TP/max(TP+FP,1); f1=2*pre*rec/max(pre+rec,1e-8); far=FP/max(FP+TN,1)
        r[f"recall_k{k}"]=rec; r[f"prec_k{k}"]=pre; r[f"f1_k{k}"]=f1; r[f"far_k{k}"]=far
    return r
st=time.time(); print("start",flush=True); best=0.0
for ep in range(1,EPOCHS+1):
    t0=time.time(); model.train(); tl=0.0; sn=0
    for bi,(x,y) in enumerate(trl):
        functional.reset_net(model)
        x=x.to(dev,dtype=torch.float32); y=y.to(dev,dtype=torch.long)
        lo=model(x); loss=crit(lo.reshape(-1,K),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); sn+=y.size(0)
        if bi%20==0: print(f"Epoch {ep}/{EPOCHS} | Batch {bi}/{len(trl)} | Loss {loss.item():.4f}",flush=True)
    functional.reset_net(model); m=evaluate(); trls=tl/sn
    dt=(time.time()-t0)/60; el=(time.time()-st)/3600
    print(f"Epoch : {ep} | TrainLoss {trls:.4f} | FrameAcc {m['facc']:.4f} | "
          f"k2 R{m['recall_k2']:.3f} F1{m['f1_k2']:.3f} FA{m['far_k2']:.3f} | "
          f"k3 R{m['recall_k3']:.3f} F1{m['f1_k3']:.3f} FA{m['far_k3']:.3f} | {dt:.1f}min {el:.2f}h",flush=True)
    with open(CSVP,"a",newline="") as f:
        row=[ep,trls,m['facc'],m['loc']]
        for k in KS: row+=[m[f"recall_k{k}"],m[f"prec_k{k}"],m[f"f1_k{k}"],m[f"far_k{k}"]]
        csv.writer(f).writerow(row)
    wandb.log(dict(epoch=ep,train_loss=trls,**{f"val/{kk}":vv for kk,vv in m.items()}),step=ep)
    torch.save({"epoch":ep,"model":model.state_dict(),"opt":opt.state_dict(),"best":best},LAST)
    if m[f"f1_k{SEL_K}"]>best:
        best=m[f"f1_k{SEL_K}"]; torch.save(model.state_dict(),BEST)
        print(f"best saved F1@K{SEL_K}={best:.4f}",flush=True)
print("training finished",flush=True); print(f"best F1@K{SEL_K}: {best:.4f}",flush=True); wandb.finish()
