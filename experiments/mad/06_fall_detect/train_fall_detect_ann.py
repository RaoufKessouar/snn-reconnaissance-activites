import sys, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
import wandb
EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT / "common"))
from src.model_ann_fpp import ANNResNetFPP
from mad_fall_dataset import MADFallDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
# ---------- config ----------
T=40; SEQ_LEN=5
BS=int(os.environ.get("BS","4")); EPOCHS=int(os.environ.get("EPOCHS","40"))
SPP=int(os.environ.get("SPP","40")); SPP_VAL=int(os.environ.get("SPP_VAL","30"))
P_FALL=float(os.environ.get("P_FALL","0.5")); FALL_BOOST=float(os.environ.get("FALL_BOOST","4.0"))
LR,WD=1e-4,0.01
ACTIVITIES=(1,2,3,4,5,6,7,8,9); FALL=9
TRAIN_P=list(range(1,71)); VAL_P=list(range(71,86))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT=EXP_ROOT/"06_fall_detect"; OUT.mkdir(parents=True,exist_ok=True)
LAST=OUT/"last_fall_ann.pth"; BEST=OUT/"best_fall_ann.pth"
run_name=f"fallANN_L{SEQ_LEN}_T{T}_bs{BS}_ep{EPOCHS}_pf{P_FALL}_fb{FALL_BOOST}"
wandb.init(project="Article5-MAD-Fall", name=run_name,
           config=dict(T=T,seq_len=SEQ_LEN,bs=BS,epochs=EPOCHS,lr=LR,wd=WD,p_fall=P_FALL,
                       fall_boost=FALL_BOOST,spp=SPP,model="SResNet38-tdBN-FramePerPos"))
train_set=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=TRAIN_P,activities=ACTIVITIES,
    fall_activity=FALL,seq_len=SEQ_LEN,T=T,samples_per_participant=SPP,p_fall=P_FALL,seed=0,split_name="anntrain")
val_set=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=VAL_P,activities=ACTIVITIES,
    fall_activity=FALL,seq_len=SEQ_LEN,T=T,samples_per_participant=SPP_VAL,p_fall=P_FALL,seed=1,split_name="annval")
train_loader=DataLoader(train_set,batch_size=BS,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
val_loader=DataLoader(val_set,batch_size=BS,shuffle=False,num_workers=4,pin_memory=True)
K=len(ACTIVITIES); fall_idx=train_set.fall_idx
counts=np.zeros(K)
for seq,_,fl in train_set.samples:
    for a,F in zip(seq,fl): counts[train_set.a2i[a]]+=F
weights=torch.ones(K,dtype=torch.float32,device=device); weights[fall_idx]=FALL_BOOST
print(f"Train seq {len(train_set)} | Val seq {len(val_set)} | fall_idx={fall_idx}",flush=True)
print(f"frame counts/class: {counts.astype(int).tolist()}",flush=True)
print(f"weights: {[round(w,2) for w in weights.tolist()]}",flush=True)
model=ANNResNetFPP(num_steps=T,num_classes=K).to(device)
criterion=nn.CrossEntropyLoss(weight=weights)
optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
start_epoch,best_f1=1,0.0
if os.environ.get("RESUME") and LAST.exists():
    ck=torch.load(LAST,map_location=device); model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["opt"]); start_epoch=ck["epoch"]+1; best_f1=ck["best"]
    print(f"RESUME epoch {ck['epoch']} best F1={best_f1:.4f}",flush=True)
def evaluate():
    model.eval(); TP=FP=FN=TN=0; ff_tp=ff_tot=0; fc=ft=0
    with torch.no_grad():
        for x,y in val_loader:
            functional.reset_net(model)
            x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
            preds=model(x).argmax(-1)                    # [B,T]
            fc+=(preds==y).sum().item(); ft+=y.numel()
            fm=(y==fall_idx); ff_tp+=((preds==fall_idx)&fm).sum().item(); ff_tot+=fm.sum().item()
            tf=(y==fall_idx).any(1); pf=(preds==fall_idx).any(1)
            TP+=(tf&pf).sum().item(); FP+=((~tf)&pf).sum().item()
            FN+=(tf&(~pf)).sum().item(); TN+=((~tf)&(~pf)).sum().item()
    functional.reset_net(model)
    rec=TP/max(TP+FN,1); pre=TP/max(TP+FP,1); f1=2*pre*rec/max(pre+rec,1e-8)
    return dict(recall=rec,precision=pre,f1=f1,far=FP/max(FP+TN,1),
                loc=ff_tp/max(ff_tot,1),facc=fc/max(ft,1))
start=time.time(); print("start training",flush=True)
for epoch in range(start_epoch,EPOCHS+1):
    t0=time.time(); model.train(); tl=0.0; seen=0
    for bi,(x,y) in enumerate(train_loader):
        functional.reset_net(model)
        x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
        logits=model(x)
        loss=criterion(logits.reshape(-1,K), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        tl+=loss.item()*y.size(0); seen+=y.size(0)
        if bi%20==0: print(f"Epoch {epoch}/{EPOCHS} | Batch {bi}/{len(train_loader)} | Loss {loss.item():.4f}",flush=True)
    functional.reset_net(model); m=evaluate()
    dt=(time.time()-t0)/60; el=(time.time()-start)/3600
    print(f"Epoch : {epoch} | TrainLoss {tl/seen:.4f} | Recall {m['recall']:.4f} "
          f"Prec {m['precision']:.4f} F1 {m['f1']:.4f} | FalseAlarm {m['far']:.4f} | "
          f"Loc {m['loc']:.4f} | FrameAcc {m['facc']:.4f} | {dt:.1f} min | {el:.2f} h",flush=True)
    wandb.log(dict(epoch=epoch,**{f"val/{k}":v for k,v in m.items()},
                   **{"train/loss":tl/seen,"time/epoch_min":dt,"time/elapsed_h":el}),step=epoch)
    torch.save({"epoch":epoch,"model":model.state_dict(),"opt":optimizer.state_dict(),"best":best_f1},LAST)
    if m['f1']>best_f1:
        best_f1=m['f1']; torch.save(model.state_dict(),BEST)
        print(f"best saved F1={best_f1:.4f} (recall {m['recall']:.4f} prec {m['precision']:.4f})",flush=True)
print("training finished",flush=True); print(f"best F1: {best_f1:.4f}",flush=True); wandb.finish()
