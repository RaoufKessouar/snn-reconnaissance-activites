import os, sys, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
ROOT=Path(__file__).resolve().parents[3]; EXP=ROOT/"experiments/mad"
for p in (ROOT, EXP/"common", EXP/"06_fall_detect"): sys.path.insert(0,str(p))
from src.model_tdbn_fpp import SResNestFPP
from mad_fall_dataset import MADFallDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
SPLIT=os.environ.get("SPLIT","test")
PARTS={"val":list(range(71,86)),"test":list(range(86,101))}[SPLIT]
T,SEQ,ACT=50,5,tuple(range(1,10))
SPP=int(os.environ.get("SPP","30")); BS=int(os.environ.get("BS","3"))
FALL_IDX=int(os.environ.get("FALL_IDX","8"))
CKPT=EXP/"06_fall_detect"/"best_T50_fb2.0_kcurve.pth"
dev="cuda" if torch.cuda.is_available() else "cpu"
ds=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=PARTS,activities=ACT,fall_activity=9,
                  seq_len=SEQ,T=T,samples_per_participant=SPP,p_fall=0.5,seed=1,split_name=SPLIT)
ld=DataLoader(ds,batch_size=BS,shuffle=False,num_workers=4)
m=SResNestFPP(num_steps=T,num_classes=len(ACT)).to(dev)
m.load_state_dict(torch.load(CKPT,map_location=dev)); m.eval()
nfall=[]; gt=[]; fcorr=ftot=0; labset=set()
with torch.no_grad():
    for x,y in ld:
        functional.reset_net(m); x=x.to(dev,torch.float32)
        p=m(x).argmax(-1).cpu().numpy(); yy=y.numpy()
        labset.update(np.unique(yy).tolist()); fcorr+=(p==yy).sum(); ftot+=yy.size
        for i in range(p.shape[0]):
            nfall.append(int((p[i]==FALL_IDX).sum())); gt.append(bool((yy[i]==FALL_IDX).any()))
functional.reset_net(m)
nfall=np.array(nfall); gt=np.array(gt)
print(f"[{SPLIT}] N={len(gt)} sequences | chutes={int(gt.sum())} | acc/trame={fcorr/ftot:.4f} | labels={sorted(labset)}")
print("K  sensibilite  precision  F1     fausses_alarmes")
for K in range(1,6):
    alert=nfall>=K
    TP=int((alert&gt).sum()); FN=int((~alert&gt).sum()); FP=int((alert&~gt).sum()); TN=int((~alert&~gt).sum())
    rec=TP/max(TP+FN,1); prec=TP/max(TP+FP,1); f1=2*prec*rec/max(prec+rec,1e-9); fa=FP/max(FP+TN,1)
    print(f"{K}  {rec:.3f}        {prec:.3f}      {f1:.3f}  {fa:.3f}")
