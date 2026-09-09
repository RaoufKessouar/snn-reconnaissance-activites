import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
ROOT=Path(__file__).resolve().parents[3]; EXP=ROOT/"experiments/mad"
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(EXP/"common"))
from src.model_tdbn_fpp import SResNestFPP
from mad_fall_dataset import MADFallDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
T,SEQ,BS=50,5,3; ACT=(1,2,3,4,5,6,7,8,9); FALL=9; VAL_P=list(range(71,86))
dev="cuda" if torch.cuda.is_available() else "cpu"
ds=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=VAL_P,activities=ACT,fall_activity=FALL,
                  seq_len=SEQ,T=T,samples_per_participant=30,p_fall=0.5,seed=1,split_name="val")
ld=DataLoader(ds,batch_size=BS,shuffle=False,num_workers=4); fi=ds.fall_idx
m=SResNestFPP(num_steps=T,num_classes=len(ACT)).to(dev)
m.load_state_dict(torch.load(EXP/"06_fall_detect"/"best_eval.pth",map_location=dev)); m.eval()
cnts,truths=[],[]
with torch.no_grad():
    for x,y in ld:
        functional.reset_net(m)
        x=x.to(dev,dtype=torch.float32); y=y.to(dev)
        preds=m(x).argmax(-1)
        cnts.append((preds==fi).sum(1).cpu()); truths.append((y==fi).any(1).cpu())
functional.reset_net(m)
cnt=torch.cat(cnts); truth=torch.cat(truths)
print(f"{'K frames':>9} {'Recall':>8} {'Prec':>8} {'F1':>8} {'FalseAlarm':>11}")
for K in [1,2,3,4,5]:
    pf=cnt>=K
    TP=(truth&pf).sum().item(); FP=((~truth)&pf).sum().item()
    FN=(truth&~pf).sum().item(); TN=((~truth)&~pf).sum().item()
    rec=TP/max(TP+FN,1); pre=TP/max(TP+FP,1); f1=2*pre*rec/max(pre+rec,1e-8); far=FP/max(FP+TN,1)
    print(f"{K:>9} {rec:>8.3f} {pre:>8.3f} {f1:>8.3f} {far:>11.3f}")
