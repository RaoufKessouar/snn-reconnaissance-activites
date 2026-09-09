from pathlib import Path
import os, sys, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments" / "dvsgc_overlap" / "overlap_078_seq3_T60"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))
from src.model import SResNest
from dvsgc_overlap import DVSGestureChain

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))
T, BS, SPLIT = 60, 4, "validation"
CKPT = EXP_DIR / "checkpoints" / "best_model_overlap078_seq3_T60_bs4_accum2.pth"
OUT = EXP_DIR / "error_analysis"; OUT.mkdir(exist_ok=True)
NAMES = {"0":"Hand_Clapping","7":"Arm_Roll","8":"Air_Drums"}

ds = DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split=SPLIT, validation=0.2,
    alpha_min=0.5, alpha_max=0.7, seq_len=3, class_num=3, repeat=True,
    dvsg_path=str(DATA_ROOT / "events_np"))
loader = DataLoader(ds, batch_size=BS, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = list(ds.classes); num_classes = len(classes)
L = len(classes[0]); GEST = sorted(set("".join(classes)))
print(f"split={SPLIT} samples={len(ds)} num_classes={num_classes} L={L} gestures={GEST}", flush=True)
print("sanity mapping:", [(i, classes[i]) for i in [0,1,2,num_classes-1]], flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device)); model.eval()

trues, preds = [], []
with torch.no_grad():
    for i,(x,y) in enumerate(loader):
        functional.reset_net(model)
        x=x.to(device,dtype=torch.float32)
        p=model(x).argmax(1).cpu().tolist()
        trues += y.tolist(); preds += p
        if i%50==0: print(f"  {i}/{len(loader)}", flush=True)
functional.reset_net(model)

ts=[classes[t] for t in trues]; ps=[classes[p] for p in preds]
n=len(ts)
chain_acc=sum(a==b for a,b in zip(ts,ps))/n
print(f"\n=== CHAIN accuracy ({SPLIT}) = {chain_acc:.4f}  (n={n}) ===", flush=True)

# per-position
print("\n--- accuracy par position ---", flush=True)
pos_rows=[]
for pos in range(L):
    acc=sum(a[pos]==b[pos] for a,b in zip(ts,ps))/n
    pos_rows.append((pos+1,acc)); print(f"position {pos+1}: {acc:.4f}", flush=True)

# primitive confusion (aggregated over positions)
M={tg:{pg:0 for pg in GEST} for tg in GEST}
for a,b in zip(ts,ps):
    for pos in range(L): M[a[pos]][b[pos]]+=1
print("\n--- confusion primitives (lignes=vrai, cols=predit) ---", flush=True)
print("      "+"  ".join(f"{g}:{NAMES[g][:5]}" for g in GEST), flush=True)
for tg in GEST:
    tot=sum(M[tg].values()); acc=M[tg][tg]/tot if tot else 0
    print(f"{tg}:{NAMES[tg][:10]:11s} "+" ".join(f"{M[tg][pg]:6d}" for pg in GEST)+f"   acc={acc:.4f}", flush=True)

print("\n--- top confusions inter-gestes ---", flush=True)
conf=sorted(((M[t][p],t,p) for t in GEST for p in GEST if t!=p), reverse=True)
for c,t,p in conf: print(f"{t}:{NAMES[t]} -> {p}:{NAMES[p]} : {c}", flush=True)

# R-error (repetition), sur les chaines mal classees
wrong=[(a,b) for a,b in zip(ts,ps) if a!=b]
def rep(a,b,ref):  # ref='pred' -> b[p-1] ; ref='true' -> a[p-1]
    for pos in range(1,L):
        prev = b[pos-1] if ref=="pred" else a[pos-1]
        if b[pos]!=a[pos] and b[pos]==prev: return True
    return False
Rp=sum(rep(a,b,"pred") for a,b in wrong)/len(wrong) if wrong else 0
Rt=sum(rep(a,b,"true") for a,b in wrong)/len(wrong) if wrong else 0
print(f"\n--- R-error (repetition) sur {len(wrong)} chaines fausses ---", flush=True)
print(f"R_error(pred[p]==pred[p-1]) = {Rp:.4f}", flush=True)
print(f"R_error(pred[p]==true[p-1]) = {Rt:.4f}", flush=True)
print(f"baseline hasard (3 gestes)  = 0.3333", flush=True)

with open(OUT/"position_accuracy.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["position","accuracy"]); w.writerows(pos_rows)
with open(OUT/"gesture_confusion_matrix.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["true"]+GEST+["acc"])
    for tg in GEST:
        tot=sum(M[tg].values()); w.writerow([tg]+[M[tg][pg] for pg in GEST]+[f"{M[tg][tg]/tot:.4f}" if tot else 0])
print(f"\nsaved CSVs in {OUT}", flush=True)
