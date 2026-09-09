import sys, numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT/"common"))
from model import SResNest
from mad_chain_dataset import MADChainDataset
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR, CACHE_DIR

T=40; device="cuda"
test_set = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=list(range(86,101)),
                           activities=[1,3,9], seq_len=3, T=T, samples_per_class=3, split_name="test")
loader = DataLoader(test_set, batch_size=6, shuffle=False, num_workers=4)
class_list = test_set.class_list
GEST = [1,3,9]; L = 3
ckpt = EXP_ROOT/"04_train_chain"/"best_chain.pth"

model = SResNest(num_steps=T, num_classes=len(class_list)).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

trues, preds = [], []
with torch.no_grad():
    for i,(x,y) in enumerate(loader):
        functional.reset_net(model)
        x=x.to(device=device,dtype=torch.float32)
        preds += model(x).argmax(1).cpu().tolist(); trues += y.tolist()
        if i%50==0: print(f"{i}/{len(loader)}", flush=True)
functional.reset_net(model)

ts=[class_list[t] for t in trues]; ps=[class_list[p] for p in preds]; n=len(ts)
print(f"\n=== CHAIN accuracy (test) = {sum(a==b for a,b in zip(ts,ps))/n:.4f}  (n={n}) ===")

print("\n--- accuracy par position ---")
for pos in range(L):
    print(f"position {pos+1}: {sum(a[pos]==b[pos] for a,b in zip(ts,ps))/n:.4f}")

M={tg:{pg:0 for pg in GEST} for tg in GEST}
for a,b in zip(ts,ps):
    for pos in range(L): M[a[pos]][b[pos]]+=1
print("\n--- confusion primitives (lignes=vrai, cols=predit) ---")
print("            "+"   ".join(f"{ACTIVITY_NAMES[g]}" for g in GEST))
for tg in GEST:
    tot=sum(M[tg].values()); acc=M[tg][tg]/tot if tot else 0
    print(f"{ACTIVITY_NAMES[tg]:11s} "+" ".join(f"{M[tg][pg]:7d}" for pg in GEST)+f"   acc={acc:.4f}")

print("\n--- top confusions inter-gestes ---")
for c,t,p in sorted(((M[t][p],t,p) for t in GEST for p in GEST if t!=p), reverse=True):
    print(f"{ACTIVITY_NAMES[t]} -> {ACTIVITY_NAMES[p]} : {c}")

wrong=[(a,b) for a,b in zip(ts,ps) if a!=b]
def rep(a,b):
    return any(b[pos]!=a[pos] and b[pos]==b[pos-1] for pos in range(1,L))
Rp=sum(rep(a,b) for a,b in wrong)/len(wrong) if wrong else 0
print(f"\n--- R-error (repetition) sur {len(wrong)} chaines fausses ---")
print(f"R_error(pred[p]==pred[p-1]) = {Rp:.4f}   (hasard 3 gestes = 0.3333)")
