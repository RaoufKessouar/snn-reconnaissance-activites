import os, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb
EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(EXP_DIR))
from model import SResNest
from dvsgc_overlap import DVSGestureChain

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))
T = 40
BS = int(os.environ.get("BS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
LR, WD = 1e-4, 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"
run_name = f"ann_bn_overlap078_seq3_T40_bs{BS}_ep{EPOCHS}"
BEST = EXP_DIR / "best_ann.pth"

wandb.init(project="Article5-DVSGC-SResNet", name=run_name,
           config={"model":"ANN-BN (ReLU+BN)","T":T,"batch_size":BS,"epochs":EPOCHS,
                   "task":"overlap 0/7/8 seq3 (27 classes)"})

def mkset(split):
    return DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split=split, validation=0.2,
        split_by="number", alpha_min=0.5, alpha_max=0.7, seq_len=3, class_num=3, repeat=True,
        dvsg_path=str(DATA_ROOT/"events_np"))
train_set, val_set = mkset("train"), mkset("validation")
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)
num_classes = len(train_set.classes)
print(f"ANN-BN | classes={num_classes} train={len(train_set)} val={len(val_set)} BS={BS} EPOCHS={EPOCHS}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)   # utilise block.py ANN local
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()
tr_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
va_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
best = 0.0

def evaluate():
    model.eval(); va_acc.reset(); vl=0.0; n=0
    with torch.no_grad():
        for x,y in val_loader:
            functional.reset_net(model)
            x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
            o=model(x); vl+=crit(o,y).item()*y.size(0); n+=y.size(0); va_acc.update(o,y)
    functional.reset_net(model)
    return vl/n, va_acc.compute().item()

for ep in range(1, EPOCHS+1):
    t0=time.time(); model.train(); tr_acc.reset(); tl=0.0; seen=0
    for i,(x,y) in enumerate(train_loader):
        functional.reset_net(model)
        x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
        o=model(x); loss=crit(o,y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); seen+=y.size(0); tr_acc.update(o.detach(),y)
        if i%20==0: print(f"ep{ep} b{i}/{len(train_loader)} loss {loss.item():.3f}", flush=True)
    vl,va = evaluate(); tr_l=tl/seen; tr_a=tr_acc.compute().item()
    print(f"Epoch : {ep} | Train Loss : {tr_l:.4f} | Train Accuracy : {tr_a:.4f} | "
          f"Val Loss : {vl:.4f} | Val Accuracy : {va:.4f} | {(time.time()-t0)/60:.2f} min", flush=True)
    wandb.log({"epoch":ep,"train/loss":tr_l,"train/accuracy":tr_a,"val/loss":vl,"val/accuracy":va}, step=ep)
    if va>best: best=va; torch.save(model.state_dict(), BEST); print(f"best {best:.4f}", flush=True)
print(f"best val acc: {best:.4f}", flush=True); wandb.finish()
