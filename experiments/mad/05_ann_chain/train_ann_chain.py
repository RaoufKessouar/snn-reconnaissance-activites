import sys, os, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb
HERE = Path(__file__).resolve().parent
MAD_ROOT = HERE.parent
sys.path.insert(0, str(MAD_ROOT/"common"))
sys.path.insert(0, str(HERE))               # priorite au model + block ANN locaux
from model import SResNest
from mad_chain_dataset import MADChainDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

T = 40
BS = int(os.environ.get("BS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
LR, WD = 1e-4, 0.01
TRAIN_P = list(range(1,71)); VAL_P = list(range(71,86)); TEST_P = list(range(86,101))
ADA = 40
device = "cuda" if torch.cuda.is_available() else "cpu"
run_name = f"ann_bn_madchain_T40_bs{BS}_ep{EPOCHS}"

wandb.init(project="Article5-MAD-Chain", name=run_name,
           config={"model":"ANN-BN (ReLU+BN)","T":T,"batch_size":BS,"epochs":EPOCHS,"note":"temoin ANN vs SNN"})

def mkset(P, spc, name): return MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=P, T=T,
                                                samples_per_class=spc, split_name=name, augment=False)
train_set = mkset(TRAIN_P,1,"train"); val_set = mkset(VAL_P,3,"val")
train_loader = DataLoader(train_set,batch_size=BS,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
val_loader   = DataLoader(val_set,batch_size=BS,shuffle=False,num_workers=4,pin_memory=True)
num_classes = len(train_set.class_list)
print(f"ANN-BN MAD-Chain | classes={num_classes} train={len(train_set)} val={len(val_set)} BS={BS}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)   # utilise le block ANN local
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()
tr_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

def acc_on(loader):
    a = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for x,y in loader:
            functional.reset_net(model)
            x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
            a.update(model(x), y)
    functional.reset_net(model); return a.compute().item()

best = 0.0
for ep in range(1, EPOCHS+1):
    t0=time.time(); model.train(); tr_acc.reset(); tl=0.0; seen=0
    for i,(x,y) in enumerate(train_loader):
        functional.reset_net(model)
        x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
        o=model(x); loss=crit(o,y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); seen+=y.size(0); tr_acc.update(o.detach(),y)
        if i%20==0: print(f"ep{ep} b{i}/{len(train_loader)} loss {loss.item():.3f}", flush=True)
    tr_l=tl/seen; tr_a=tr_acc.compute().item()
    model.eval(); vsa=acc_on(val_loader)
    update_bntt_running_stats(model, train_loader, device, ADA); model.eval(); vaa=acc_on(val_loader)
    print(f"Epoch : {ep} | Train Loss : {tr_l:.4f} | Train Accuracy : {tr_a:.4f} | "
          f"Val(std) Accuracy : {vsa:.4f} | Val(ada) Accuracy : {vaa:.4f} | {(time.time()-t0)/60:.2f} min", flush=True)
    wandb.log({"epoch":ep,"train/accuracy":tr_a,"val_std/accuracy":vsa,"val_ada/accuracy":vaa}, step=ep)
    if vsa>best: best=vsa; torch.save(model.state_dict(), HERE/"best_ann.pth")

test_loader = DataLoader(mkset(TEST_P,3,"test"), batch_size=BS, shuffle=False, num_workers=4)
model.load_state_dict(torch.load(HERE/"best_ann.pth"))
model.eval(); ts_std=acc_on(test_loader)
update_bntt_running_stats(model, test_loader, device, 60); model.eval(); ts_ada=acc_on(test_loader)
print(f"[ANN TEST] std={ts_std:.4f}  ada={ts_ada:.4f}", flush=True)
wandb.finish()
