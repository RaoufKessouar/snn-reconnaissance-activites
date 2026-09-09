import sys, os, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb

EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT/"common"))
from src.model import SResNest
from mad_chain_dataset import MADChainDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

T = 50
BS = int(os.environ.get("BS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "80"))
SWA_START = int(os.environ.get("SWA_START", "55"))   # moyennage a partir de cet epoch
LR, WD = 1e-4, 0.01
TRAIN_P = list(range(1,71)); VAL_P = list(range(71,86)); TEST_P = list(range(86,101))
ADA = 40
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = EXP_ROOT/"04_train_chain"

def mkset(P, spc, name): return MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=P, T=T,
                                                samples_per_class=spc, split_name=name, augment=False)
train_set = mkset(TRAIN_P,1,"train"); val_set = mkset(VAL_P,3,"val")
train_loader = DataLoader(train_set,batch_size=BS,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
val_loader   = DataLoader(val_set,batch_size=BS,shuffle=False,num_workers=4,pin_memory=True)
num_classes = len(train_set.class_list)
print(f"classes={num_classes} train={len(train_set)} val={len(val_set)} BS={BS} EPOCHS={EPOCHS} SWA_START={SWA_START}", flush=True)

wandb.init(project="Article5-MAD-Chain", name=f"madchain_wsf_T50_bs{BS}_ep{EPOCHS}_SWA",
           config={"T":T,"batch_size":BS,"epochs":EPOCHS,"swa_start":SWA_START,"note":"baseline+SWA"})

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()
tr_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

def acc_on(loader):
    a = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    with torch.no_grad():
        for x,y in loader:
            functional.reset_net(model)
            x=x.to(device=device,dtype=torch.float32); y=y.to(device=device,dtype=torch.long)
            a.update(model(x), y)
    functional.reset_net(model); return a.compute().item()

swa=None; swa_n=0; best=0.0
for ep in range(1, EPOCHS+1):
    t0=time.time(); model.train(); tr_acc.reset(); tl=0.0; seen=0
    for i,(x,y) in enumerate(train_loader):
        functional.reset_net(model)
        x=x.to(device=device,dtype=torch.float32); y=y.to(device=device,dtype=torch.long)
        o=model(x); loss=crit(o,y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); seen+=y.size(0); tr_acc.update(o.detach(),y)
        if i%20==0: print(f"ep{ep} b{i}/{len(train_loader)} loss {loss.item():.3f}", flush=True)
    tr_l=tl/seen; tr_a=tr_acc.compute().item()
    model.eval(); vsa=acc_on(val_loader)
    update_bntt_running_stats(model, train_loader, device, ADA); model.eval(); vaa=acc_on(val_loader)
    print(f"Epoch : {ep} | Train Loss : {tr_l:.4f} | Train Accuracy : {tr_a:.4f} | "
          f"Val(std) Accuracy : {vsa:.4f} | Val(ada) Accuracy : {vaa:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
    wandb.log({"epoch":ep,"train/accuracy":tr_a,"val_std/accuracy":vsa,"val_ada/accuracy":vaa}, step=ep)
    if ep >= SWA_START:
        cur={n:p.detach().clone().float() for n,p in model.named_parameters()}
        if swa is None: swa=cur; swa_n=1
        else:
            for k in swa: swa[k]=(swa[k]*swa_n+cur[k])/(swa_n+1)
            swa_n+=1
        print(f"  [SWA] snapshot {swa_n}", flush=True)
    if vaa>best: best=vaa; torch.save(model.state_dict(), OUT/"best_chain.pth")

# ---- finalisation SWA ----
if swa is not None:
    print(f"\n[SWA] moyennage de {swa_n} snapshots + recalcul BN", flush=True)
    with torch.no_grad():
        for n,p in model.named_parameters(): p.copy_(swa[n].to(p.dtype))
    update_bntt_running_stats(model, train_loader, device, 100)
    torch.save(model.state_dict(), OUT/"swa_chain.pth")
    test_loader = DataLoader(mkset(TEST_P,3,"test"), batch_size=BS, shuffle=False, num_workers=4)
    def full(tag, ld):
        update_bntt_running_stats(model, train_loader, device, 100); model.eval(); s=acc_on(ld)
        update_bntt_running_stats(model, ld, device, 60); model.eval(); a=acc_on(ld)
        print(f"[SWA] {tag} : std={s:.4f}  ada={a:.4f}", flush=True)
    full("VAL", val_loader); full("TEST", test_loader)
print("done", flush=True); wandb.finish()
