import sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional

EXP_ROOT = Path(__file__).resolve().parents[1]          # .../mad
SRESNET_ROOT = EXP_ROOT.parents[1]                       # .../SResNet
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT / "common"))
from model import SResNest
from mad_dataset import MADDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

T, BS, EPOCHS, LR, WD = 40, 6, 30, 1e-4, 0.01
TRAIN_P = list(range(1, 25)); VAL_P = list(range(25, 31))
device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = MADDataset(EXTRACT_DIR, CACHE_DIR, participants=TRAIN_P, T=T)
val_set   = MADDataset(EXTRACT_DIR, CACHE_DIR, participants=VAL_P,  T=T)
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
num_classes = len(train_set.classes)
print(f"classes={train_set.classes} | train={len(train_set)} val={len(val_set)} | num_classes={num_classes}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.CrossEntropyLoss()
tr_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
va_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
best = 0.0

for ep in range(1, EPOCHS + 1):
    t0 = time.time(); model.train(); tr_acc.reset(); tl = 0.0; seen = 0
    for i, (x, y) in enumerate(train_loader):
        functional.reset_net(model)
        x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.long)
        out = model(x); loss = crit(out, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tl += loss.item() * y.size(0); seen += y.size(0); tr_acc.update(out.detach(), y)
        if i % 20 == 0: print(f"ep{ep} b{i}/{len(train_loader)} loss {loss.item():.3f}", flush=True)
    update_bntt_running_stats(model, train_loader, device, num_batches=50)
    model.eval(); va_acc.reset(); vl = 0.0; vs = 0
    with torch.no_grad():
        for x, y in val_loader:
            functional.reset_net(model)
            x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.long)
            out = model(x); vl += crit(out, y).item() * y.size(0); vs += y.size(0); va_acc.update(out, y)
    functional.reset_net(model)
    va = va_acc.compute().item()
    print(f"Epoch : {ep} | Train Loss {tl/seen:.4f} Acc {tr_acc.compute().item():.4f} | "
          f"Val Loss {vl/vs:.4f} Acc {va:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
    if va > best:
        best = va; torch.save(model.state_dict(), EXP_ROOT / "03_train_9class" / "best_9class.pth")
print(f"best val acc: {best:.4f}", flush=True)
