import os, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent                       # overlap_078_seq3_T60
ROOT = HERE.parents[2]                                       # repository root
sys.path.insert(0, str(HERE))                                # dvsgc_overlap
sys.path.insert(0, str(ROOT / "experiments/mad/04_train_chain"))  # model_gru
from dvsgc_overlap import DVSGestureChain
from model_gru import RNNChain

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", ROOT / "data"))
T      = int(os.environ.get("T", "60"))
BS     = int(os.environ.get("BS", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
RNN    = os.environ.get("RNN", "gru")
LR     = float(os.environ.get("LR", "1e-4"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = HERE / "checkpoints" / f"best_dvsgc_{RNN}.pth"
CKPT.parent.mkdir(parents=True, exist_ok=True)

def ds(split):
    return DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split=split,
        validation=0.2, split_by="number", alpha_min=0.5, alpha_max=0.7,
        seq_len=3, class_num=3, repeat=True, dvsg_path=str(DATA_ROOT / "events_np"))

tr = ds("train"); va = ds("validation")
num_classes = len(tr.classes)
print(f"Train {len(tr)} | Val {len(va)} | classes {num_classes} | T={T} | rnn={RNN}", flush=True)
ldtr = DataLoader(tr, batch_size=BS, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
ldva = DataLoader(va, batch_size=BS, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

m = RNNChain(num_classes=num_classes, rnn=RNN).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss()

def evaluate():
    m.eval(); c = t = 0
    with torch.no_grad():
        for x, y in ldva:
            x = x.to(dev, torch.float32); y = y.to(dev).long()
            c += (m(x).argmax(1) == y).sum().item(); t += y.numel()
    return c / max(t, 1)

best = 0.0
for ep in range(1, EPOCHS + 1):
    m.train(); t0 = time.time(); run = nb = 0
    for x, y in ldtr:
        x = x.to(dev, torch.float32); y = y.to(dev).long()
        opt.zero_grad(); loss = crit(m(x), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        run += loss.item(); nb += 1
    va_acc = evaluate()
    if va_acc > best: best = va_acc; torch.save(m.state_dict(), CKPT)
    print(f"Epoch {ep}/{EPOCHS} | loss {run/max(nb,1):.3f} | val {va_acc:.4f} | best {best:.4f} | {time.time()-t0:.0f}s", flush=True)
print(f"[{RNN.upper()} DVS-GC] best validation = {best:.4f}", flush=True)
