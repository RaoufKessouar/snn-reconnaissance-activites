import os, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/mad"
for p in (ROOT, EXP / "common", EXP / "04_train_chain"): sys.path.insert(0, str(p))
from mad_chain_dataset import MADChainDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
from model_gru import RNNChain

ACT = (1, 3, 9); SEQ = 3; T = 40
TRAIN_P = list(range(1, 71)); VAL_P = list(range(71, 86)); TEST_P = list(range(86, 101))
SPC_TRAIN = int(os.environ.get("SPC_TRAIN", "70"))   # <-- a verifier (cf. note)
SPC_VAL   = int(os.environ.get("SPC_VAL",   "3"))
SPC_TEST  = int(os.environ.get("SPC_TEST",  "3"))
BS = int(os.environ.get("BS", "16")); EPOCHS = int(os.environ.get("EPOCHS", "60"))
RNN = os.environ.get("RNN", "gru"); LR = float(os.environ.get("LR", "1e-4"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = EXP / "04_train_chain" / f"best_chain_{RNN}.pth"

def make(parts, spc, split):
    return MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=parts, activities=ACT,
                           seq_len=SEQ, T=T, samples_per_class=spc, split_name=split)
tr = make(TRAIN_P, SPC_TRAIN, "train"); va = make(VAL_P, SPC_VAL, "val"); te = make(TEST_P, SPC_TEST, "test")
print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | classes {len(tr.class_list)} | rnn={RNN}", flush=True)
ldtr = DataLoader(tr, batch_size=BS, shuffle=True,  num_workers=6, drop_last=True)
ldva = DataLoader(va, batch_size=BS, shuffle=False, num_workers=4)
ldte = DataLoader(te, batch_size=BS, shuffle=False, num_workers=4)
m = RNNChain(num_classes=len(tr.class_list), rnn=RNN).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss()

def evaluate(ld):
    m.eval(); c = t = 0
    with torch.no_grad():
        for x, y in ld:
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
    va_acc = evaluate(ldva)
    if va_acc > best: best = va_acc; torch.save(m.state_dict(), CKPT)
    print(f"Epoch {ep}/{EPOCHS} | loss {run/max(nb,1):.3f} | val {va_acc:.4f} | best {best:.4f} | {time.time()-t0:.0f}s", flush=True)

m.load_state_dict(torch.load(CKPT, map_location=dev))
print(f"[{RNN.upper()} TEST] participants 86-100 | test acc = {evaluate(ldte):.4f}", flush=True)
