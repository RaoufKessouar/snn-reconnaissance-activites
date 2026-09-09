import os
os.environ.setdefault("NEURON", "relu")          # ANN : ReLU au lieu de LIF
import sys, torch
from pathlib import Path
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
ROOT = Path(__file__).resolve().parents[3]; EXP = ROOT / "experiments/mad"
for p in (ROOT, EXP / "common", EXP / "04_train_chain"): sys.path.insert(0, str(p))
from src.model_tdbn import SResNest
from mad_chain_dataset import MADChainDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR

T, SEQ, ACT = 40, 3, (1, 3, 9)
TEST_P = list(range(86, 101))
SPC = int(os.environ.get("SPC_TEST", "1"))        # meme jeu test que le GRU (405 chaines)
BS  = int(os.environ.get("BS", "4"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = EXP / "04_train_chain" / "best_chain_tdbn_annrelu.pth"

ds = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=TEST_P, activities=ACT,
                     seq_len=SEQ, T=T, samples_per_class=SPC, split_name="test")
ld = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4)
m = SResNest(num_steps=T, num_classes=len(ds.class_list)).to(dev)
m.load_state_dict(torch.load(CKPT, map_location=dev)); m.eval()
c = t = 0
with torch.no_grad():
    for x, y in ld:
        functional.reset_net(m)
        x = x.to(dev, torch.float32); y = y.to(dev).long()
        c += (m(x).argmax(1) == y).sum().item(); t += y.numel()
print(f"[ANN-tdBN TEST] participants 86-100 | test acc = {c/max(t,1):.4f} | N={t} | SPC={SPC}", flush=True)
