import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/mad"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(EXP/"common"))
from src.model_tdbn import SResNest
from mad_chain_dataset import MADChainDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR

T, SEQ, BS = 40, 4, 4
ACT = (1, 3, 9); TEST_P = list(range(86, 101))
dev = "cuda" if torch.cuda.is_available() else "cpu"
BEST = EXP/"04_train_chain"/"best_chain_tdbn_L4.pth"

ds = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=TEST_P, activities=ACT,
                     seq_len=SEQ, T=T, samples_per_class=3, split_name="test")
ld = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
nc = len(ds.class_list); class_list = ds.class_list
m = SResNest(num_steps=T, num_classes=nc).to(dev)
m.load_state_dict(torch.load(BEST, map_location=dev))

def run():
    m.eval(); P=[]; Y=[]
    with torch.no_grad():
        for x, y in ld:
            functional.reset_net(m)
            x = x.to(dev, dtype=torch.float32); y = y.to(dev, dtype=torch.long)
            P.append(m(x).argmax(1).cpu()); Y.append(y.cpu())
    functional.reset_net(m)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()

def met(preds, trues):
    exact = float((preds == trues).mean())
    Pp = np.array([class_list[i] for i in preds])
    Tt = np.array([class_list[i] for i in trues])
    pos = (Pp == Tt).mean(axis=0)
    return exact, [round(float(a), 4) for a in pos]

preds, trues = run(); exact, position_accuracy = met(preds, trues)

print(f"[L4 TEST] classes={nc}")
print(f"  exact-match (chaines completes) : {exact:.4f}")
print(f"  accuracy par position : {position_accuracy}")
