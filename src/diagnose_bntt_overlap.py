from pathlib import Path
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional

from src.model import SResNest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments" / "dvsgc_overlap" / "overlap_078_seq3_T60"
sys.path.insert(0, str(EXP_DIR))

from dvsgc_overlap import DVSGestureChain

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))

T = 60
batch_size = 4

ckpt = EXP_DIR / "checkpoints" / "best_model_overlap078_seq3_T60_bs4_accum2.pth"

val_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="validation",
    validation=0.2,
    alpha_min=0.5,
    alpha_max=0.7,
    seq_len=3,
    class_num=3,
    repeat=True,
    dvsg_path=str(DATA_ROOT / "events_np"),
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(val_set.classes)
nb = len(val_loader)

print(f"checkpoint={ckpt}", flush=True)
print(f"device={device}", flush=True)
print(f"num_classes={num_classes} val_samples={len(val_set)} batches={nb}", flush=True)

def evaluate(mode):
    model = SResNest(num_steps=T, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    if mode == "train":
        model.train()
    else:
        model.eval()

    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    tot = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            functional.reset_net(model)

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            p = model(x)
            tot += crit(p, y).item()
            acc.update(p, y)

            if i % 50 == 0:
                print(f"  {mode} {i}/{nb}", flush=True)

    functional.reset_net(model)

    a = acc.compute().item()
    l = tot / nb

    print(f">>> mode={mode:5s} loss={l:.4f} acc={a:.4f}", flush=True)
    return a

ae1 = evaluate("eval")
ae2 = evaluate("eval")
at = evaluate("train")

print(f"\nOVERLAP eval1={ae1:.4f} eval2={ae2:.4f} diff_eval={abs(ae1-ae2):.4f}", flush=True)
print(f"OVERLAP eval_running={ae1:.4f} train_batch_stats={at:.4f} diff_train_eval={at-ae1:+.4f}", flush=True)
