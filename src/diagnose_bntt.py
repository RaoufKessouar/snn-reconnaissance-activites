from pathlib import Path
import os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
from src.model import SResNest
from src.dvsgc import DVSGestureChain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))
T, batch_size = 60, 4
ckpt = PROJECT_ROOT / "experiments/dvsgc_standard/best_model_sresnet38_bs4_accum2_lr1e-4_val.pth"

val_set = DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split="validation",
    validation=0.2, split_by="number", alpha_min=0.5, alpha_max=0.7,
    seq_len=4, class_num=3, repeat=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(val_set.classes); nb = len(val_loader)
print(f"num_classes={num_classes} val_samples={len(val_set)} batches={nb}", flush=True)

def evaluate(mode):
    model = SResNest(num_steps=T, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.train() if mode=="train" else model.eval()
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss(); tot=0.0
    with torch.no_grad():
        for i,(x,y) in enumerate(val_loader):
            functional.reset_net(model)
            x=x.to(device,dtype=torch.float32); y=y.to(device,dtype=torch.long)
            p=model(x); tot+=crit(p,y).item(); acc.update(p,y)
            if i%50==0: print(f"  {mode} {i}/{nb}", flush=True)
    functional.reset_net(model)
    a=acc.compute().item(); l=tot/nb
    print(f">>> mode={mode:5s} loss={l:.4f} acc={a:.4f}", flush=True)
    return a

ae=evaluate("eval")
at=evaluate("train")
print(f"\nH1 BNTT : (train-stats) - (eval-stats) = {at-ae:+.4f}", flush=True)
print("   ecart grand (>~0.05) -> running stats BNTT en cause", flush=True)
