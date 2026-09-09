import sys, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT/"common"))
from model import SResNest
from mad_dataset import MADDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

T=40; device="cuda"
val_set = MADDataset(EXTRACT_DIR, CACHE_DIR, participants=list(range(25,31)), T=T)
val_loader = DataLoader(val_set, batch_size=6, shuffle=False, num_workers=4)
num_classes = len(val_set.classes)
ckpt = EXP_ROOT/"03_train_9class"/"best_9class.pth"
print(f"val_samples={len(val_set)} num_classes={num_classes}", flush=True)

def ev(tag, setup):
    m = SResNest(num_steps=T, num_classes=num_classes).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    setup(m)
    acc=Accuracy(task="multiclass",num_classes=num_classes).to(device); crit=nn.CrossEntropyLoss(); tot=0.0;n=0
    with torch.no_grad():
        for x,y in val_loader:
            functional.reset_net(m)
            x=x.to(device=device,dtype=torch.float32); y=y.to(device=device,dtype=torch.long)
            o=m(x); tot+=crit(o,y).item()*y.size(0); n+=y.size(0); acc.update(o,y)
    functional.reset_net(m)
    print(f"[{tag}]  val_acc={acc.compute().item():.4f}  val_loss={tot/n:.4f}", flush=True)

ev("A: eval, stats du checkpoint (train)", lambda m: m.eval())
ev("B: precise-BN adapte a la VAL puis eval", lambda m: (update_bntt_running_stats(m, val_loader, device, 50), m.eval()))
ev("C: batch-stats (train mode sur val)", lambda m: m.train())
