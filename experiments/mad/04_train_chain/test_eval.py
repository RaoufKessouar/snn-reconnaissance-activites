import sys, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT/"common"))
from model import SResNest
from mad_chain_dataset import MADChainDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

T=40; device="cuda"
test_set = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=list(range(86,101)),
                           activities=[1,3,9], seq_len=3, T=T, samples_per_class=3, split_name="test")
test_loader = DataLoader(test_set, batch_size=6, shuffle=False, num_workers=4)
num_classes = len(test_set.class_list)
ckpt = EXP_ROOT/"04_train_chain"/"best_chain.pth"
print(f"test_samples={len(test_set)} num_classes={num_classes}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))

def ev(adapt):
    if adapt: update_bntt_running_stats(model, test_loader, device, num_batches=60)
    model.eval()
    acc=Accuracy(task="multiclass",num_classes=num_classes).to(device); crit=nn.CrossEntropyLoss(); tot=0.0;n=0
    with torch.no_grad():
        for x,y in test_loader:
            functional.reset_net(model)
            x=x.to(device=device,dtype=torch.float32); y=y.to(device=device,dtype=torch.long)
            o=model(x); tot+=crit(o,y).item()*y.size(0); n+=y.size(0); acc.update(o,y)
    functional.reset_net(model)
    print(f"[{'ada' if adapt else 'std'}] TEST acc={acc.compute().item():.4f}  loss={tot/n:.4f}", flush=True)

ev(False)
ev(True)
