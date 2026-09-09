from pathlib import Path
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model import SResNest
from src.dvsgc import DVSGestureChain


DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", REPOSITORY_ROOT / "data"))

T = 60
batch_size = 4

val_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="validation",
    validation=0.2,
    split_by="number",
    alpha_min=0.5,
    alpha_max=0.7,
    seq_len=4,
    class_num=3,
    repeat=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(val_set.classes)

print(f"device: {device}")
print(f"num_classes: {num_classes}")
print(f"validation samples: {len(val_set)}")

model = SResNest(num_steps=T, num_classes=num_classes).to(device)

checkpoint_path = Path(os.environ.get(
    "DVSGC_CHECKPOINT",
    Path(__file__).resolve().parent / "best_model_sresnet38_bs4_accum2_lr1e-4_val.pth",
))
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

criterion = nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

model.eval()
accuracy.reset()

val_loss = 0.0

with torch.no_grad():
    for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
        functional.reset_net(model)

        x_batch = x_batch.to(device=device, dtype=torch.float32)
        y_batch = y_batch.to(device=device, dtype=torch.long)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        val_loss += loss.item()
        accuracy.update(y_pred, y_batch)

        if batch_idx % 50 == 0:
            print(f"Validation batch {batch_idx}/{len(val_loader)}", flush=True)

functional.reset_net(model)

val_loss_moy = val_loss / len(val_loader)
val_accuracy = accuracy.compute().item()

print(f"Validation Loss : {val_loss_moy:.4f}")
print(f"Validation Accuracy : {val_accuracy:.4f}")
