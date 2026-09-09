from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional

from src.model import SResNest
from src.dvsgc import DVSGestureChain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))

T = 60
batch_size = 4
checkpoint_path = PROJECT_ROOT / "experiments/dvsgc_standard/best_model_val.pth"


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
print(f"checkpoint: {checkpoint_path}")
print(f"num_classes: {num_classes}")
print(f"validation samples: {len(val_set)}")
print(f"batch_size: {batch_size}")


def evaluate(mode_name):
    model = SResNest(num_steps=T, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if mode_name == "eval":
        model.eval()
    elif mode_name == "train":
        model.train()
    else:
        raise ValueError(mode_name)

    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    accuracy.reset()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            functional.reset_net(model)

            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.long)

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            accuracy.update(y_pred, y_batch)

            if batch_idx % 50 == 0:
                print(f"{mode_name} | batch {batch_idx}/{len(val_loader)}", flush=True)

    functional.reset_net(model)

    avg_loss = total_loss / len(val_loader)
    acc = accuracy.compute().item()

    print(f"\nMode: {mode_name}")
    print(f"Validation Loss     : {avg_loss:.4f}")
    print(f"Validation Accuracy : {acc:.4f}")
    print("-" * 50)

    return avg_loss, acc


print("\n===== EVALUATION EN model.eval() =====")
evaluate("eval")

print("\n===== EVALUATION EN model.train() SANS GRADIENT =====")
evaluate("train")
