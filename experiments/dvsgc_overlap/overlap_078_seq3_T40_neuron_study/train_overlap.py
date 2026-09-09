from pathlib import Path
import os
import sys
import time

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))  # priorite au code local (block.py switch, model, dataset)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb

from model import SResNest
from dvsgc_overlap import DVSGestureChain
from precise_bn import update_bntt_running_stats
from src.reproducibility import seed_from_environment

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", PROJECT_ROOT / "data"))
SEED = seed_from_environment()

# ---- config (variables env : NEURON_MODE, BATCH_SIZE, EPOCHS, PRECISE_BN_BATCHES) ----
NEURON_MODE = os.environ.get("NEURON_MODE", "lif_sub")
T = 40
batch_size = int(os.environ.get("BATCH_SIZE", "8"))   # batch REEL (pas d'accumulation) -> stats BNTT
accum_steps = 1
effective_batch_size = batch_size * accum_steps
epochs = int(os.environ.get("EPOCHS", "80"))
PRECISE_BN_BATCHES = int(os.environ.get("PRECISE_BN_BATCHES", "50"))
lr = 1e-4
weight_decay = 0.01

run_name = f"sresnet38_overlap078_seq3_T40_bs{batch_size}_{NEURON_MODE}_ep{epochs}_preciseBN"
best_checkpoint_path = str(EXP_DIR / "checkpoints" / f"best_{NEURON_MODE}.pth")
last_checkpoint_path = str(EXP_DIR / "checkpoints" / f"last_{NEURON_MODE}.pth")

wandb.init(
    project="Article5-DVSGC-SResNet",
    name=run_name,
    config={
        "T": T, "batch_size": batch_size, "accum_steps": accum_steps,
        "effective_batch_size": effective_batch_size, "epochs": epochs,
        "lr": lr, "weight_decay": weight_decay, "optimizer": "AdamW",
        "model": "SResNet38", "class_num": 3, "seq_len": 3, "num_chain_classes": 27,
        "neuron_mode": NEURON_MODE, "precise_bn_batches": PRECISE_BN_BATCHES,
        "seed": SEED,
    },
)

train_set = DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split="train", validation=0.2,
    split_by="number", alpha_min=0.5, alpha_max=0.7, seq_len=3, class_num=3, repeat=True,
    dvsg_path=str(DATA_ROOT / "events_np"))
val_set = DVSGestureChain(root=str(DATA_ROOT), frames_number=T, split="validation", validation=0.2,
    split_by="number", alpha_min=0.5, alpha_max=0.7, seq_len=3, class_num=3, repeat=True,
    dvsg_path=str(DATA_ROOT / "events_np"))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_set.classes)
print(f"NEURON_MODE={NEURON_MODE} | batch_size={batch_size} | epochs={epochs} | precise_bn={PRECISE_BN_BATCHES}", flush=True)
print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)} | num_classes: {num_classes} | device: {device}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)


def evaluate(epoch):
    if PRECISE_BN_BATCHES > 0:
        nbn = update_bntt_running_stats(model, train_loader, device, num_batches=PRECISE_BN_BATCHES)
        print(f"precise-BN: stats BNTT recalculees sur {nbn} lots train", flush=True)
    model.eval()
    val_accuracy.reset()
    val_loss_sum = 0.0
    val_seen = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            functional.reset_net(model)
            x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            bs = y_batch.size(0)
            val_loss_sum += loss.item() * bs
            val_seen += bs
            val_accuracy.update(y_pred.detach(), y_batch)
    functional.reset_net(model)
    val_loss = val_loss_sum / val_seen
    val_acc = val_accuracy.compute().item()
    print(f"Validation | Epoch {epoch} | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}", flush=True)
    return val_loss, val_acc


best_val_acc = 0.0
start_time = time.time()
print("start training", flush=True)

for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    model.train()
    train_accuracy.reset()
    train_loss_sum = 0.0
    train_seen = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        functional.reset_net(model)
        x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        (loss / accum_steps).backward()
        do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bs = y_batch.size(0)
        train_loss_sum += loss.item() * bs
        train_seen += bs
        train_accuracy.update(y_pred.detach(), y_batch)
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}", flush=True)

    functional.reset_net(model)
    train_loss = train_loss_sum / train_seen
    train_acc = train_accuracy.compute().item()

    val_loss, val_acc = evaluate(epoch)

    epoch_time = time.time() - epoch_start
    elapsed_hours = (time.time() - start_time) / 3600.0
    print(f"Epoch : {epoch} | Train Loss : {train_loss:.4f} | Train Accuracy : {train_acc:.4f} | "
          f"Val Loss : {val_loss:.4f} | Val Accuracy : {val_acc:.4f} | "
          f"Epoch Time : {epoch_time / 60:.2f} min | Elapsed : {elapsed_hours:.2f} h", flush=True)

    wandb.log({"epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc,
               "val/loss": val_loss, "val/accuracy": val_acc,
               "time/epoch_minutes": epoch_time / 60.0, "time/elapsed_hours": elapsed_hours,
               "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_val_acc,
                "train_acc": train_acc, "val_acc": val_acc, "neuron_mode": NEURON_MODE}, last_checkpoint_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_checkpoint_path)
        print(f"best model saved with validation accuracy {best_val_acc:.4f}", flush=True)

print("training finished", flush=True)
print(f"best validation accuracy: {best_val_acc:.4f}", flush=True)
wandb.finish()
