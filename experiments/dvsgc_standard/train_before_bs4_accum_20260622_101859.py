from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb

from model import SResNest
from dvsgc import DVSGestureChain


DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")

T = 60
batch_size = 8
epochs = 100
lr = 1e-4
weight_decay = 0.01

alpha_min = 0.5
alpha_max = 0.7
seq_len = 4
class_num = 3
repeat = True
validation_ratio = 0.2
split_by = "number"

run_name = "sresnet38_81p_bs8_ep100_constant_lr1e-4_val"
best_checkpoint_path = "best_model_sresnet38_bs8_lr1e-4_val.pth"
last_checkpoint_path = "last_model_sresnet38_bs8_lr1e-4.pth"

run = wandb.init(
    project="Article5-DVSGC-SResNet",
    name=run_name,
    config={
        "model": "SResNet38",
        "dataset": "DVS-GC-81p",
        "T": T,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scheduler": "None",
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "seq_len": seq_len,
        "class_num": class_num,
        "repeat": repeat,
        "validation_ratio": validation_ratio,
        "split_by": split_by,
    },
)

train_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="train",
    validation=validation_ratio,
    split_by=split_by,
    alpha_min=alpha_min,
    alpha_max=alpha_max,
    seq_len=seq_len,
    class_num=class_num,
    repeat=repeat,
)

val_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="validation",
    validation=validation_ratio,
    split_by=split_by,
    alpha_min=alpha_min,
    alpha_max=alpha_max,
    seq_len=seq_len,
    class_num=class_num,
    repeat=repeat,
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=True,
)

x, y = next(iter(train_loader))
print("Input shape:", x.shape, flush=True)
print("Label shape:", y.shape, flush=True)
print("Labels:", y[:10], flush=True)
print("Input min/max:", x.min().item(), x.max().item(), flush=True)

print("dataloaders created", flush=True)
print(f"Train samples: {len(train_set)}", flush=True)
print(f"Validation samples: {len(val_set)}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}", flush=True)

num_classes = len(train_set.classes)
print(f"num_classes: {num_classes}", flush=True)

wandb.config.update({
    "num_classes": num_classes,
    "train_samples": len(train_set),
    "val_samples": len(val_set),
    "device": device,
})

model = SResNest(
    num_steps=T,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
)

train_accuracy_metric = Accuracy(
    task="multiclass",
    num_classes=num_classes,
).to(device)

val_accuracy_metric = Accuracy(
    task="multiclass",
    num_classes=num_classes,
).to(device)


def evaluate(model, loader, criterion, accuracy_metric, device):
    model.eval()
    accuracy_metric.reset()

    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            functional.reset_net(model)

            x_batch = x_batch.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            y_batch = y_batch.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            accuracy_metric.update(y_pred, y_batch)

    functional.reset_net(model)

    avg_loss = total_loss / len(loader)
    avg_acc = accuracy_metric.compute().item()

    return avg_loss, avg_acc


best_val_acc = 0.0
best_epoch = 0

print("start training", flush=True)

for epoch in range(epochs):
    epoch_start = time.time()

    model.train()
    train_accuracy_metric.reset()

    epoch_loss = 0.0

    print(f"start epoch {epoch + 1}/{epochs}", flush=True)

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        functional.reset_net(model)

        x_batch = x_batch.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        y_batch = y_batch.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        train_accuracy_metric.update(y_pred, y_batch)
        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}",
                flush=True,
            )

    functional.reset_net(model)

    train_loss = epoch_loss / len(train_loader)
    train_acc = train_accuracy_metric.compute().item()

    val_loss, val_acc = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        accuracy_metric=val_accuracy_metric,
        device=device,
    )

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start

    wandb.log(
        {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": current_lr,
            "time/epoch_seconds": epoch_time,
        },
        step=epoch + 1,
    )

    print(
        f"Epoch : {epoch + 1} | "
        f"Train Loss : {train_loss:.4f} | "
        f"Train Accuracy : {train_acc:.4f} | "
        f"Val Loss : {val_loss:.4f} | "
        f"Val Accuracy : {val_acc:.4f} | "
        f"LR : {current_lr:.8f} | "
        f"Time : {epoch_time:.1f}s",
        flush=True,
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1

        torch.save(model.state_dict(), best_checkpoint_path)
        wandb.save(best_checkpoint_path)

        print(
            f"best model saved at epoch {best_epoch} "
            f"with val accuracy {best_val_acc:.4f}",
            flush=True,
        )

torch.save(model.state_dict(), last_checkpoint_path)
wandb.save(last_checkpoint_path)

print(
    f"training finished | best epoch: {best_epoch} | "
    f"best val accuracy: {best_val_acc:.4f}",
    flush=True,
)

wandb.finish()