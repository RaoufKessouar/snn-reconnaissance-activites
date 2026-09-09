import sys, os, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from spikingjelly.clock_driven import functional
import wandb

EXP_ROOT = Path(__file__).resolve().parents[1]
SRESNET_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(EXP_ROOT / "common"))
from src.model_tdbn import SResNest
from src.reproducibility import seed_from_environment
from mad_chain_dataset import MADChainDataset
from precise_bn import update_bntt_running_stats
from mad_paths import EXTRACT_DIR, CACHE_DIR

# ---------------- config ----------------
SEED = seed_from_environment()
T = 40
BS = int(os.environ.get("BS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
LR, WD = 1e-4, 0.01
TRAIN_P = list(range(1, 71)); VAL_P = list(range(71, 86))
ADA_BN_BATCHES = 40
ACTIVITIES = [1, 3, 9]; SEQ_LEN = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = EXP_ROOT / "04_train_chain"
LAST = OUT / "last_chain_tdbn_L4.pth"; BEST = OUT / "best_chain_tdbn_L4.pth"
run_name = f"madchain_TDBN_L{SEQ_LEN}_T{T}_bs{BS}_ep{EPOCHS}"

wandb.init(
    project="Article5-MAD-Chain",
    name=run_name,
    config={"T": T, "batch_size": BS, "epochs": EPOCHS, "lr": LR, "weight_decay": WD,
            "optimizer": "AdamW", "model": "SResNet38", "activities": ACTIVITIES,
            "seq_len": SEQ_LEN, "num_classes": len(ACTIVITIES) ** SEQ_LEN,
            "seed": SEED,
            "split": "participants 1-70 train / 71-85 val", "calibration_source": "train",
            "calibration_batches": ADA_BN_BATCHES},
)

# ---------------- data ----------------
train_set = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=TRAIN_P, activities=ACTIVITIES,
                            seq_len=SEQ_LEN, T=T, samples_per_class=1, split_name="train", augment=True)
val_set = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=VAL_P, activities=ACTIVITIES,
                          seq_len=SEQ_LEN, T=T, samples_per_class=3, split_name="val")
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
num_classes = len(train_set.class_list)

print(f"Train samples: {len(train_set)}", flush=True)
print(f"Validation samples: {len(val_set)}", flush=True)
print(f"device: {device}", flush=True)
print(f"num_classes: {num_classes}", flush=True)
print(f"batch_size: {BS} | epochs: {EPOCHS}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

start_epoch, best_val_acc = 1, 0.0
if os.environ.get("RESUME") and LAST.exists():
    ck = torch.load(LAST, map_location=device)
    model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
    start_epoch = ck["epoch"] + 1; best_val_acc = ck["best"]
    print(f"RESUME depuis epoch {ck['epoch']} (best={best_val_acc:.4f})", flush=True)


def evaluate(epoch, tag):
    model.eval()
    val_accuracy.reset()
    val_loss_sum = 0.0; val_seen = 0
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
            functional.reset_net(model)
            x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            bs = y_batch.size(0)
            val_loss_sum += loss.item() * bs; val_seen += bs
            val_accuracy.update(y_pred.detach(), y_batch)
            if batch_idx % 50 == 0:
                print(f"Validation[{tag}] | Epoch {epoch} | batch {batch_idx}/{len(val_loader)}", flush=True)
    functional.reset_net(model)
    return val_loss_sum / val_seen, val_accuracy.compute().item()


start_time = time.time()
print("start training", flush=True)

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    train_accuracy.reset()
    train_loss_sum = 0.0; train_seen = 0

    print(f"start epoch {epoch}/{EPOCHS}", flush=True)
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        functional.reset_net(model)
        x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = y_batch.size(0)
        train_loss_sum += loss.item() * bs; train_seen += bs
        train_accuracy.update(y_pred.detach(), y_batch)
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}", flush=True)

    functional.reset_net(model)
    train_loss = train_loss_sum / train_seen
    train_acc = train_accuracy.compute().item()

    val_loss_std, val_acc_std = evaluate(epoch, "std")
    update_bntt_running_stats(model, train_loader, device, num_batches=ADA_BN_BATCHES)
    val_loss_ada, val_acc_ada = evaluate(epoch, "ada")

    epoch_time = time.time() - epoch_start
    elapsed_hours = (time.time() - start_time) / 3600.0

    print(
        f"Epoch : {epoch} | "
        f"Train Loss : {train_loss:.4f} | Train Accuracy : {train_acc:.4f} | "
        f"Val(std) Loss : {val_loss_std:.4f} | Val(std) Accuracy : {val_acc_std:.4f} | "
        f"Val(ada) Loss : {val_loss_ada:.4f} | Val(ada) Accuracy : {val_acc_ada:.4f} | "
        f"Epoch Time : {epoch_time / 60:.2f} min | Elapsed : {elapsed_hours:.2f} h",
        flush=True,
    )

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss, "train/accuracy": train_acc,
        "val_std/loss": val_loss_std, "val_std/accuracy": val_acc_std,
        "val_ada/loss": val_loss_ada, "val_ada/accuracy": val_acc_ada,
        "time/epoch_minutes": epoch_time / 60.0, "time/elapsed_hours": elapsed_hours,
        "lr": optimizer.param_groups[0]["lr"],
    }, step=epoch)

    torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict(),
                "best": best_val_acc}, LAST)
    if val_acc_ada > best_val_acc:
        best_val_acc = val_acc_ada
        torch.save(model.state_dict(), BEST)
        print(f"best model saved with Val(ada) accuracy {best_val_acc:.4f}", flush=True)

print("training finished", flush=True)
print(f"best validation accuracy (ada): {best_val_acc:.4f}", flush=True)
wandb.finish()
