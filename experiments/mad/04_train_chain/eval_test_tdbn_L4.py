"""Evaluate the MAD-Chain L=4 checkpoint without test-set adaptation."""

import os
import sys
from pathlib import Path

import torch
from spikingjelly.clock_driven import functional
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

EXP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(EXP_ROOT / "common"))

from src.model_tdbn import SResNest
from mad_chain_dataset import MADChainDataset
from mad_paths import CACHE_DIR, EXTRACT_DIR
from precise_bn import recalibrate_running_stats

T, BATCH_SIZE = 40, int(os.environ.get("BS", "4"))
ACTIVITIES, SEQUENCE_LENGTH = (1, 3, 9), 4
TRAIN_PARTICIPANTS = list(range(1, 71))
TEST_PARTICIPANTS = list(range(86, 101))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path(os.environ.get(
    "MAD_CHECKPOINT",
    EXP_ROOT / "04_train_chain" / "best_chain_tdbn_L4.pth",
))


def make_dataset(participants, samples_per_class, split_name):
    return MADChainDataset(
        EXTRACT_DIR,
        CACHE_DIR,
        participants=participants,
        activities=ACTIVITIES,
        seq_len=SEQUENCE_LENGTH,
        T=T,
        samples_per_class=samples_per_class,
        split_name=split_name,
    )


test_set = make_dataset(TEST_PARTICIPANTS, samples_per_class=3, split_name="test")
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
model = SResNest(num_steps=T, num_classes=len(test_set.class_list)).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
accuracy = Accuracy(task="multiclass", num_classes=len(test_set.class_list)).to(DEVICE)


def evaluate():
    model.eval()
    accuracy.reset()
    with torch.no_grad():
        for inputs, targets in test_loader:
            functional.reset_net(model)
            inputs = inputs.to(DEVICE, dtype=torch.float32)
            targets = targets.to(DEVICE, dtype=torch.long)
            accuracy.update(model(inputs), targets)
    functional.reset_net(model)
    return accuracy.compute().item()


standard_accuracy = evaluate()
print(
    f"[tdBN L4 TEST] participants 86-100 | standard={standard_accuracy:.4f} "
    f"| N={len(test_set)} | classes={len(test_set.class_list)}"
)

if os.environ.get("RECALIBRATE_FROM_TRAIN") == "1":
    train_set = make_dataset(TRAIN_PARTICIPANTS, samples_per_class=1, split_name="train_calibration")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    batches = recalibrate_running_stats(model, train_loader, DEVICE, num_batches=40)
    print(f"[tdBN L4 TEST] train-recalibrated={evaluate():.4f} | calibration_batches={batches}")
