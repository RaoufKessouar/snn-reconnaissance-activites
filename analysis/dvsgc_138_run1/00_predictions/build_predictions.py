from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import SResNest
from dvsgc import DVSGestureChain


DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")
CHECKPOINT_PATH = PROJECT_ROOT / "analysis_output_1" / "best_model_first_run.pth"
OUTPUT_CSV = Path(__file__).resolve().parent / "predictions_test.csv"

T = 60
batch_size = 4

test_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="test",
    validation=0.2,
    split_by="number",
    alpha_min=0.5,
    alpha_max=0.7,
    seq_len=4,
    class_num=3,
    repeat=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(test_set.classes)

print(f"device: {device}", flush=True)
print(f"checkpoint: {CHECKPOINT_PATH}", flush=True)
print(f"test samples: {len(test_set)}", flush=True)
print(f"num_classes: {num_classes}", flush=True)

model = SResNest(num_steps=T, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

criterion_per_sample = nn.CrossEntropyLoss(reduction="none")

rows = []
global_index = 0

with torch.no_grad():
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        functional.reset_net(model)

        event_counts = x_batch.reshape(x_batch.shape[0], -1).sum(dim=1)

        x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)

        logits = model(x_batch)
        losses = criterion_per_sample(logits, y_batch)

        probs = torch.softmax(logits, dim=1)
        top5_probs, top5_labels = torch.topk(probs, k=5, dim=1)

        pred_labels = top5_labels[:, 0]
        confidences = top5_probs[:, 0]
        top2_margins = top5_probs[:, 0] - top5_probs[:, 1]
        entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

        for i in range(x_batch.shape[0]):
            true_label = int(y_batch[i].item())
            pred_label = int(pred_labels[i].item())
            correct = int(true_label == pred_label)

            top5_label_list = [int(v) for v in top5_labels[i].detach().cpu().tolist()]
            top5_prob_list = [float(v) for v in top5_probs[i].detach().cpu().tolist()]
            top5_name_list = [test_set.classes[v] for v in top5_label_list]

            rows.append({
                "sample_idx": global_index,
                "true_label": true_label,
                "true_class_name": test_set.classes[true_label],
                "pred_label": pred_label,
                "pred_class_name": test_set.classes[pred_label],
                "correct": correct,
                "loss": float(losses[i].detach().cpu().item()),
                "confidence": float(confidences[i].detach().cpu().item()),
                "top2_margin": float(top2_margins[i].detach().cpu().item()),
                "entropy": float(entropies[i].detach().cpu().item()),
                "event_count": float(event_counts[i].item()),
                "top5_labels": ";".join(map(str, top5_label_list)),
                "top5_class_names": ";".join(top5_name_list),
                "top5_probs": ";".join(f"{p:.6f}" for p in top5_prob_list),
            })

            global_index += 1

        if batch_idx % 50 == 0:
            print(f"batch {batch_idx}/{len(test_loader)}", flush=True)

functional.reset_net(model)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

acc = sum(r["correct"] for r in rows) / len(rows)

print(f"saved: {OUTPUT_CSV}", flush=True)
print(f"rows: {len(rows)}", flush=True)
print(f"accuracy from csv: {acc:.4f}", flush=True)
