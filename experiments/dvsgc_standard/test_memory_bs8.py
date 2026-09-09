from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional

from model import SResNest
from dvsgc import DVSGestureChain


DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")

T = 60
batch_size = 8

train_set = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="train",
    validation=0.2,
    split_by="number",
    alpha_min=0.5,
    alpha_max=0.7,
    seq_len=4,
    class_num=3,
    repeat=True,
)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(train_set.classes)

print(f"device: {device}", flush=True)
print(f"num_classes: {num_classes}", flush=True)
print(f"batch_size: {batch_size}", flush=True)

model = SResNest(
    num_steps=T,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

x_batch, y_batch = next(iter(train_loader))

print("Input shape:", x_batch.shape, flush=True)
print("Label shape:", y_batch.shape, flush=True)

x_batch = x_batch.to(device=device, dtype=torch.float32)
y_batch = y_batch.to(device=device, dtype=torch.long)

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

model.train()
functional.reset_net(model)

optimizer.zero_grad(set_to_none=True)

print("start forward", flush=True)
y_pred = model(x_batch)

print("Output shape:", y_pred.shape, flush=True)

loss = criterion(y_pred, y_batch)
print("Loss:", loss.item(), flush=True)

print("start backward", flush=True)
loss.backward()

print("start optimizer step", flush=True)
optimizer.step()

functional.reset_net(model)

if device == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"CUDA allocated GB: {allocated:.2f}", flush=True)
    print(f"CUDA reserved GB: {reserved:.2f}", flush=True)
    print(f"CUDA peak allocated GB: {peak:.2f}", flush=True)

print("MEMORY TEST PASSED", flush=True)
