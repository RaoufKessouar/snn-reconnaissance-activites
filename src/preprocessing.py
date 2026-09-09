import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.dvsgc import DVSGestureChain

DATA_ROOT = Path(os.environ.get("DVSGC_DATA_ROOT", REPOSITORY_ROOT / "data"))

T = 60

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
    repeat=True
)

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
    repeat=True
)

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
    repeat=True
)

print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Test:", len(test_set))

sample, label = train_set[0]
print("Sample shape:", sample.shape)
print("Label:", label)
