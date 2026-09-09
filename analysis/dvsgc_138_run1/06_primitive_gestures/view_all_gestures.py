# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import csv
import random

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dvsgc import DVSGestureChain

DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")
OUT_DIR = Path(__file__).resolve().parent
PER_CLASS_DIR = OUT_DIR / "per_class_examples"

PER_CLASS_DIR.mkdir(exist_ok=True)

T = 60
N_EXAMPLES_PER_GESTURE = 4
FRAME_IDS = [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 59]

random.seed(7)
np.random.seed(7)

GESTURE_NAMES = {
    "0": "Hand_Clapping",
    "1": "Right_Hand_Wave",
    "2": "Left_Hand_Wave",
    "3": "Right_Arm_CW",
    "4": "Right_Arm_CCW",
    "5": "Left_Arm_CW",
    "6": "Left_Arm_CCW",
    "7": "Arm_Roll",
    "8": "Air_Drums",
    "9": "Air_Guitar",
    "10": "Other",
}

dataset = DVSGestureChain(
    root=str(DATA_ROOT),
    frames_number=T,
    split="train",
    validation=0.2,
    split_by="number",
    alpha_min=0.5,
    alpha_max=0.7,
    seq_len=1,
    class_num=11,
    repeat=True,
)

print(f"samples: {len(dataset)}")
print(f"classes: {dataset.classes}")


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def label_to_int(y):
    if hasattr(y, "item"):
        return int(y.item())
    return int(y)


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(";", "_")


def gesture_title(class_name):
    code = str(class_name)
    return f"{code} - {GESTURE_NAMES.get(code, 'Unknown')}"


def combine_polarities(frames):
    # frames shape: [T, 2, H, W]
    return frames[:, 1] - frames[:, 0]


def normalize_frame(frame):
    vmax = np.percentile(np.abs(frame), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(frame / vmax, -1, 1)


indices_by_label = {}

for idx, sample in enumerate(dataset.samples):
    label = label_to_int(sample[1])
    indices_by_label.setdefault(label, []).append(idx)

print("available gestures:")
for label, indices in sorted(indices_by_label.items()):
    class_name = dataset.classes[label]
    print(f"{label}: {gesture_title(class_name)} | samples={len(indices)}")


selected_main = {}
manifest_rows = []

for label, indices in sorted(indices_by_label.items()):
    selected = random.sample(indices, min(N_EXAMPLES_PER_GESTURE, len(indices)))
    selected_main[label] = selected[0]

    class_name = dataset.classes[label]

    fig, axes = plt.subplots(
        len(selected),
        len(FRAME_IDS),
        figsize=(22, 2.6 * len(selected)),
        gridspec_kw={
            "left": 0.02,
            "right": 0.98,
            "bottom": 0.04,
            "top": 0.86,
            "wspace": 0.02,
            "hspace": 0.22,
        },
    )

    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_id, sample_idx in enumerate(selected):
        frames, y = dataset[sample_idx]
        frames = to_numpy(frames).astype(np.float32)
        signed = combine_polarities(frames)
        event_count = float(frames.sum())

        manifest_rows.append({
            "gesture_label": label,
            "gesture_code": str(class_name),
            "gesture_name": GESTURE_NAMES.get(str(class_name), "Unknown"),
            "sample_idx": sample_idx,
            "event_count": event_count,
        })

        for col_id, t in enumerate(FRAME_IDS):
            ax = axes[row_id, col_id]
            ax.imshow(
                normalize_frame(signed[t]),
                cmap="bwr",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            ax.axis("off")

            if row_id == 0:
                ax.set_title(f"t={t}", fontsize=10)

            if col_id == 0:
                ax.set_ylabel(f"sample {sample_idx}", fontsize=9)

    fig.suptitle(
        f"Primitive gesture: {gesture_title(class_name)}",
        fontsize=18,
    )

    out_path = PER_CLASS_DIR / f"gesture_{label:02d}_{safe_name(class_name)}.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"saved: {out_path}")


# One big overview: one sample per gesture
labels = sorted(selected_main.keys())

fig, axes = plt.subplots(
    len(labels),
    len(FRAME_IDS),
    figsize=(22, 2.25 * len(labels)),
    gridspec_kw={
        "left": 0.08,
        "right": 0.98,
        "bottom": 0.03,
        "top": 0.95,
        "wspace": 0.02,
        "hspace": 0.18,
    },
)

axes = np.asarray(axes)

for row_id, label in enumerate(labels):
    sample_idx = selected_main[label]
    class_name = dataset.classes[label]

    frames, y = dataset[sample_idx]
    frames = to_numpy(frames).astype(np.float32)
    signed = combine_polarities(frames)

    for col_id, t in enumerate(FRAME_IDS):
        ax = axes[row_id, col_id]
        ax.imshow(
            normalize_frame(signed[t]),
            cmap="bwr",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.axis("off")

        if row_id == 0:
            ax.set_title(f"t={t}", fontsize=10)

        if col_id == 0:
            ax.set_ylabel(
                gesture_title(class_name),
                fontsize=10,
                rotation=0,
                labelpad=52,
                va="center",
            )

fig.suptitle("Overview of all primitive DVS-Gesture classes", fontsize=20)
overview_path = OUT_DIR / "all_gestures_overview.png"
plt.savefig(overview_path, dpi=200)
plt.close(fig)

with open(OUT_DIR / "visualized_samples_manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_rows)

print("done")
print(f"main overview: {overview_path}")
print(f"per-class examples: {PER_CLASS_DIR}")
print(f"manifest: {OUT_DIR / 'visualized_samples_manifest.csv'}")
