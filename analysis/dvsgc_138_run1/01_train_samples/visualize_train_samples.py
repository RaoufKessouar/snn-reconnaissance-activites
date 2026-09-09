from pathlib import Path
import sys
import random
import csv

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dvsgc import DVSGestureChain


DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")
OUT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

T = 60
SEQ_LEN = 4
N_SAMPLES = 12

random.seed(3)
np.random.seed(3)

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

print(f"train samples: {len(train_set)}")
print(f"num classes: {len(train_set.classes)}")


def safe_name(text):
    return str(text).replace("/", "_").replace(" ", "_")


def combine_polarities(frames):
    pos = frames[:, 1]
    neg = frames[:, 0]
    return pos - neg


def normalize_frame(frame):
    vmax = np.percentile(np.abs(frame), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(frame / vmax, -1, 1)


def class_to_human(class_name):
    return " -> ".join(
        f"{code}:{GESTURE_NAMES.get(code, 'Unknown')}"
        for code in str(class_name)
    )


def estimated_gesture_segments(total_frames=60, seq_len=4):
    base = total_frames // seq_len
    segments = []
    start = 0

    for i in range(seq_len):
        end = total_frames if i == seq_len - 1 else start + base
        segments.append((start, end))
        start = end

    return segments


def select_frames(start, end, max_frames=12):
    n = end - start
    if n <= max_frames:
        return list(range(start, end))
    return np.linspace(start, end - 1, max_frames, dtype=int).tolist()


def plot_frames_grid(frames, frame_ids, title, out_path, subtitle=None):
    combined = combine_polarities(frames)

    n = len(frame_ids)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(22, 11),
        gridspec_kw={
            "left": 0.02,
            "right": 0.98,
            "bottom": 0.03,
            "top": 0.84,
            "wspace": 0.03,
            "hspace": 0.18,
        },
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, t in zip(axes, frame_ids):
        img = normalize_frame(combined[t])
        ax.imshow(img, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"t={t}", fontsize=14, pad=2)
        ax.axis("off")

    full_title = title
    if subtitle is not None:
        full_title += "\n" + subtitle

    fig.suptitle(full_title, fontsize=22, y=0.97)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sample(sample_idx):
    frames, label = train_set[sample_idx]
    frames = np.asarray(frames)

    class_name = train_set.classes[label]
    human_chain = class_to_human(class_name)

    total_events = float(frames.sum())
    pos_events = float(frames[:, 1].sum())
    neg_events = float(frames[:, 0].sum())

    sample_dir = SAMPLES_DIR / f"sample_{sample_idx:05d}_label_{label:02d}_class_{safe_name(class_name)}"
    sample_dir.mkdir(exist_ok=True)

    overview_ids = np.linspace(0, T - 1, 12, dtype=int).tolist()
    overview_title = (
        f"Train sample {sample_idx} | label={label} | class={class_name}\n"
        f"{human_chain}"
    )
    overview_subtitle = (
        f"frames={T} | events={total_events:.0f} | "
        f"pos={pos_events:.0f} | neg={neg_events:.0f}"
    )

    plot_frames_grid(
        frames=frames,
        frame_ids=overview_ids,
        title=overview_title,
        subtitle=overview_subtitle,
        out_path=sample_dir / "chain_overview.png",
    )

    segments = estimated_gesture_segments(T, SEQ_LEN)

    for gesture_pos, (start, end) in enumerate(segments, start=1):
        code = str(class_name)[gesture_pos - 1]
        gesture_name = GESTURE_NAMES.get(code, "Unknown")
        frame_ids = select_frames(start, end, max_frames=12)

        gesture_frames_count = end - start
        gesture_events = float(frames[start:end].sum())
        gesture_pos_events = float(frames[start:end, 1].sum())
        gesture_neg_events = float(frames[start:end, 0].sum())

        title = (
            f"Sample {sample_idx} | class={class_name} | gesture {gesture_pos}/{SEQ_LEN}\n"
            f"code={code} | {gesture_name}"
        )
        subtitle = (
            f"estimated frames={start}-{end - 1} | "
            f"n_frames={gesture_frames_count} | shown={len(frame_ids)} | "
            f"events={gesture_events:.0f} | pos={gesture_pos_events:.0f} | neg={gesture_neg_events:.0f}"
        )

        out_name = (
            f"gesture_{gesture_pos:02d}_"
            f"code_{code}_{safe_name(gesture_name)}_"
            f"frames_{start:02d}_{end - 1:02d}_n{gesture_frames_count}.png"
        )

        plot_frames_grid(
            frames=frames,
            frame_ids=frame_ids,
            title=title,
            subtitle=subtitle,
            out_path=sample_dir / out_name,
        )

    return {
        "sample_idx": sample_idx,
        "label": label,
        "class_name": class_name,
        "human_chain": human_chain,
        "total_events": total_events,
        "pos_events": pos_events,
        "neg_events": neg_events,
        "sample_dir": str(sample_dir),
    }


indices_by_class = {}
for idx, (_, label) in enumerate(train_set.samples):
    indices_by_class.setdefault(label, []).append(idx)

class_ids = np.linspace(0, len(train_set.classes) - 1, N_SAMPLES, dtype=int).tolist()

selected_indices = []
for class_id in class_ids:
    selected_indices.append(random.choice(indices_by_class[class_id]))

print("selected indices:", selected_indices)

rows = []
for sample_idx in selected_indices:
    row = plot_sample(sample_idx)
    rows.append(row)
    print(f"saved sample folder: {row['sample_dir']}")

manifest_path = OUT_DIR / "train_samples_manifest.csv"
with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"manifest saved: {manifest_path}")
print("done")
