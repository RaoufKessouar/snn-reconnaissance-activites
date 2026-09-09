# -*- coding: utf-8 -*-
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

T = 60
FRAME_IDS = [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 59]
N_EXAMPLES_PER_GESTURE = 3

random.seed(7)

GESTURE_NAMES = {
    0: "Hand_Clapping",
    1: "Right_Hand_Wave",
    2: "Left_Hand_Wave",
    3: "Right_Arm_CW",
    4: "Right_Arm_CCW",
    5: "Left_Arm_CW",
    6: "Left_Arm_CCW",
    7: "Arm_Roll",
    8: "Air_Drums",
    9: "Air_Guitar",
    10: "Other",
}

dataset = DVS128Gesture(
    root=str(DATA_ROOT),
    train=True,
    data_type="frame",
    frames_number=T,
    split_by="number",
)

print("Nombre total d'exemples train :", len(dataset))


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def label_to_int(y):
    if hasattr(y, "item"):
        return int(y.item())
    return int(y)


def signed_activity(frames):
    frames = to_numpy(frames).astype(np.float32)

    if frames.shape == (T, 2, 128, 128):
        return frames[:, 1] - frames[:, 0]

    if frames.ndim == 4 and frames.shape[1] == 2:
        return frames[:, 1] - frames[:, 0]

    if frames.ndim == 4 and frames[-1] == 2:
        return frames[..., 1] - frames[..., 0]

    raise ValueError(f"Format inattendu: {frames.shape}")


def normalize(img):
    vmax = np.percentile(np.abs(img), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(img / vmax, -1, 1)


indices_by_label = {}

for idx in range(len(dataset)):
    _, label = dataset[idx]
    label = label_to_int(label)
    indices_by_label.setdefault(label, []).append(idx)

print("Classes de gestes trouvees :")
for label in sorted(indices_by_label):
    print(label, GESTURE_NAMES.get(label, "Unknown"), "samples =", len(indices_by_label[label]))

for label in sorted(indices_by_label):
    gesture_name = GESTURE_NAMES.get(label, "Unknown")
    selected = random.sample(
        indices_by_label[label],
        min(N_EXAMPLES_PER_GESTURE, len(indices_by_label[label]))
    )

    fig, axes = plt.subplots(
        len(selected),
        len(FRAME_IDS),
        figsize=(22, 2.8 * len(selected)),
        gridspec_kw={
            "left": 0.03,
            "right": 0.98,
            "bottom": 0.05,
            "top": 0.84,
            "wspace": 0.02,
            "hspace": 0.25,
        },
    )

    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row, sample_idx in enumerate(selected):
        frames, y = dataset[sample_idx]
        signed = signed_activity(frames)

        for col, t in enumerate(FRAME_IDS):
            ax = axes[row, col]
            ax.imshow(
                normalize(signed[t]),
                cmap="bwr",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            ax.axis("off")

            if row == 0:
                ax.set_title(f"t={t}", fontsize=10)

            if col == 0:
                ax.set_ylabel(f"sample {sample_idx}", fontsize=9)

    fig.suptitle(
        f"Geste {label} : {gesture_name}",
        fontsize=20,
    )

    out_path = OUT_DIR / f"gesture_{label:02d}_{gesture_name}.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    print("saved:", out_path)

print("Termine.")
print("Images sauvegardees dans:", OUT_DIR)
