from pathlib import Path
import tarfile
import tempfile
import shutil
import io
import os

import numpy as np
import matplotlib.pyplot as plt
import spikingjelly.datasets as sjds

ARCHIVE = Path("/users/abdekess61/raouf/SResNet/data/DvsGesture.tar.gz")
OUT_DIR = Path("/users/abdekess61/raouf/SResNet/analysis_output_1/06_gesture_classes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 60
N_EXAMPLES_PER_GESTURE = 3
FRAME_IDS = [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 59]

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

def events_to_frames(events, start_t, end_t, frames_number=60, H=128, W=128):
    mask = (events["t"] >= start_t) & (events["t"] < end_t)
    x = events["x"][mask].astype(np.int64)
    y = events["y"][mask].astype(np.int64)
    p = events["p"][mask].astype(np.int64)

    if len(x) == 0:
        return np.zeros((frames_number, 2, H, W), dtype=np.float32)

    p = (p > 0).astype(np.int64)
    borders = np.linspace(0, len(x), frames_number + 1, dtype=np.int64)

    frames = np.zeros((frames_number, 2, H, W), dtype=np.float32)

    for t in range(frames_number):
        s, e = borders[t], borders[t + 1]
        if e <= s:
            continue

        xs = x[s:e]
        ys = y[s:e]
        ps = p[s:e]

        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        np.add.at(frames[t], (ps[valid], ys[valid], xs[valid]), 1)

    return frames

def normalize(img):
    vmax = np.percentile(np.abs(img), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(img / vmax, -1, 1)

def signed(frames):
    return frames[:, 1] - frames[:, 0]

def plot_gesture(label, examples):
    name = GESTURE_NAMES[label]

    fig, axes = plt.subplots(
        len(examples),
        len(FRAME_IDS),
        figsize=(22, 2.8 * len(examples)),
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

    for row, (frames, source) in enumerate(examples):
        sframes = signed(frames)

        for col, t in enumerate(FRAME_IDS):
            ax = axes[row, col]
            ax.imshow(normalize(sframes[t]), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"t={t}", fontsize=10)
            if col == 0:
                ax.set_ylabel(source, fontsize=8)

    fig.suptitle(f"Geste {label} : {name}", fontsize=20)
    out_path = OUT_DIR / f"gesture_{label:02d}_{name}.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    print("saved:", out_path, flush=True)

def plot_overview(collected):
    labels = sorted(collected.keys())

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

    for row, label in enumerate(labels):
        frames, source = collected[label][0]
        sframes = signed(frames)

        for col, t in enumerate(FRAME_IDS):
            ax = axes[row, col]
            ax.imshow(normalize(sframes[t]), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"t={t}", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"{label} - {GESTURE_NAMES[label]}",
                    fontsize=10,
                    rotation=0,
                    labelpad=55,
                    va="center",
                )

    fig.suptitle("Toutes les classes de gestes primitives DVS-Gesture", fontsize=20)
    out_path = OUT_DIR / "all_gesture_classes_overview.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("saved:", out_path, flush=True)

collected = {i: [] for i in range(11)}
tmp_dir = Path(tempfile.mkdtemp(prefix="dvsgesture_vis_", dir="/tmp"))

try:
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        label_files = sorted([name for name in members if name.endswith("_labels.csv")])

        for label_file in label_files:
            if all(len(collected[i]) >= N_EXAMPLES_PER_GESTURE for i in range(11)):
                break

            base = label_file.replace("_labels.csv", "")
            aedat_file = base + ".aedat"

            if aedat_file not in members:
                continue

            csv_bytes = tar.extractfile(members[label_file]).read()
            csv_data = np.loadtxt(io.BytesIO(csv_bytes), dtype=np.uint32, delimiter=",", skiprows=1)

            needed = False
            for row in csv_data:
                label = int(row[0]) - 1
                if 0 <= label <= 10 and len(collected[label]) < N_EXAMPLES_PER_GESTURE:
                    needed = True
                    break

            if not needed:
                continue

            print("processing:", aedat_file, flush=True)

            tar.extract(members[aedat_file], path=tmp_dir)
            local_aedat = tmp_dir / aedat_file

            events = sjds.load_aedat_v3(str(local_aedat))

            for row in csv_data:
                label = int(row[0]) - 1
                if label < 0 or label > 10:
                    continue
                if len(collected[label]) >= N_EXAMPLES_PER_GESTURE:
                    continue

                start_t = int(row[1])
                end_t = int(row[2])

                frames = events_to_frames(events, start_t, end_t, frames_number=T)
                source = Path(aedat_file).name.replace(".aedat", "")
                collected[label].append((frames, source))

            os.remove(local_aedat)

    for label in range(11):
        print(label, GESTURE_NAMES[label], "examples:", len(collected[label]), flush=True)
        if len(collected[label]) > 0:
            plot_gesture(label, collected[label])

    plot_overview(collected)

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

print("done")
