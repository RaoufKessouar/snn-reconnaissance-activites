from pathlib import Path
import re
import csv
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path("/data/abdekess61/datasets/lab_gesture_dataset/raw")
OUT_ROOT = Path("/users/abdekess61/raouf/SResNet/lab_gesture_dataset_visualization")
OUT = OUT_ROOT / "outputs"
PER_ACTION = OUT / "per_action"
OVERVIEWS = OUT / "overviews"
META = OUT_ROOT / "metadata"

for d in [PER_ACTION, OVERVIEWS, META]:
    d.mkdir(parents=True, exist_ok=True)

W, H = 640, 480
N_FRAMES = 12
N_SAMPLES_PER_ACTION = 6
PAD = 25

PATTERN = re.compile(r"A(?P<action>\d+)P(?P<participant>\d+)R(?P<rep>\d+)S(?P<session>\d+)D(?P<domain>\d+)\.csv")


def parse_file(path):
    m = PATTERN.match(path.name)
    if not m:
        return None
    info = m.groupdict()
    info = {k: int(v) for k, v in info.items()}
    info["action_code"] = f"A{info['action']:03d}"
    return info


def load_labels():
    label_file = META / "action_labels_template.csv"
    labels = {}
    if label_file.exists():
        with open(label_file, "r", newline="") as f:
            for row in csv.DictReader(f):
                code = row["action_code"]
                name = row.get("action_name", "").strip()
                labels[code] = name
    return labels


def read_events(path):
    df = pd.read_csv(path, header=None, names=["x", "y", "p", "t"])
    df = df.sort_values("t")
    return df


def crop_bounds(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    x0 = max(0, int(np.percentile(x, 1)) - PAD)
    x1 = min(W, int(np.percentile(x, 99)) + PAD)
    y0 = max(0, int(np.percentile(y, 1)) - PAD)
    y1 = min(H, int(np.percentile(y, 99)) + PAD)

    if x1 <= x0 or y1 <= y0:
        return 0, W, 0, H

    return x0, x1, y0, y1


def events_to_frames(df, n_frames=N_FRAMES):
    x = df["x"].to_numpy(dtype=np.int64)
    y = df["y"].to_numpy(dtype=np.int64)
    p = df["p"].to_numpy(dtype=np.int64)

    frames = np.zeros((n_frames, 2, H, W), dtype=np.float32)

    if len(df) == 0:
        return frames

    borders = np.linspace(0, len(df), n_frames + 1, dtype=np.int64)

    for i in range(n_frames):
        s, e = borders[i], borders[i + 1]
        xs = x[s:e]
        ys = y[s:e]
        ps = p[s:e]

        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H) & ((ps == 0) | (ps == 1))
        np.add.at(frames[i], (ps[valid], ys[valid], xs[valid]), 1)

    return frames


def normalize(img):
    vmax = np.percentile(np.abs(img), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(img / vmax, -1, 1)


def select_diverse(files, n=N_SAMPLES_PER_ACTION):
    parsed = [(f, parse_file(f)) for f in files]
    parsed = [(f, info) for f, info in parsed if info is not None]
    parsed.sort(key=lambda x: (x[1]["participant"], x[1]["rep"]))

    participants = sorted(set(info["participant"] for _, info in parsed))
    if len(participants) == 0:
        return []

    target_ids = np.linspace(0, len(participants) - 1, min(n, len(participants)), dtype=int)
    target_participants = [participants[i] for i in target_ids]

    selected = []
    used = set()

    for k, p in enumerate(target_participants):
        candidates = [(f, info) for f, info in parsed if info["participant"] == p]
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1]["rep"])
        choice = candidates[k % len(candidates)][0]
        selected.append(choice)
        used.add(choice)

    for f, info in parsed:
        if len(selected) >= n:
            break
        if f not in used:
            selected.append(f)
            used.add(f)

    return selected


def build_manifest(files):
    rows = []
    for f in files:
        info = parse_file(f)
        if info is None:
            continue
        rows.append({
            "path": str(f),
            "file": f.name,
            "action_code": info["action_code"],
            "participant": f"P{info['participant']:03d}",
            "repetition": f"R{info['rep']}",
            "session": f"S{info['session']}",
            "domain": f"D{info['domain']}",
        })

    manifest = META / "manifest.csv"
    with open(manifest, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    actions = sorted(set(r["action_code"] for r in rows))
    label_template = META / "action_labels_template.csv"
    with open(label_template, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["action_code", "action_name"])
        for code in actions:
            writer.writerow([code, ""])

    print("saved:", manifest)
    print("saved:", label_template)

    return rows, actions


def plot_action(action_code, files, labels):
    selected = select_diverse(files)
    action_name = labels.get(action_code, "")
    display_name = action_name if action_name else "nom_geste_a_completer"

    fig, axes = plt.subplots(
        len(selected),
        N_FRAMES,
        figsize=(22, 2.7 * len(selected)),
        gridspec_kw={
            "left": 0.035,
            "right": 0.985,
            "bottom": 0.04,
            "top": 0.86,
            "wspace": 0.02,
            "hspace": 0.22,
        },
    )

    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row, path in enumerate(selected):
        info = parse_file(path)
        df = read_events(path)
        frames = events_to_frames(df)
        signed = frames[:, 1] - frames[:, 0]

        x0, x1, y0, y1 = crop_bounds(df)

        for col in range(N_FRAMES):
            ax = axes[row, col]
            img = signed[col, y0:y1, x0:x1]
            ax.imshow(normalize(img), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"f{col}", fontsize=9)

            if col == 0:
                ax.set_ylabel(
                    f"P{info['participant']:03d} R{info['rep']} | {len(df)} ev.",
                    fontsize=8,
                )

    fig.suptitle(
        f"Classe {action_code} | {display_name} | exemples diversifies par participant",
        fontsize=16,
    )

    out_path = PER_ACTION / f"{action_code}_{display_name}.png"
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print("saved:", out_path)


def plot_overviews(actions, files_by_action, labels):
    per_page = 10
    pages = math.ceil(len(actions) / per_page)

    for page in range(pages):
        subset = actions[page * per_page:(page + 1) * per_page]

        fig, axes = plt.subplots(
            len(subset),
            N_FRAMES,
            figsize=(22, 2.25 * len(subset)),
            gridspec_kw={
                "left": 0.095,
                "right": 0.985,
                "bottom": 0.04,
                "top": 0.94,
                "wspace": 0.02,
                "hspace": 0.20,
            },
        )

        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for row, action_code in enumerate(subset):
            sample = select_diverse(files_by_action[action_code], n=1)[0]
            info = parse_file(sample)
            df = read_events(sample)
            frames = events_to_frames(df)
            signed = frames[:, 1] - frames[:, 0]
            x0, x1, y0, y1 = crop_bounds(df)

            name = labels.get(action_code, "")
            label_text = f"{action_code}" if not name else f"{action_code} - {name}"

            for col in range(N_FRAMES):
                ax = axes[row, col]
                img = signed[col, y0:y1, x0:x1]
                ax.imshow(normalize(img), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
                ax.axis("off")

                if row == 0:
                    ax.set_title(f"f{col}", fontsize=9)

                if col == 0:
                    ax.set_ylabel(
                        label_text,
                        fontsize=9,
                        rotation=0,
                        labelpad=65,
                        va="center",
                    )

        fig.suptitle(f"Overview des classes de gestes - page {page + 1}/{pages}", fontsize=18)
        out_path = OVERVIEWS / f"overview_actions_page_{page + 1:02d}.png"
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
        print("saved:", out_path)


def main():
    files = sorted(RAW.glob("*/*.csv"))
    print("csv files:", len(files))

    rows, actions = build_manifest(files)
    labels = load_labels()

    files_by_action = {}
    for f in files:
        info = parse_file(f)
        if info is None:
            continue
        files_by_action.setdefault(info["action_code"], []).append(f)

    actions = sorted(files_by_action.keys())
    print("actions:", actions)
    print("num actions:", len(actions))

    for action_code in actions:
        plot_action(action_code, files_by_action[action_code], labels)

    plot_overviews(actions, files_by_action, labels)

    print("done")


if __name__ == "__main__":
    main()
