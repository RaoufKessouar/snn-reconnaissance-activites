from pathlib import Path
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
PRED_CSV = BASE_DIR / "00_predictions" / "predictions_test.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from dvsgc import DVSGestureChain


DATA_ROOT = Path("/users/abdekess61/datasets/SResNet")
T = 60
SEQ_LEN = 4
GESTURES = ["1", "3", "8"]

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


def safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_").replace(";", "_")


def class_to_human(class_name):
    return " -> ".join(
        f"{c}:{GESTURE_NAMES.get(c, 'Unknown')}"
        for c in str(class_name)
    )


def combine_polarities(frames):
    return frames[:, 1] - frames[:, 0]


def normalize_frame(frame):
    vmax = np.percentile(np.abs(frame), 99)
    if vmax <= 0:
        vmax = 1.0
    return np.clip(frame / vmax, -1, 1)


def read_predictions():
    rows = []

    with open(PRED_CSV, "r") as f:
        for row in csv.DictReader(f):
            row["sample_idx"] = int(row["sample_idx"])
            row["true_label"] = int(row["true_label"])
            row["pred_label"] = int(row["pred_label"])
            row["correct"] = int(row["correct"])
            row["loss"] = float(row["loss"])
            row["confidence"] = float(row["confidence"])
            row["top2_margin"] = float(row["top2_margin"])
            row["entropy"] = float(row["entropy"])
            row["event_count"] = float(row["event_count"])

            row["true_seq"] = str(row["true_class_name"])
            row["pred_seq"] = str(row["pred_class_name"])

            rows.append(row)

    return rows


def position_stats(rows):
    stats = []

    for pos in range(SEQ_LEN):
        total = len(rows)
        correct = sum(r["true_seq"][pos] == r["pred_seq"][pos] for r in rows)
        acc = correct / total

        stats.append({
            "position": pos + 1,
            "total": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": acc,
        })

    return stats


def gesture_confusion(rows):
    matrix = {
        true_g: {pred_g: 0 for pred_g in GESTURES}
        for true_g in GESTURES
    }

    by_position = {
        pos + 1: {
            true_g: {pred_g: 0 for pred_g in GESTURES}
            for true_g in GESTURES
        }
        for pos in range(SEQ_LEN)
    }

    for r in rows:
        for pos in range(SEQ_LEN):
            true_g = r["true_seq"][pos]
            pred_g = r["pred_seq"][pos]

            matrix[true_g][pred_g] += 1
            by_position[pos + 1][true_g][pred_g] += 1

    return matrix, by_position


def save_position_accuracy(stats):
    out_csv = OUT_DIR / "position_accuracy.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["position", "total", "correct", "wrong", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(stats)

    positions = [s["position"] for s in stats]
    accuracies = [s["accuracy"] for s in stats]

    plt.figure(figsize=(8, 5))
    plt.bar(positions, accuracies)
    plt.xticks(positions, [f"Position {p}" for p in positions])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy par position dans la chaine")
    plt.grid(axis="y", alpha=0.3)

    for p, acc in zip(positions, accuracies):
        plt.text(p, acc + 0.015, f"{acc:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "position_accuracy.png", dpi=180)
    plt.close()

    print(f"saved: {out_csv}")
    print(f"saved: {OUT_DIR / 'position_accuracy.png'}")


def save_gesture_confusions(matrix, by_position):
    out_csv = OUT_DIR / "gesture_confusion_matrix.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_gesture", "pred_1", "pred_3", "pred_8", "total", "accuracy"])

        for true_g in GESTURES:
            total = sum(matrix[true_g].values())
            correct = matrix[true_g][true_g]
            acc = correct / total if total > 0 else 0.0
            writer.writerow([
                true_g,
                matrix[true_g]["1"],
                matrix[true_g]["3"],
                matrix[true_g]["8"],
                total,
                acc,
            ])

    mat = np.array([[matrix[t][p] for p in GESTURES] for t in GESTURES], dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = mat / np.maximum(row_sums, 1)

    plt.figure(figsize=(7, 6))
    plt.imshow(mat_norm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Proportion")
    plt.xticks(
        range(len(GESTURES)),
        [f"{g}:{GESTURE_NAMES[g]}" for g in GESTURES],
        rotation=30,
        ha="right",
    )
    plt.yticks(range(len(GESTURES)), [f"{g}:{GESTURE_NAMES[g]}" for g in GESTURES])
    plt.xlabel("Geste predit")
    plt.ylabel("Geste vrai")
    plt.title("Matrice de confusion des gestes individuels")

    for i, true_g in enumerate(GESTURES):
        for j, pred_g in enumerate(GESTURES):
            plt.text(
                j,
                i,
                f"{int(mat[i, j])}\n{mat_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "gesture_confusion_matrix.png", dpi=180)
    plt.close()

    with open(OUT_DIR / "gesture_confusion_by_position.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "true_gesture", "pred_gesture", "count"])

        for pos in range(1, SEQ_LEN + 1):
            for true_g in GESTURES:
                for pred_g in GESTURES:
                    writer.writerow([pos, true_g, pred_g, by_position[pos][true_g][pred_g]])

    confusions = []
    for true_g in GESTURES:
        for pred_g in GESTURES:
            if true_g != pred_g:
                confusions.append((matrix[true_g][pred_g], true_g, pred_g))

    confusions.sort(reverse=True)

    with open(OUT_DIR / "top_gesture_confusions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "true_gesture", "true_name", "pred_gesture", "pred_name"])

        for count, true_g, pred_g in confusions:
            writer.writerow([
                count,
                true_g,
                GESTURE_NAMES[true_g],
                pred_g,
                GESTURE_NAMES[pred_g],
            ])

    print(f"saved: {out_csv}")
    print(f"saved: {OUT_DIR / 'gesture_confusion_matrix.png'}")
    print(f"saved: {OUT_DIR / 'gesture_confusion_by_position.csv'}")
    print(f"saved: {OUT_DIR / 'top_gesture_confusions.csv'}")


def plot_case(test_set, row, out_dir, reason):
    sample_idx = row["sample_idx"]
    frames, _ = test_set[sample_idx]
    frames = np.asarray(frames)

    frame_ids = np.linspace(0, T - 1, 12, dtype=int).tolist()
    combined = combine_polarities(frames)

    fig, axes = plt.subplots(
        2,
        6,
        figsize=(22, 11),
        gridspec_kw={
            "left": 0.02,
            "right": 0.98,
            "bottom": 0.03,
            "top": 0.76,
            "wspace": 0.03,
            "hspace": 0.18,
        },
    )

    axes = axes.reshape(-1)

    for ax, t in zip(axes, frame_ids):
        ax.imshow(
            normalize_frame(combined[t]),
            cmap="bwr",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(f"t={t}", fontsize=14, pad=2)
        ax.axis("off")

    true_seq = row["true_seq"]
    pred_seq = row["pred_seq"]

    pos_details = []
    for i in range(SEQ_LEN):
        mark = "OK" if true_seq[i] == pred_seq[i] else "ERR"
        pos_details.append(f"P{i+1}:{true_seq[i]}->{pred_seq[i]}({mark})")

    top5_names = row["top5_class_names"].split(";")
    top5_probs = row["top5_probs"].split(";")
    top5_text = " | ".join([f"{n}:{p}" for n, p in zip(top5_names, top5_probs)])

    title = (
        f"{reason} | sample={sample_idx} | true={true_seq} | pred={pred_seq}\n"
        f"true: {class_to_human(true_seq)}\n"
        f"pred: {class_to_human(pred_seq)}\n"
        f"{' | '.join(pos_details)}\n"
        f"confidence={row['confidence']:.4f} | loss={row['loss']:.4f} | "
        f"top2_margin={row['top2_margin']:.4f} | entropy={row['entropy']:.4f} | "
        f"events={row['event_count']:.0f}\n"
        f"top5: {top5_text}"
    )

    fig.suptitle(title, fontsize=14, y=0.97)
    plt.savefig(out_dir / "chain_overview.png", dpi=180)
    plt.close(fig)


def select_diverse(rows, n=8):
    rows = sorted(rows, key=lambda r: (r["confidence"], r["loss"]), reverse=True)

    selected = []
    used_pairs = set()

    for r in rows:
        pair = (r["true_seq"], r["pred_seq"])
        if pair not in used_pairs:
            selected.append(r)
            used_pairs.add(pair)
        if len(selected) == n:
            return selected

    for r in rows:
        if r not in selected:
            selected.append(r)
        if len(selected) == n:
            break

    return selected


def save_examples_by_position(test_set, rows):
    base = OUT_DIR / "examples_by_position"
    base.mkdir(exist_ok=True)

    manifest = []

    for pos in range(SEQ_LEN):
        wrong_rows = [
            r for r in rows
            if r["true_seq"][pos] != r["pred_seq"][pos]
        ]

        selected = select_diverse(wrong_rows, n=8)
        pos_dir = base / f"position_{pos + 1}_errors"
        pos_dir.mkdir(exist_ok=True)

        for r in selected:
            true_g = r["true_seq"][pos]
            pred_g = r["pred_seq"][pos]

            case_dir = pos_dir / (
                f"sample_{r['sample_idx']:05d}_"
                f"true_{r['true_seq']}_pred_{r['pred_seq']}_"
                f"p{pos+1}_{true_g}_to_{pred_g}"
            )
            case_dir.mkdir(exist_ok=True)

            reason = (
                f"POSITION {pos + 1} ERROR | "
                f"{true_g}:{GESTURE_NAMES[true_g]} -> {pred_g}:{GESTURE_NAMES[pred_g]}"
            )
            plot_case(test_set, r, case_dir, reason)

            manifest.append({
                "position": pos + 1,
                "sample_idx": r["sample_idx"],
                "true_class": r["true_seq"],
                "pred_class": r["pred_seq"],
                "true_gesture": true_g,
                "pred_gesture": pred_g,
                "confidence": r["confidence"],
                "loss": r["loss"],
                "top2_margin": r["top2_margin"],
                "entropy": r["entropy"],
                "event_count": r["event_count"],
                "case_dir": str(case_dir),
            })

            print(f"saved position example: {case_dir}")

    with open(base / "position_examples_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"saved: {base / 'position_examples_manifest.csv'}")


def save_examples_by_gesture_confusion(test_set, rows):
    base = OUT_DIR / "examples_by_gesture_confusion"
    base.mkdir(exist_ok=True)

    grouped = defaultdict(list)

    for r in rows:
        for pos in range(SEQ_LEN):
            true_g = r["true_seq"][pos]
            pred_g = r["pred_seq"][pos]

            if true_g != pred_g:
                grouped[(true_g, pred_g)].append((pos + 1, r))

    manifest = []

    for (true_g, pred_g), items in sorted(grouped.items()):
        pair_rows = []
        seen = set()

        for pos, r in items:
            key = r["sample_idx"]
            if key not in seen:
                copied = dict(r)
                copied["error_position_for_pair"] = pos
                pair_rows.append(copied)
                seen.add(key)

        selected = select_diverse(pair_rows, n=6)

        pair_dir = base / (
            f"true_{true_g}_{safe_name(GESTURE_NAMES[true_g])}_"
            f"pred_{pred_g}_{safe_name(GESTURE_NAMES[pred_g])}"
        )
        pair_dir.mkdir(exist_ok=True)

        for r in selected:
            pos = r["error_position_for_pair"]

            case_dir = pair_dir / (
                f"sample_{r['sample_idx']:05d}_"
                f"true_{r['true_seq']}_pred_{r['pred_seq']}_"
                f"p{pos}"
            )
            case_dir.mkdir(exist_ok=True)

            reason = (
                f"GESTURE CONFUSION | P{pos} | "
                f"{true_g}:{GESTURE_NAMES[true_g]} -> {pred_g}:{GESTURE_NAMES[pred_g]}"
            )
            plot_case(test_set, r, case_dir, reason)

            manifest.append({
                "true_gesture": true_g,
                "true_name": GESTURE_NAMES[true_g],
                "pred_gesture": pred_g,
                "pred_name": GESTURE_NAMES[pred_g],
                "position": pos,
                "sample_idx": r["sample_idx"],
                "true_class": r["true_seq"],
                "pred_class": r["pred_seq"],
                "confidence": r["confidence"],
                "loss": r["loss"],
                "top2_margin": r["top2_margin"],
                "entropy": r["entropy"],
                "event_count": r["event_count"],
                "case_dir": str(case_dir),
            })

            print(f"saved gesture-confusion example: {case_dir}")

    with open(base / "gesture_confusion_examples_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"saved: {base / 'gesture_confusion_examples_manifest.csv'}")


def main():
    print(f"reading: {PRED_CSV}")
    rows = read_predictions()
    print(f"rows: {len(rows)}")

    stats = position_stats(rows)
    matrix, by_position = gesture_confusion(rows)

    save_position_accuracy(stats)
    save_gesture_confusions(matrix, by_position)

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

    save_examples_by_position(test_set, rows)
    save_examples_by_gesture_confusion(test_set, rows)

    print("done")


if __name__ == "__main__":
    main()
