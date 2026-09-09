from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
PRED_CSV = BASE_DIR / "00_predictions" / "predictions_test.csv"
OUT_DIR = Path(__file__).resolve().parent

true_labels = []
pred_labels = []
true_names = []

with open(PRED_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        true_labels.append(int(row["true_label"]))
        pred_labels.append(int(row["pred_label"]))
        true_names.append(row["true_class_name"])

num_classes = max(max(true_labels), max(pred_labels)) + 1

confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

for t, p in zip(true_labels, pred_labels):
    confusion[t, p] += 1

class_names = [""] * num_classes
with open(PRED_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names[int(row["true_label"])] = row["true_class_name"]

row_sums = confusion.sum(axis=1, keepdims=True)
confusion_norm = confusion / np.maximum(row_sums, 1)

acc_per_class = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)

# Figure 1: matrice normalisee
plt.figure(figsize=(18, 16))
plt.imshow(confusion_norm, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Proportion")
plt.title("Matrice de confusion normalisee - Test set")
plt.xlabel("Classe predite")
plt.ylabel("Classe vraie")
plt.xticks(range(num_classes), class_names, rotation=90, fontsize=7)
plt.yticks(range(num_classes), class_names, fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_normalized.png", dpi=200)
plt.close()

# Figure 2: accuracy par classe
order = np.argsort(acc_per_class)

plt.figure(figsize=(18, 7))
plt.bar(range(num_classes), acc_per_class[order])
plt.xticks(range(num_classes), [class_names[i] for i in order], rotation=90, fontsize=8)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy par classe - triee de la plus faible a la plus forte")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_accuracy_sorted.png", dpi=200)
plt.close()

# Sauvegarde CSV par classe
with open(OUT_DIR / "per_class_accuracy.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "class_name", "support", "correct", "accuracy"])

    for i in range(num_classes):
        support = int(confusion[i].sum())
        correct = int(confusion[i, i])
        acc = float(acc_per_class[i])
        writer.writerow([i, class_names[i], support, correct, acc])

# Top confusions hors diagonale
confusions = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and confusion[i, j] > 0:
            confusions.append((int(confusion[i, j]), i, j, class_names[i], class_names[j]))

confusions.sort(reverse=True)

with open(OUT_DIR / "top_confusions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["count", "true_label", "pred_label", "true_class", "pred_class"])

    for count, true_label, pred_label, true_class, pred_class in confusions[:50]:
        writer.writerow([count, true_label, pred_label, true_class, pred_class])

print(f"samples: {len(true_labels)}")
print(f"num_classes: {num_classes}")
print(f"global accuracy: {np.mean(np.array(true_labels) == np.array(pred_labels)):.4f}")
print(f"saved: {OUT_DIR / 'confusion_matrix_normalized.png'}")
print(f"saved: {OUT_DIR / 'per_class_accuracy_sorted.png'}")
print(f"saved: {OUT_DIR / 'per_class_accuracy.csv'}")
print(f"saved: {OUT_DIR / 'top_confusions.csv'}")
