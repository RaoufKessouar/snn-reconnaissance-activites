import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/users/abdekess61/raouf/SResNet/experiments/overlap_078_seq3_T80/error_analysis")
OUT  = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
MARINE = "#043353"
NAMES = {"0": "Hand Clapping", "7": "Arm Roll", "8": "Air Drums"}

# ---- Figure 14 : exactitude par position ----
pa = pd.read_csv(BASE/"position_accuracy.csv")
acc = pa["accuracy"].astype(float).tolist()
fig, ax = plt.subplots(figsize=(5.2, 4))
ax.bar(range(len(acc)), acc, color=MARINE, width=0.55)
for i, a in enumerate(acc):
    ax.text(i, a + 0.012, f"{a:.3f}", ha="center", color=MARINE, fontsize=11, fontweight="bold")
ax.set_xticks(range(len(acc))); ax.set_xticklabels([f"Position {i+1}" for i in range(len(acc))])
ax.set_ylim(0, 1); ax.set_ylabel("Exactitude")
ax.grid(axis="y", alpha=.2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT/"figure14_accuracy_position.png", dpi=200, facecolor="white"); plt.close()
print("saved figure14  |  positions:", len(acc), acc)

# ---- Figure 15 : matrice de confusion des primitives ----
cm = pd.read_csv(BASE/"gesture_confusion_matrix.csv", index_col=0)
cm = cm.drop(columns=[c for c in cm.columns if str(c).lower() == "acc"])
M = cm.values.astype(float)
rows = [NAMES.get(str(r), str(r)) for r in cm.index]
cols = [NAMES.get(str(c), str(c)) for c in cm.columns]
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(M, cmap="Blues")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20, ha="right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
vmax = M.max() if M.max() > 0 else 1
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{int(v)}", ha="center", va="center",
                color="white" if v > vmax*0.5 else MARINE, fontsize=11)
ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(OUT/"figure15_confusion_primitives.png", dpi=200, facecolor="white"); plt.close()
print("saved figure15  |  shape:", M.shape)
