import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("/users/abdekess61/raouf/SResNet/experiments/mad/04_train_chain/train_chain_tdbn.log")
OUT = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
VAL = "#c0392b"    # validation = rouge
TRN = "#2f6fb0"    # entrainement = bleu

PAT = re.compile(r"Epoch : (\d+).*?Train Accuracy : ([\d.]+).*?Val\(std\) Accuracy : ([\d.]+)")
ep, tr, val = [], [], []
for ln in open(LOG):
    m = PAT.search(ln)
    if m:
        ep.append(int(m.group(1))); tr.append(float(m.group(2))); val.append(float(m.group(3)))
print("epochs lus:", len(ep))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

# ----- GAUCHE : validation -----
axL.plot(ep, val, color=VAL, lw=1.8, label="validation")
axL.set_xlabel("epoch"); axL.set_ylabel("Exactitude"); axL.set_ylim(0, 1)
axL.grid(alpha=.25); axL.legend(frameon=False, loc="lower right")

# ----- DROITE : entrainement -----
axR.plot(ep, tr, color=TRN, lw=1.8, label="entraînement")
axR.set_xlabel("epoch"); axR.set_ylim(0, 1)
axR.grid(alpha=.25); axR.legend(frameon=False, loc="lower right")

for ax in (axL, axR):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
out = OUT/"figure_courbes_tdbn.png"
plt.savefig(out, dpi=200, facecolor="white"); plt.close()
print("saved:", out)
