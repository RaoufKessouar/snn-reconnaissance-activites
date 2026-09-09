import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["pdf.fonttype"] = 42   # polices embarquees (texte net, vectoriel)
plt.rcParams["ps.fonttype"]  = 42

BASE = Path("/users/abdekess61/raouf/SResNet/experiments/mad/06_fall_detect")
OUT  = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
CSV  = BASE/"metrics_T50_fb2.0_kcurve.csv"
MARINE="#043353"; RED="#c0392b"; BLUE="#2f6fb0"; KS=[1,2,3,4,5]

def save(fig, name):
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")              # vectoriel
    fig.savefig(OUT/f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved:", OUT/f"{name}.pdf")

df = pd.read_csv(CSV)
b = df.loc[df["f1_k3"].idxmax()]
print("best epoch:", int(b["epoch"]), "| F1@K3 =", round(float(b["f1_k3"]),4))

# ---- FIG 1 : courbe de fonctionnement (sensibilite vs fausses alarmes) ----
rec = [float(b[f"recall_k{k}"]) for k in KS]
far = [float(b[f"far_k{k}"])    for k in KS]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.plot(far, rec, "-o", color=MARINE, lw=1.6, ms=6)
for k, x, y in zip(KS, far, rec):
    ax.annotate(f"K={k}", (x, y), textcoords="offset points", xytext=(7, -3),
                fontsize=9.5, color=MARINE)
i3 = KS.index(3)
ax.plot(far[i3], rec[i3], "o", ms=13, mfc="none", mec=RED, mew=2.2)
ax.set_xlabel("Taux de fausses alarmes"); ax.set_ylabel("Sensibilité (rappel)")
ax.set_xlim(-0.02, 0.52); ax.set_ylim(0.82, 1.02); ax.grid(alpha=.25)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); save(fig, "fig_chute_operating")

# ---- FIG 2 : validation par epoch (exactitude image + F1@K3) ----
ep = df["epoch"].values
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(ep, df["frameacc"], color=BLUE, lw=1.8, label="exactitude par image")
ax.plot(ep, df["f1_k3"],    color=RED,  lw=1.5, label="F1 de détection (K=3)")
ax.set_xlabel("epoch"); ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.grid(alpha=.25)
ax.legend(frameon=False, loc="lower right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); save(fig, "fig_chute_validation")
