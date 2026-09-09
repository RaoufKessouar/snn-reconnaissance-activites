import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/users/abdekess61/raouf/SResNet/experiments/mad/common")
from mad_io import load_events
from mad_naming import parse
from mad_dataset import active_window, make_frames
from mad_paths import EXTRACT_DIR

OUT = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
NF = 6                                  # frames par ligne
W_S = 3.0; H = W = 128; CROP = (120, 520, 80, 440); W_US = int(W_S * 1e6)
TARGET = {1: "Marcher", 3: "S'asseoir", 9: "Chuter"}   # primitives du chainage
# PARTICIPANT = 5   # decommente pour forcer un participant precis si l'exemple auto n'est pas net

chosen = {}
for c in sorted(Path(EXTRACT_DIR).rglob("*.csv")):
    d = parse(c.name)
    if not d or d.get("sensor") != 1:
        continue
    # if 'PARTICIPANT' in globals() and d["participant"] != PARTICIPANT: continue
    a = d["activity"]
    if a in TARGET and a not in chosen:
        chosen[a] = c
    if all(k in chosen for k in TARGET):
        break
print("clips retenus:", {a: chosen[a].name for a in chosen})

def signed_frames(csv):
    x, y, p, t = load_events(str(csv))
    ws, we = active_window(t, W_US)
    fr = make_frames(x, y, p, t, ws, we, NF, H, W, CROP)   # [NF, 2, H, W]
    return fr[:, 1] - fr[:, 0]

def norm(img):
    vmax = np.percentile(np.abs(img), 99); vmax = vmax if vmax > 0 else 1.0
    return np.clip(img / vmax, -1, 1)

acts = [a for a in (1, 3, 9) if a in chosen]
fig, axes = plt.subplots(len(acts), NF, figsize=(2.1*NF, 2.4*len(acts)),
    gridspec_kw={"left":0.10,"right":0.99,"bottom":0.04,"top":0.94,"wspace":0.05,"hspace":0.22})
axes = np.atleast_2d(axes)
for row, a in enumerate(acts):
    sf = signed_frames(chosen[a])
    for col in range(NF):
        ax = axes[row, col]
        ax.imshow(norm(sf[col]), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"t = {col}", fontsize=11)
        if col == 0: ax.set_ylabel(TARGET[a], fontsize=12)
out = OUT/"figure_primitives_mad.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
print("saved:", out)
