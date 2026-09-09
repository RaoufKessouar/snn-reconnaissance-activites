import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_io import load_events
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

PART, SUB, REP = 1, 1, 1
pairs = [(2,3),(4,5)]
fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 5*len(pairs)))
for r,(a1,a2) in enumerate(pairs):
    for cc,a in enumerate((a1,a2)):
        ax = axes[r,cc]
        m = list(EXTRACT_DIR.rglob(f"A{a:02d}{SUB}P{PART:03d}R{REP}*.csv"))
        if not m: ax.axis("off"); continue
        x,y,p,t = load_events(m[0]); s = slice(None,None,max(1,len(t)//120000))
        ax.scatter((t[s]-t.min())/1e6, y[s], c=p[s], cmap="coolwarm", s=0.3, alpha=0.4)
        ax.invert_yaxis(); ax.set_title(f"{a}:{ACTIVITY_NAMES[a]}")
        ax.set_xlabel("t (s)"); ax.set_ylabel("y (haut->bas)")
plt.suptitle("Paires inversees en projection y-t")
plt.tight_layout(); plt.savefig(FIG/"05_reversed_pairs_yt.png", dpi=150); plt.close()
print("OK -> 05_reversed_pairs_yt.png")
