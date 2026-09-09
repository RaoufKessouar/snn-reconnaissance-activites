import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_io import load_events, to_frames
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

PART, SUB, REP, NCOL = 1, 1, 1, 6
fig, axes = plt.subplots(9, NCOL, figsize=(NCOL*2.0, 9*1.7))
for r, a in enumerate(range(1, 10)):
    m = list(EXTRACT_DIR.rglob(f"A{a:02d}{SUB}P{PART:03d}R{REP}*.csv"))
    if not m:
        for cc in range(NCOL): axes[r,cc].axis("off"); continue
    x,y,p,t = load_events(m[0]); fr = to_frames(x,y,p,t,T=NCOL,ds=2)
    comb = fr[:,1]-fr[:,0]; vmax = np.percentile(np.abs(comb),99) or 1
    for cc in range(NCOL):
        axes[r,cc].imshow(np.clip(comb[cc]/vmax,-1,1),cmap="bwr",vmin=-1,vmax=1)
        axes[r,cc].set_xticks([]); axes[r,cc].set_yticks([])
        if r==0: axes[r,cc].set_title(f"t{cc+1}", fontsize=9)
    axes[r,0].set_ylabel(f"{a}:{ACTIVITY_NAMES[a]}", fontsize=8)
plt.suptitle(f"Planche A01-A09 (P{PART:03d}, sub{SUB}, rep{REP})")
plt.tight_layout(); plt.savefig(FIG/"04_montage_activities.png", dpi=140); plt.close()
print("OK -> 04_montage_activities.png")
