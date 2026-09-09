import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MAD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAD_ROOT/"common"))
from mad_io import load_events
from mad_dataset import active_window, make_frames
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR

crop=(120,520,80,440); H=W=128; W_us=3_000_000
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for k, a in enumerate(range(1, 10)):
    ax = axes[k//3, k%3]
    m = list(Path(EXTRACT_DIR).rglob(f"A{a:02d}1P001R1*.csv"))
    if not m: ax.axis("off"); continue
    x,y,p,t = load_events(m[0]); ws,we = active_window(t, W_us)
    fr = make_frames(x,y,p,t,ws,we,12,H,W,crop)
    agg = (fr[:,1]-fr[:,0]).sum(0); vmax = np.percentile(np.abs(agg),99) or 1
    ax.imshow(np.clip(agg/vmax,-1,1), cmap="bwr", vmin=-1, vmax=1)
    ax.set_title(f"{a}: {ACTIVITY_NAMES[a]}", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("MAD — les 9 activités (caméra événementielle)", fontsize=13)
plt.tight_layout(); plt.savefig(MAD_ROOT/"04_train_chain"/"mad_activities_overview.png", dpi=160, bbox_inches="tight")
print("OK -> mad_activities_overview.png")
