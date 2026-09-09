import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_dataset import MADDataset, active_window
from mad_io import load_events
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR, CACHE_DIR
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

ds = MADDataset(EXTRACT_DIR, CACHE_DIR, participants=[1,2,3], T=40, W_s=3.0, H=128, W=128)
print(f"tag preprocessing : {ds.tag}")

def find(a):
    m = list(Path(EXTRACT_DIR).rglob(f"A{a:02d}1P001R1*.csv")); return m[0] if m else None

# --- montage 9 activites (6 frames sur 40) + fenetre active detectee ---
NCOL = 6
fig, axes = plt.subplots(9, NCOL, figsize=(NCOL*2.0, 9*1.7))
for r, a in enumerate(range(1, 10)):
    f = find(a)
    if f is None:
        for cc in range(NCOL): axes[r,cc].axis("off"); continue
    x,y,p,t = load_events(f); ws,we = active_window(t, ds.W_us)
    print(f"{a}:{ACTIVITY_NAMES[a]:16s} fenetre active = [{(ws-t.min())/1e6:.2f}, {(we-t.min())/1e6:.2f}] s")
    fr = ds.frames_for(f); comb = fr[:,1]-fr[:,0]
    vmax = np.percentile(np.abs(comb),99) or 1
    for cc, fidx in enumerate(np.linspace(0, fr.shape[0]-1, NCOL).astype(int)):
        axes[r,cc].imshow(np.clip(comb[fidx]/vmax,-1,1), cmap="bwr", vmin=-1, vmax=1)
        axes[r,cc].set_xticks([]); axes[r,cc].set_yticks([])
        if r==0: axes[r,cc].set_title(f"f{fidx}", fontsize=8)
    axes[r,0].set_ylabel(f"{a}:{ACTIVITY_NAMES[a]}", fontsize=8)
plt.suptitle("MAD preprocessed : crop+downsample 128, fenetre active 3s, T=40 (6 frames /40)")
plt.tight_layout(); plt.savefig(FIG/"preproc_montage_A01-A09.png", dpi=140); plt.close()

# --- paires inversees preprocessed (2v3, 4v5) ---
pairs = [(2,3),(4,5)]
fig, axes = plt.subplots(2, 12, figsize=(24, 2*3.2))
for r,(a1,a2) in enumerate(pairs):
    for k,a in enumerate((a1,a2)):
        f = find(a); fr = ds.frames_for(f); comb = fr[:,1]-fr[:,0]
        vmax = np.percentile(np.abs(comb),99) or 1
        for cc, fidx in enumerate(np.linspace(0, fr.shape[0]-1, 6).astype(int)):
            ax = axes[r, k*6+cc]
            ax.imshow(np.clip(comb[fidx]/vmax,-1,1), cmap="bwr", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if cc==0: ax.set_ylabel(f"{a}:{ACTIVITY_NAMES[a]}", fontsize=8)
plt.suptitle("Paires inversees preprocessed (gauche vs droite = ordre temporel oppose)")
plt.tight_layout(); plt.savefig(FIG/"preproc_reversed_pairs.png", dpi=140); plt.close()
print("figures ->", FIG)
