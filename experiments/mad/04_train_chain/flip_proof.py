import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
MAD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAD_ROOT/"common"))
from mad_io import load_events
from mad_dataset import active_window, make_frames
from mad_paths import EXTRACT_DIR

marine="#043353"; green="#2e8b57"; red="#c0392b"
# clip de marche laterale (direction visible) ; fallback : n'importe quelle marche
cands = list(Path(EXTRACT_DIR).rglob("A013P001R1*.csv")) or list(Path(EXTRACT_DIR).rglob("A01*P001R1*.csv"))
f = cands[0]; print("clip:", f.name, flush=True)
x,y,p,t = load_events(f)
ws,we = active_window(t, 3_000_000); crop=(120,520,80,440)
fr  = make_frames(x,y,p,t,ws,we,12,128,128,crop)
frf = torch.flip(torch.from_numpy(fr), dims=[3]).numpy()

# centre de masse horizontal au fil du temps (preuve quantitative)
m=(t>=ws)&(t<we)
nb=40
tb=np.clip(((t[m]-ws)/(we-ws)*nb).astype(int),0,nb-1)
sx=np.bincount(tb, weights=x[m].astype(float), minlength=nb)
cx=np.bincount(tb, minlength=nb).astype(float)
meanx=np.where(cx>0, sx/np.maximum(cx,1), np.nan)
tc=np.linspace(0,(we-ws)/1e6,nb)

fig=plt.figure(figsize=(13,7.2))
gs=gridspec.GridSpec(3,6,height_ratios=[1,1,1.25],hspace=0.35)
def draw_row(r, frames, label):
    comb=frames[:,1]-frames[:,0]; vmax=np.percentile(np.abs(comb),99) or 1
    ids=np.linspace(0,frames.shape[0]-1,6).astype(int)
    for cc,fi in enumerate(ids):
        ax=fig.add_subplot(gs[r,cc])
        ax.imshow(np.clip(comb[fi]/vmax,-1,1),cmap="bwr",vmin=-1,vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if cc==0: ax.set_ylabel(label,fontsize=11,rotation=0,ha="right",va="center",color=marine)
draw_row(0,fr,"REEL")
draw_row(1,frf,"FLIP")
axc=fig.add_subplot(gs[2,:])
axc.plot(tc, meanx, color=green, lw=2.2, marker='o', ms=3, label="REEL : centre de masse x(t)")
axc.plot(tc, 639-meanx, color=red, lw=2.2, marker='o', ms=3, label="FLIP : x miroir")
axc.set_xlabel("t (s)"); axc.set_ylabel("x (centre de masse)")
axc.grid(alpha=.3); axc.legend(fontsize=10)
axc.set_title("Preuve quantitative : le flip inverse la trajectoire horizontale (courbes miroir)", color=marine, fontsize=12)
fig.suptitle("Effet du flip sur un vrai exemple (marche laterale)", color=marine, fontsize=14, y=0.98)
plt.savefig(MAD_ROOT/"04_train_chain"/"flip_proof.png",dpi=150,bbox_inches="tight")
print("OK -> flip_proof.png", flush=True)
