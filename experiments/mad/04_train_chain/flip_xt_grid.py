import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MAD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAD_ROOT/"common"))
from mad_io import load_events
from mad_dataset import active_window
from mad_paths import EXTRACT_DIR

W=640
# clips avec mouvement HORIZONTAL (marche laterale A013, diagonale A014)
pats=["A013P001R1*","A014P001R1*"]
clips=[c[0] for c in (list(Path(EXTRACT_DIR).rglob(p)) for p in pats) if c]
if not clips: clips=list(Path(EXTRACT_DIR).rglob("A01*P001R1*.csv"))[:2]

fig,axes=plt.subplots(len(clips),2,figsize=(12,4.5*len(clips)))
if len(clips)==1: axes=axes[None,:]
for r,f in enumerate(clips):
    x,y,p,t=load_events(f); ws,we=active_window(t,3_000_000)
    m=(t>=ws)&(t<we); xx=x[m]; tt=(t[m]-ws)/1e6; pp=p[m]
    s=slice(None,None,max(1,len(xx)//100000))
    axes[r,0].scatter(tt[s],xx[s],c=pp[s],cmap="coolwarm",s=0.3,alpha=0.4)
    axes[r,0].set_title(f"{f.name[:5]} — REEL : x=f(t)"); axes[r,0].set_ylim(0,W)
    axes[r,0].set_xlabel("t (s)"); axes[r,0].set_ylabel("x")
    axes[r,1].scatter(tt[s],(W-1-xx[s]),c=pp[s],cmap="coolwarm",s=0.3,alpha=0.4)
    axes[r,1].set_title(f"{f.name[:5]} — FLIP : x miroir"); axes[r,1].set_ylim(0,W)
    axes[r,1].set_xlabel("t (s)"); axes[r,1].set_ylabel("x")
plt.suptitle("Projection x-t avant / apres flip : la direction (pente) s'inverse", fontsize=13)
plt.tight_layout(); plt.savefig(MAD_ROOT/"04_train_chain"/"flip_xt_grid.png",dpi=150,bbox_inches="tight")
print("OK -> flip_xt_grid.png", flush=True)
