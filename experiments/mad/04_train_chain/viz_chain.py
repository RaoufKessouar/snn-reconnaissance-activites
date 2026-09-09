import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_chain_dataset import MADChainDataset
from mad_naming import ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR, CACHE_DIR
FIG=Path(__file__).resolve().parent/"figures"; FIG.mkdir(exist_ok=True)

# tailles reelles du run (juste la liste des echantillons, pas de lecture CSV)
tr=MADChainDataset(EXTRACT_DIR,CACHE_DIR,participants=list(range(1,25)),samples_per_class=3,split_name="train")
va=MADChainDataset(EXTRACT_DIR,CACHE_DIR,participants=list(range(25,31)),samples_per_class=3,split_name="val")
print(f"classes = {len(tr.class_list)} (ex: {tr.class_list[:4]} ...)")
print(f"train samples = {len(tr)} | val samples = {len(va)}")

# visualiser 4 chaines (participants deja extraits)
ds=MADChainDataset(EXTRACT_DIR,CACHE_DIR,participants=[1,2,3],samples_per_class=2,split_name="viz")
fig,axes=plt.subplots(4,6,figsize=(13,9))
for r in range(4):
    fr,ci=ds[r*11 % len(ds)]
    seq=ds.class_list[ci]
    comb=fr[:,1]-fr[:,0]; vmax=np.percentile(np.abs(comb),99) or 1
    for cc,fidx in enumerate(np.linspace(0,fr.shape[0]-1,6).astype(int)):
        axes[r,cc].imshow(np.clip(comb[fidx]/vmax,-1,1),cmap="bwr",vmin=-1,vmax=1)
        axes[r,cc].set_xticks([]); axes[r,cc].set_yticks([])
    name=" -> ".join(ACTIVITY_NAMES[a] for a in seq)
    axes[r,0].set_ylabel(name,fontsize=8)
plt.suptitle("MAD-Chain : exemples de chaines (Walk/Sit Down/Fall), 6 frames /40")
plt.tight_layout(); plt.savefig(FIG/"chain_examples.png",dpi=140); plt.close()
print("figure ->",FIG/"chain_examples.png")
