import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42
BASE=Path("/users/abdekess61/raouf/SResNet/experiments/overlap_078_seq3_T80/error_analysis")
OUT=Path("/users/abdekess61/raouf/SResNet/figs_rapport"); MARINE="#043353"
NAMES={"0":"Hand Clapping","7":"Arm Roll","8":"Air Drums"}
cm=pd.read_csv(BASE/"gesture_confusion_matrix.csv",index_col=0)
cm=cm.drop(columns=[c for c in cm.columns if str(c).lower()=="acc"])
M=cm.values.astype(float); labs=[NAMES.get(str(i),str(i)) for i in cm.index]
fig,ax=plt.subplots(figsize=(5.4,4.6)); im=ax.imshow(M,cmap="Blues")
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=20,ha="right")
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
vmax=M.max() or 1
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j,i,f"{int(M[i,j])}",ha="center",va="center",
                color="white" if M[i,j]>vmax*.5 else MARINE,fontsize=11)
ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité")
plt.colorbar(im,fraction=0.046,pad=0.04); plt.tight_layout()
fig.savefig(OUT/"confmat_dvsgc.pdf",bbox_inches="tight")
fig.savefig(OUT/"confmat_dvsgc.png",dpi=200,bbox_inches="tight",facecolor="white")
print("saved confmat_dvsgc.pdf")
