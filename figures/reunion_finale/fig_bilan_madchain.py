import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
MARINE="#043353"; RED="#C0392B"; BLUE="#2E6DB4"; GREY="#888888"
labels=["ANN-tdBN\n(sans mémoire)","SNN-tdBN\n(impulsionnel)","GRU\n(récurrent)"]
acc=[0.3185,0.9712,1.0000]
fig,ax=plt.subplots(figsize=(6,4.5))
b=ax.bar(labels,acc,color=[RED,MARINE,BLUE],width=0.6)
for bar in b: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.012,f"{bar.get_height():.3f}",ha="center")
ax.axhline(0.370,ls="--",color=GREY,label="Borne invariante à la permutation (0,37)")
ax.axhline(0.037,ls=":",color=GREY,label="Hasard (0,037)")
ax.set_ylabel("Exactitude test (MAD-Chain, 27 classes)"); ax.set_ylim(0,1.1)
ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3); plt.tight_layout()
fig.savefig(OUT/"fig_bilan_madchain.pdf"); fig.savefig(OUT/"fig_bilan_madchain.png",dpi=200,facecolor="white"); print("OK fig_bilan_madchain")
