import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
MARINE="#043353"; RED="#C0392B"
labels=["LIF + tdBN","QIF"]; acc=[0.9704,0.18]
fig,ax=plt.subplots(figsize=(5,4.3))
b=ax.bar(labels,acc,color=[MARINE,RED],width=0.55)
ax.text(b[0].get_x()+b[0].get_width()/2,0.9704+0.012,"0,97",ha="center")
ax.text(b[1].get_x()+b[1].get_width()/2,0.18+0.012,"~0,18",ha="center")
ax.set_ylabel("Exactitude validation (L=4, 81 classes)"); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=0.3)
ax.text(1,0.32,"x2 mémoire\n~2,6 h/époque\n~6,8 jours",ha="center",fontsize=8,color=RED)
plt.tight_layout(); fig.savefig(OUT/"fig_qif_vs_lif.pdf")
fig.savefig(OUT/"fig_qif_vs_lif.png",dpi=200,facecolor="white"); print("OK fig_qif_vs_lif")
