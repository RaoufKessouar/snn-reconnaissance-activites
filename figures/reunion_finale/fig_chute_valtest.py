import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
metrics=["Rappel","Précision","F1","Fausses alarmes"]
val=[0.987,0.895,0.939,0.124]; test=[0.991,0.902,0.945,0.115]
BLUE="#2E6DB4"; RED="#C0392B"; x=np.arange(len(metrics)); w=0.38
fig,ax=plt.subplots(figsize=(6.5,4.3))
b1=ax.bar(x-w/2,val,w,color=BLUE,label="Validation"); b2=ax.bar(x+w/2,test,w,color=RED,label="Test")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.012,f"{b.get_height():.3f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0,1.1); ax.set_ylabel("Valeur (seuil K=3)")
ax.legend(); ax.grid(axis="y",alpha=0.3); plt.tight_layout()
fig.savefig(OUT/"fig_chute_valtest.pdf"); fig.savefig(OUT/"fig_chute_valtest.png",dpi=200,facecolor="white"); print("OK fig_chute_valtest")
