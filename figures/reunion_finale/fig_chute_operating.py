import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
K=[1,2,3,4,5]
val_fa=[0.468,0.239,0.124,0.064,0.037]; val_rec=[1.000,0.996,0.987,0.935,0.858]
test_fa=[0.404,0.206,0.115,0.069,0.037]; test_rec=[0.996,0.996,0.991,0.957,0.892]
MARINE="#043353"; RED="#C0392B"; BLUE="#2E6DB4"
fig,ax=plt.subplots(figsize=(6,4.5))
ax.plot(val_fa,val_rec,"o-",color=BLUE,label="Validation (71-85)")
ax.plot(test_fa,test_rec,"s-",color=RED,label="Test (86-100)")
for x,y,k in zip(test_fa,test_rec,K):
    ax.annotate(f"K={k}",(x,y),textcoords="offset points",xytext=(6,-11),fontsize=8,color=MARINE)
ax.set_xlabel("Taux de fausses alarmes"); ax.set_ylabel("Sensibilité (rappel)")
ax.set_ylim(0.82,1.01); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); fig.savefig(OUT/"fig_chute_operating.pdf")
fig.savefig(OUT/"fig_chute_operating.png",dpi=200,facecolor="white"); print("OK fig_chute_operating")
