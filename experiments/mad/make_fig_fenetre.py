import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

MARINE="#043353"; RED="#c0392b"; GREEN="#2e7d5b"; GREY="#8a8a8a"
plt.rcParams.update({"font.size":12,"axes.titlecolor":MARINE,"axes.labelcolor":MARINE,
    "xtick.color":MARINE,"ytick.color":MARINE,"figure.dpi":150})
OUT=Path("/users/abdekess61/raouf/SResNet/experiments/mad/figs_reunion3"); OUT.mkdir(parents=True,exist_ok=True)

t=np.linspace(0,8,500)
walk=0.55*np.exp(-((t-2.2)**2)/(2*1.35**2))     # marche : densite large et soutenue
fall=0.92*np.exp(-((t-4.9)**2)/(2*0.33**2))      # chute : pic bref et intense
dens=np.maximum(0.05, walk+fall)
dens=np.where(t>5.7, 0.06, dens)                 # immobilite : quasi nul

fig,axes=plt.subplots(2,1,figsize=(9,6.4),sharex=True)

def panel(ax,title,win,color,outcome):
    ax.fill_between(t,dens,color=GREY,alpha=0.25)
    ax.plot(t,dens,color=GREY,lw=1.3)
    for xb in (4.0,5.5): ax.axvline(xb,ls="--",color=GREY,lw=1,alpha=.6)
    ax.add_patch(Rectangle((win[0],0),win[1]-win[0],1.02,color=color,alpha=0.16,ec=color,lw=2))
    ax.text((win[0]+win[1])/2,1.07,"fenetre 3 s",ha="center",color=color,fontsize=11,fontweight="bold")
    ax.text(6.85,0.62,outcome,ha="center",color=color,fontsize=12,fontweight="bold")
    ax.set_title(title,loc="left",fontsize=13,fontweight="bold")
    ax.set_ylim(0,1.25); ax.set_yticks([])
    for s in ["top","right","left"]: ax.spines[s].set_visible(False)

panel(axes[0],"Ancienne : densite max",(0.7,3.7),RED,"attrape la MARCHE\n-> predit Walk (faux)")
panel(axes[1],"Nouvelle : fin d'activite",(2.6,5.6),GREEN,"attrape la CHUTE\n-> predit Fall (ok)")

axes[1].set_xticks([2.0,4.75,6.8]); axes[1].set_xticklabels(["marche d'approche","chute","immobilite"])
axes[1].set_xlabel("temps  ->  (courbe grise = densite d'evenements)")
fig.suptitle("Fenetre active : viser le geste, pas la marche d'approche",color=MARINE,fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
p=OUT/"fig7_fenetre_concept.png"; fig.savefig(p,bbox_inches="tight",facecolor="white")
print("->",p)
