import re, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- style ----------
MARINE="#043353"; BEIGE="#E8E2D6"; RED="#c0392b"; GREEN="#2e7d5b"; GREY="#8a8a8a"; ORANGE="#e08a1e"
plt.rcParams.update({"font.size":12,"axes.edgecolor":MARINE,"axes.labelcolor":MARINE,
    "xtick.color":MARINE,"ytick.color":MARINE,"axes.titlecolor":MARINE,"figure.dpi":150})

BASE=Path("/users/abdekess61/raouf/SResNet/experiments/mad/04_train_chain")
OUT=Path("/users/abdekess61/raouf/SResNet/experiments/mad/figs_reunion3"); OUT.mkdir(parents=True,exist_ok=True)

# ---------- parseur de log train_chain*.log ----------
PAT=re.compile(r"Epoch : (\d+).*?Train Accuracy : ([\d.]+).*?Val\(std\) Accuracy : ([\d.]+).*?Val\(ada\) Accuracy : ([\d.]+)")
def parse(path):
    ep,tr,std,ada=[],[],[],[]
    if not Path(path).exists():
        print("  [manquant]",path); return ep,tr,std,ada
    for ln in open(path):
        m=PAT.search(ln)
        if m:
            ep.append(int(m.group(1))); tr.append(float(m.group(2)))
            std.append(float(m.group(3))); ada.append(float(m.group(4)))
    return ep,tr,std,ada

def save(fig,name):
    p=OUT/name; fig.savefig(p,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("  -> ",p)

# ================= FIG 1 : BNTT vs tdBN (validation) =================
try:
    e1,_,s1,_=parse(BASE/"train_chain_100.log")      # BNTT baseline seq3
    e2,_,s2,_=parse(BASE/"train_chain_tdbn.log")     # tdBN seq3
    fig,ax=plt.subplots(figsize=(8,4.5))
    if e1: ax.plot(e1,s1,color=RED,lw=1.8,label="BNTT (oscille)")
    if e2: ax.plot(e2,s2,color=GREEN,lw=2.2,label="tdBN (lisse)")
    ax.set_xlabel("epoch"); ax.set_ylabel("Accuracy validation")
    ax.set_title("Validation MAD-Chain : BNTT vs tdBN")
    ax.legend(frameon=False); ax.grid(alpha=.2); ax.set_ylim(0,1)
    save(fig,"fig1_bntt_vs_tdbn_val.png")
except Exception as ex: print("FIG1 err:",ex)

# ================= FIG 2 : ANN vs SNN =================
try:
    tasks=["DVS-GC (0/7/8)","MAD-Chain (réel)"]; ann=[0.35,0.15]; snn=[0.82,0.78]; chance=0.037
    x=range(len(tasks)); w=0.35
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.bar([i-w/2 for i in x],ann,w,color=RED,label="ANN-BN")
    ax.bar([i+w/2 for i in x],snn,w,color=MARINE,label="SNN")
    ax.axhline(chance,ls="--",color=GREY,lw=1); ax.text(1.35,chance+.02,"hasard",color=GREY,fontsize=10)
    for i,(a,s) in enumerate(zip(ann,snn)):
        ax.text(i-w/2,a+.02,f"{a:.2f}",ha="center",color=RED,fontsize=10)
        ax.text(i+w/2,s+.02,f"{s:.2f}",ha="center",color=MARINE,fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks); ax.set_ylabel("Accuracy (test)")
    ax.set_title("Perception d'ordre : SNN vs ANN"); ax.set_ylim(0,1); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    save(fig,"fig2_ann_vs_snn.png")
except Exception as ex: print("FIG2 err:",ex)

# ================= FIG 3 : progression test BNTT / SWA / tdBN =================
try:
    labels=["BNTT\nbaseline","BNTT\n+ SWA","tdBN"]; vals=[0.78,0.73,0.975]; cols=[GREY,ORANGE,GREEN]
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    ax.bar(labels,vals,color=cols,width=.6)
    for i,v in enumerate(vals): ax.text(i,v+.015,f"{v:.3f}",ha="center",color=MARINE,fontsize=11,fontweight="bold")
    ax.set_ylabel("Accuracy test (sujets inédits)"); ax.set_ylim(0,1)
    ax.set_title("MAD-Chain : bond de performance avec tdBN"); ax.grid(axis="y",alpha=.2)
    save(fig,"fig3_test_progression.png")
except Exception as ex: print("FIG3 err:",ex)

# ================= FIG 4 : chaînes plus longues (seq_len=4) =================
try:
    ea,_,_,aa=parse(BASE/"train_chain_tdbn_L4.log")            # ep 1-40
    er,_,_,ar=parse(BASE/"train_chain_tdbn_L4_resume.log")     # ep 41-60
    ep=ea+er; val=aa+ar
    fig,ax=plt.subplots(figsize=(8,4.5))
    if ep: ax.plot(ep,val,color=MARINE,lw=2.2,label="seq_len=4 (81 classes)")
    ax.axhline(0.969,ls="--",color=GREEN,lw=1.5,label="plateau seq_len=3 (27 classes)")
    ax.set_xlabel("epoch"); ax.set_ylabel("Accuracy validation (ada)")
    ax.set_title("La perception d'ordre scale aux chaînes longues")
    ax.set_ylim(0,1); ax.legend(frameon=False,loc="lower right"); ax.grid(alpha=.2)
    save(fig,"fig4_seqlen4_scaling.png")
except Exception as ex: print("FIG4 err:",ex)

# ================= FIG 5 : fenêtre active (résultat négatif) =================
try:
    em,_,_,am=parse(BASE/"train_chain_tdbn.log")        # maxdensity
    el,_,_,al=parse(BASE/"train_chain_tdbn_aw.log")     # late
    fig,ax=plt.subplots(figsize=(8,4.5))
    if em: ax.plot(em,am,color=GREEN,lw=2.2,label="fenêtre « densité max » (0,97)")
    if el: ax.plot(el,al,color=RED,lw=1.8,label="fenêtre « late » (0,80)")
    ax.set_xlabel("epoch"); ax.set_ylabel("Accuracy validation (ada)")
    ax.set_title("Fenêtre active « late » : résultat négatif")
    ax.set_ylim(0,1); ax.legend(frameon=False,loc="lower right"); ax.grid(alpha=.2)
    save(fig,"fig5_fenetre_active_negatif.png")
except Exception as ex: print("FIG5 err:",ex)

# ================= FIG 6 : schéma BNTT vs tdBN (concept) =================
try:
    fig,(axL,axR)=plt.subplots(1,2,figsize=(10,4.2))
    T,B=5,3
    for ax,title,mode in [(axL,"BNTT : 1 BN par pas de temps\n(μ,σ sur B valeurs)","bntt"),
                          (axR,"tdBN : 1 seule normalisation\n(μ,σ sur T×B valeurs)","tdbn")]:
        if mode=="tdbn":
            ax.add_patch(Rectangle((-.15,-.15),B-1+.3,T-1+.3,fill=True,color=GREEN,alpha=.15,lw=2,ec=GREEN))
        for t in range(T):
            if mode=="bntt":
                ax.add_patch(Rectangle((-.15,t-.15),B-1+.3,.3+.0,fill=True,color=RED,alpha=.12,ec=RED,ls="--",lw=1))
            for b in range(B):
                ax.add_patch(Rectangle((b-.13,t-.13),.26,.26,fill=True,color="#c9c9d2",ec=MARINE,lw=1))
        ax.set_xlim(-.6,B-.4); ax.set_ylim(-.6,T-.4); ax.invert_yaxis()
        ax.set_xticks(range(B)); ax.set_xticklabels([f"éch {i+1}" for i in range(B)],fontsize=9)
        ax.set_yticks(range(T)); ax.set_yticklabels([f"t={i+1}" for i in range(T)],fontsize=9)
        ax.set_title(title,fontsize=12); ax.set_aspect("equal")
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Où chaque normalisation calcule ses statistiques",color=MARINE,fontsize=13)
    save(fig,"fig6_schema_bntt_tdbn.png")
except Exception as ex: print("FIG6 err:",ex)

print("\nToutes les figures dans:",OUT)
print(os.listdir(OUT))
