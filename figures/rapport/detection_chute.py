from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

OUT = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)

NAVY="#1F3A5F"; CONV="#D6E4F0"; NORM="#F3DDB3"; LIF="#D9E8D5"
PALE="#EFEFEA"; GREY="#4A4A4A"; BORD="#800020"; FALL="#E9CFD5"
plt.rcParams["font.family"]="DejaVu Sans"

fig,ax=plt.subplots(figsize=(11.0,5.0))
ax.set_xlim(0,11.6); ax.set_ylim(0,5.2); ax.axis("off")

def dash(x1,y1,x2,y2,c="#9AA7B4"):
    ax.add_line(plt.Line2D([x1,x2],[y1,y2],color=c,lw=1.0,
                ls=(0,(3,2)),zorder=1))
def arrow(x1,y1,x2,y2,c=GREY,lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
        mutation_scale=12,linewidth=lw,color=c,zorder=5))

X0=2.35; W=7.4
segs=[("Marcher",1.55,CONV),("Enjamber",1.35,LIF),("Chuter",1.10,FALL),
      ("Marcher",1.75,CONV),("S'asseoir",1.65,NORM)]

# ---- niveau 1 : sequence -------------------------------------------
y1,h1=4.05,0.55
x=X0; fall_x0=fall_x1=None
for lab,w,col in segs:
    ec = BORD if col==FALL else NAVY
    lw = 1.8 if col==FALL else 1.2
    ax.add_patch(Rectangle((x,y1),w,h1,facecolor=col,edgecolor=ec,
                 lw=lw,zorder=3))
    ax.text(x+w/2,y1+h1/2,lab,ha="center",va="center",fontsize=8.2,
            color=BORD if col==FALL else NAVY,zorder=4)
    if col==FALL: fall_x0,fall_x1=x,x+w
    x+=w
ax.text(X0-0.15,y1+h1/2,"Séquence",ha="right",va="center",
        fontsize=9.5,fontweight="bold",color=NAVY)
ax.text(X0+W/2,y1+h1+0.24,"cinq activités, la chute pouvant être présente "
        "ou absente",ha="center",va="center",fontsize=8.5,color=GREY)

# ---- niveau 2 : prediction par image --------------------------------
y2,h2=2.75,0.42
N=30; g=0.035
cw=(W-(N-1)*g)/N
fall_idx=set()
for k in range(N):
    xk=X0+k*(cw+g); xc=xk+cw/2
    if fall_x0<=xc<=fall_x1: fall_idx.add(k)
faux=8                      # erreur isolee
for k in range(N):
    xk=X0+k*(cw+g)
    is_fall = (k in fall_idx) or (k==faux)
    col = FALL if is_fall else PALE
    ec  = BORD if is_fall else NAVY
    lw  = 1.5 if is_fall else 0.7
    ax.add_patch(Rectangle((xk,y2),cw,h2,facecolor=col,edgecolor=ec,
                 lw=lw,zorder=3))
ax.text(X0-0.15,y2+h2/2,"Prédiction\npar image",ha="right",va="center",
        fontsize=9.5,fontweight="bold",color=NAVY,linespacing=1.3)
for xb in (X0,fall_x0,fall_x1,X0+W):
    dash(xb,y1,xb,y2+h2)

xf=X0+faux*(cw+g)+cw/2
ax.annotate("erreur isolée",xy=(xf,y2-0.06),xytext=(xf-0.35,y2-0.62),
            ha="center",fontsize=8.2,color=GREY,
            arrowprops=dict(arrowstyle="-|>",color=GREY,lw=1.0))

# ---- niveau 3 : decision --------------------------------------------
ka=min(fall_idx); kb=max(fall_idx)
xa=X0+ka*(cw+g); xb=X0+kb*(cw+g)+cw
ax.annotate("",xy=(xa,y2-0.14),xytext=(xb,y2-0.14),
            arrowprops=dict(arrowstyle="<->",color=BORD,lw=1.1))
ax.text((xa+xb)/2,y2-0.46,"$n$ images prédites comme chute",ha="center",
        va="center",fontsize=8.5,color=BORD)

y3=1.05
ax.add_patch(FancyBboxPatch((3.55,y3),2.05,0.62,
    boxstyle="round,pad=0.02,rounding_size=0.10",linewidth=1.4,
    edgecolor=NAVY,facecolor="#FCFCFA",zorder=3))
ax.text(4.57,y3+0.31,"$n \\geq K$ ?",ha="center",va="center",
        fontsize=10.5,color=NAVY,zorder=4)
arrow(5.65,y3+0.31,6.35,y3+0.31)
ax.add_patch(FancyBboxPatch((6.40,y3),1.55,0.62,
    boxstyle="round,pad=0.02,rounding_size=0.10",linewidth=1.6,
    edgecolor=BORD,facecolor=FALL,zorder=3))
ax.text(7.17,y3+0.31,"alerte",ha="center",va="center",fontsize=10,
        color=BORD,zorder=4)
dash((xa+xb)/2,y2-0.58,4.57,y3+0.68,BORD)
ax.text(X0-0.15,y3+0.31,"Décision",ha="right",va="center",
        fontsize=9.5,fontweight="bold",color=NAVY)
ax.text(8.15,y3+0.31,"la position des images concernées\ndonne la "
        "localisation temporelle",ha="left",va="center",fontsize=8.2,
        color=GREY,linespacing=1.35)

ax.text(5.8,0.28,"Le seuil $K$ règle le compromis : trop bas, une erreur "
        "isolée déclenche une alerte ; trop haut, une chute brève est "
        "manquée.",ha="center",va="center",fontsize=9,color=BORD,
        style="italic")

fig.savefig(OUT/"detection_chute.pdf",bbox_inches="tight")
fig.savefig(OUT/"detection_chute.png",dpi=170,bbox_inches="tight")
print("ok ->", OUT/"detection_chute.png")
