import sys; from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42
EXP=Path("/users/abdekess61/raouf/SResNet/experiments/mad"); R=Path("/users/abdekess61/raouf/SResNet")
sys.path.insert(0,str(R)); sys.path.insert(0,str(EXP/"common"))
from model_tdbn import SResNest
from mad_chain_dataset import MADChainDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
OUT=Path("/users/abdekess61/raouf/SResNet/figs_rapport"); MARINE="#043353"
T,SEQ,BS=40,3,4; ACT=(1,3,9); VAL_P=list(range(71,86))
NAMES={1:"Marcher",3:"S'asseoir",9:"Chuter"}
dev="cuda" if torch.cuda.is_available() else "cpu"
ds=MADChainDataset(EXTRACT_DIR,CACHE_DIR,participants=VAL_P,activities=ACT,seq_len=SEQ,T=T,
                   samples_per_class=3,split_name="val")
ld=DataLoader(ds,batch_size=BS,shuffle=False,num_workers=4); cl=ds.class_list
m=SResNest(num_steps=T,num_classes=len(cl)).to(dev)
m.load_state_dict(torch.load(EXP/"04_train_chain"/"best_chain_tdbn.pth",map_location=dev)); m.eval()
P=[]; Y=[]
with torch.no_grad():
    for x,y in ld:
        functional.reset_net(m); x=x.to(dev,dtype=torch.float32)
        P.append(m(x).argmax(1).cpu()); Y.append(y)
functional.reset_net(m); P=torch.cat(P).numpy(); Y=torch.cat(Y).numpy()
acts=list(ACT); a2i={a:i for i,a in enumerate(acts)}; M=np.zeros((3,3),int)
for p,y in zip(P,Y):
    for pos in range(SEQ): M[a2i[cl[y][pos]], a2i[cl[p][pos]]]+=1
labs=[NAMES[a] for a in acts]
pd.DataFrame(M,index=labs,columns=labs).to_csv(OUT/"confmat_madchain.csv")
fig,ax=plt.subplots(figsize=(5.2,4.4)); im=ax.imshow(M,cmap="Blues"); vmax=M.max() or 1
ax.set_xticks(range(3)); ax.set_xticklabels(labs,rotation=20,ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(labs)
for i in range(3):
    for j in range(3):
        ax.text(j,i,str(M[i,j]),ha="center",va="center",
                color="white" if M[i,j]>vmax*.5 else MARINE,fontsize=11)
ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité")
plt.colorbar(im,fraction=0.046,pad=0.04); plt.tight_layout()
fig.savefig(OUT/"confmat_madchain.pdf",bbox_inches="tight")
fig.savefig(OUT/"confmat_madchain.png",dpi=200,bbox_inches="tight",facecolor="white")
print("saved confmat_madchain.pdf")
