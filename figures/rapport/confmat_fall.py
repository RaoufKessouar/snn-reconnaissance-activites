import sys; from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"]=42; plt.rcParams["ps.fonttype"]=42
EXP=Path("/users/abdekess61/raouf/SResNet/experiments/mad"); R=Path("/users/abdekess61/raouf/SResNet")
sys.path.insert(0,str(R)); sys.path.insert(0,str(EXP/"common"))
from model_tdbn_fpp import SResNestFPP
from mad_fall_dataset import MADFallDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR
OUT=Path("/users/abdekess61/raouf/SResNet/figs_rapport"); MARINE="#043353"
T,SEQ,BS=50,5,3; ACT=(1,2,3,4,5,6,7,8,9); VAL_P=list(range(71,86))
NAMES={1:"Walk",2:"Stand Up",3:"Sit Down",4:"Up Stairs",5:"Down Stairs",6:"Pick",7:"Step Over",8:"Semi Turn",9:"Fall"}
dev="cuda" if torch.cuda.is_available() else "cpu"
ds=MADFallDataset(EXTRACT_DIR,CACHE_DIR,participants=VAL_P,activities=ACT,fall_activity=9,seq_len=SEQ,
                  T=T,samples_per_participant=30,p_fall=0.5,seed=1,split_name="val")
ld=DataLoader(ds,batch_size=BS,shuffle=False,num_workers=4); K=len(ACT)
m=SResNestFPP(num_steps=T,num_classes=K).to(dev)
m.load_state_dict(torch.load(EXP/"06_fall_detect"/"best_T50_fb2.0_kcurve.pth",map_location=dev)); m.eval()
M=np.zeros((K,K),int)
with torch.no_grad():
    for x,y in ld:
        functional.reset_net(m); x=x.to(dev,dtype=torch.float32)
        p=m(x).argmax(-1).cpu().numpy(); yy=y.numpy()
        for a,b in zip(yy.ravel(),p.ravel()): M[a,b]+=1
functional.reset_net(m)
labs=[NAMES[a] for a in ACT]
pd.DataFrame(M,index=labs,columns=labs).to_csv(OUT/"confmat_fall.csv")
Mn=M/np.maximum(M.sum(1,keepdims=True),1)   # normalisee par ligne
fig,ax=plt.subplots(figsize=(6.6,5.6)); im=ax.imshow(Mn,cmap="Blues",vmin=0,vmax=1)
ax.set_xticks(range(K)); ax.set_xticklabels(labs,rotation=35,ha="right",fontsize=8)
ax.set_yticks(range(K)); ax.set_yticklabels(labs,fontsize=8)
for i in range(K):
    for j in range(K):
        if Mn[i,j]>=0.01:
            ax.text(j,i,f"{Mn[i,j]:.2f}",ha="center",va="center",
                    color="white" if Mn[i,j]>.5 else MARINE,fontsize=7)
ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité")
plt.colorbar(im,fraction=0.046,pad=0.04); plt.tight_layout()
fig.savefig(OUT/"confmat_fall.pdf",bbox_inches="tight")
fig.savefig(OUT/"confmat_fall.png",dpi=200,bbox_inches="tight",facecolor="white")
print("saved confmat_fall.pdf (ligne 'Fall' = avec quoi la chute est confondue)")
