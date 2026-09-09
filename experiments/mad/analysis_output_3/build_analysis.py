import sys, csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional

HERE = Path(__file__).resolve().parent            # analysis_output_3
MAD_ROOT = HERE.parent                            # mad
SRESNET_ROOT = MAD_ROOT.parents[1]                # SResNet
sys.path.insert(0, str(SRESNET_ROOT)); sys.path.insert(0, str(MAD_ROOT/"common"))
from model import SResNest
from mad_chain_dataset import MADChainDataset
from mad_paths import EXTRACT_DIR, CACHE_DIR

T=40; device="cuda"
SHORT={1:"Walk",3:"Sit",9:"Fall"}
test_set = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=list(range(86,101)),
                           activities=[1,3,9], seq_len=3, T=T, samples_per_class=3, split_name="test")
loader = DataLoader(test_set, batch_size=6, shuffle=False, num_workers=4)
class_list = test_set.class_list
ckpt = MAD_ROOT/"04_train_chain"/"best_chain.pth"

model = SResNest(num_steps=T, num_classes=len(class_list)).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

trues, preds = [], []
with torch.no_grad():
    for i,(x,y) in enumerate(loader):
        functional.reset_net(model)
        x=x.to(device=device,dtype=torch.float32)
        preds += model(x).argmax(1).cpu().tolist(); trues += y.tolist()
        if i%50==0: print(f"pred {i}/{len(loader)}", flush=True)
functional.reset_net(model)
n=len(trues)

def montage(dataset, rows, title, path):
    ncol=6; nrow=len(rows)
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*1.9, nrow*1.75))
    if nrow==1: axes=axes.reshape(1,-1)
    for r,(idx,lbl) in enumerate(rows):
        fr=dataset[idx][0].numpy(); comb=fr[:,1]-fr[:,0]
        vmax=np.percentile(np.abs(comb),99) or 1.0
        for cc,fidx in enumerate(np.linspace(0,fr.shape[0]-1,ncol).astype(int)):
            axes[r,cc].imshow(np.clip(comb[fidx]/vmax,-1,1),cmap="bwr",vmin=-1,vmax=1)
            axes[r,cc].set_xticks([]); axes[r,cc].set_yticks([])
        axes[r,0].set_ylabel(lbl,fontsize=7,rotation=0,ha="right",va="center")
    plt.suptitle(title,fontsize=12); plt.tight_layout()
    plt.savefig(path,dpi=140,bbox_inches="tight"); plt.close()
    print("saved", path, flush=True)

def lab(i):
    t=class_list[trues[i]]; p=class_list[preds[i]]
    ts="->".join(SHORT[a] for a in t); ps="->".join(SHORT[a] for a in p)
    marks="".join("o" if t[k]==p[k] else "x" for k in range(3))
    return f"V: {ts}\nP: {ps} [{marks}]"

def diverse(idxs, keyfn, k):
    seen=set(); out=[]
    for i in idxs:
        key=keyfn(i)
        if key not in seen:
            seen.add(key); out.append(i)
        if len(out)>=k: break
    return out

correct=[i for i in range(n) if preds[i]==trues[i]]
wrong=[i for i in range(n) if preds[i]!=trues[i]]
corr_sel=diverse(correct, lambda i: trues[i], 8)
wrong_sel=diverse(wrong, lambda i: (trues[i],preds[i]), 8)

montage(test_set, [(i,lab(i)) for i in corr_sel],
        "MAD-Chain — exemples BIEN classes (test)", HERE/"correct_cases.png")
montage(test_set, [(i,lab(i)) for i in wrong_sel],
        "MAD-Chain — exemples MAL classes (test)", HERE/"wrong_cases.png")

# sequences d'entree variees (entrainement, participants 1-3)
tv = MADChainDataset(EXTRACT_DIR, CACHE_DIR, participants=[1,2,3], activities=[1,3,9],
                     seq_len=3, T=T, samples_per_class=1, split_name="train")
want=[(1,1,1),(3,3,3),(9,9,9),(1,3,9),(9,3,1),(1,9,3),(3,1,9),(1,1,9)]
train_rows=[]
for w in want:
    ci=tv.class_list.index(w)
    for j,(cidx,_,_) in enumerate(tv.samples):
        if cidx==ci:
            train_rows.append((j,"->".join(SHORT[a] for a in w))); break
montage(tv, train_rows, "MAD-Chain — sequences d'entree variees (entrainement)",
        HERE/"train_sequences.png")

with open(HERE/"cases_manifest.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["set","idx","true","pred"])
    for i in corr_sel: wr.writerow(["correct",i,class_list[trues[i]],class_list[preds[i]]])
    for i in wrong_sel: wr.writerow(["wrong",i,class_list[trues[i]],class_list[preds[i]]])
print("done -> analysis_output_3/", flush=True)
