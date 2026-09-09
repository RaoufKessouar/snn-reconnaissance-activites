import re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
LOG=Path("/users/abdekess61/raouf/SResNet/experiments/mad/04_train_chain/train_chain_tdbn_L4_qif.log")
ep=[]; vs=[]; tr=[]
for line in LOG.read_text().splitlines():
    m=re.search(r"Epoch : (\d+).*Train Accuracy : ([\d.]+).*Val\(std\) Accuracy : ([\d.]+)",line)
    if m: ep.append(int(m.group(1))); tr.append(float(m.group(2))); vs.append(float(m.group(3)))
MARINE="#043353"; RED="#C0392B"; BLUE="#2E6DB4"; GREY="#888888"
fig,ax=plt.subplots(figsize=(6.5,4.3))
ax.plot(ep,tr,color=BLUE,label="QIF — entraînement")
ax.plot(ep,vs,color=RED,label="QIF — validation")
ax.axhline(0.9704,ls="--",color=MARINE,label="LIF+tdBN (réf. L=4) = 0,97")
ax.axhline(1/81,ls=":",color=GREY,label="Hasard = 0,012")
ax.set_xlabel("Époque"); ax.set_ylabel("Exactitude"); ax.set_ylim(0,1.02)
ax.grid(alpha=0.3); ax.legend(fontsize=8,loc="center right"); plt.tight_layout()
fig.savefig(OUT/"fig_qif_courbe.pdf"); fig.savefig(OUT/"fig_qif_courbe.png",dpi=200,facecolor="white")
print(f"OK fig_qif_courbe ({len(ep)} epoques)")
