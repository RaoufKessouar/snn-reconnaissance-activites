import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
plt.rcParams["pdf.fonttype"]=42
OUT=Path("/users/abdekess61/raouf/SResNet/reunion_finale")
df=pd.read_csv("/users/abdekess61/raouf/SResNet/experiments/mad/06_fall_detect/metrics_T50_fb2.0_kcurve.csv")
print("Colonnes:",list(df.columns))
cols={c.lower():c for c in df.columns}
def find(*keys):
    for k in keys:
        for lc,c in cols.items():
            if k in lc: return c
    return None
ep=find("epoch") or df.columns[0]; acc=find("frame_acc","frame","acc"); f1=find("f1_k3","f1_3","f1")
if acc is None or f1 is None:
    print("!! colonnes acc/F1 introuvables, colle le header ci-dessus"); sys.exit(0)
BLUE="#2E6DB4"; RED="#C0392B"
fig,ax1=plt.subplots(figsize=(6.5,4.3))
ax1.plot(df[ep],df[acc],color=BLUE); ax1.set_xlabel("Époque")
ax1.set_ylabel("Exactitude par trame",color=BLUE); ax1.tick_params(axis="y",labelcolor=BLUE)
ax2=ax1.twinx(); ax2.plot(df[ep],df[f1],color=RED)
ax2.set_ylabel("F1 de détection (K=3)",color=RED); ax2.tick_params(axis="y",labelcolor=RED)
ax1.grid(alpha=0.3); plt.tight_layout()
fig.savefig(OUT/"fig_chute_courbe.pdf"); fig.savefig(OUT/"fig_chute_courbe.png",dpi=200,facecolor="white")
print(f"OK fig_chute_courbe (ep={ep} acc={acc} f1={f1})")
