import re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
log = HERE / "train_chain_100.log"
marine = "#043353"
pat = re.compile(r"Epoch : (\d+) \| Train Loss : ([\d.]+) \| Train Accuracy : ([\d.]+) \| "
                 r"Val\(std\) Loss : ([\d.]+) \| Val\(std\) Accuracy : ([\d.]+) \| "
                 r"Val\(ada\) Loss : ([\d.]+) \| Val\(ada\) Accuracy : ([\d.]+)")
E,tl,ta,vsl,vsa,val,vaa = [],[],[],[],[],[],[]
for line in open(log):
    m = pat.search(line)
    if m:
        g = m.groups()
        E.append(int(g[0])); tl.append(float(g[1])); ta.append(float(g[2]))
        vsl.append(float(g[3])); vsa.append(float(g[4])); val.append(float(g[5])); vaa.append(float(g[6]))

# --- train ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(E, tl, color=marine, lw=2); ax[0].set_title("Train — loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
ax[1].plot(E, ta, color=marine, lw=2); ax[1].set_title("Train — accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0,1); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(HERE/"curve_madchain_train.png", dpi=160, bbox_inches="tight"); plt.close()

# --- val (std + ada, + train en pointille pour montrer l'ecart) ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(E, vsl, color="#7aa6c2", lw=1.5, label="val std")
ax[0].plot(E, val, color="#c0392b", lw=1.5, label="val ada")
ax[0].set_title("Validation — loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3); ax[0].legend()
ax[1].plot(E, ta,  color="0.6", lw=1.2, ls="--", label="train")
ax[1].plot(E, vsa, color="#7aa6c2", lw=1.5, label="val std")
ax[1].plot(E, vaa, color="#c0392b", lw=1.5, label="val ada")
ax[1].set_title("Validation — accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0,1); ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.savefig(HERE/"curve_madchain_val.png", dpi=160, bbox_inches="tight"); plt.close()
print("OK -> curve_madchain_train.png & curve_madchain_val.png")
