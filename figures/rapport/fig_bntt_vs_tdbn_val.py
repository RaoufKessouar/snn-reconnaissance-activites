import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

D = Path("/users/abdekess61/raouf/SResNet/experiments/mad/04_train_chain")
OUT = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
LOG_BNTT = D/"train_chain_100.log"      # baseline BNTT (seq3) -- adapte si autre nom
LOG_TDBN = D/"train_chain_tdbn.log"     # run tdBN (seq3)
BNTT_COL = "#c0392b"    # rouge
TDBN_COL = "#2f6fb0"    # bleu (couleur differente)

def parse_val(path, prefer="std"):
    pats = {"std":  re.compile(r"Epoch : (\d+).*?Val\(std\) Accuracy : ([\d.]+)"),
            "ada":  re.compile(r"Epoch : (\d+).*?Val\(ada\) Accuracy : ([\d.]+)"),
            "plain":re.compile(r"Epoch : (\d+).*?Val Accuracy : ([\d.]+)")}
    order = [prefer] + [k for k in ("std","ada","plain") if k != prefer]
    ep, val = [], []
    for ln in open(path):
        for k in order:
            m = pats[k].search(ln)
            if m:
                ep.append(int(m.group(1))); val.append(float(m.group(2))); break
    return ep, val

eb, vb = parse_val(LOG_BNTT)
et, vt = parse_val(LOG_TDBN)
print(f"BNTT: {len(eb)} pts | tdBN: {len(et)} pts")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

axL.plot(eb, vb, color=BNTT_COL, lw=1.6, label="BNTT")
axL.set_xlabel("epoch"); axL.set_ylabel("Exactitude en validation"); axL.set_ylim(0, 1)
axL.grid(alpha=.25); axL.legend(frameon=False, loc="lower right")

axR.plot(et, vt, color=TDBN_COL, lw=1.8, label="tdBN")
axR.set_xlabel("epoch"); axR.set_ylim(0, 1)
axR.grid(alpha=.25); axR.legend(frameon=False, loc="lower right")

for ax in (axL, axR):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT/"fig_bntt_vs_tdbn_val.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig_bntt_vs_tdbn_val.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved:", OUT/"fig_bntt_vs_tdbn_val.pdf")
