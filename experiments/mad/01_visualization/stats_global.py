import sys, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_io import load_events
from mad_naming import parse, ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR
FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)

csvs = sorted(EXTRACT_DIR.rglob("*.csv"))
rate, cnt = defaultdict(list), defaultdict(int)
occ = np.zeros((480, 640))
for i, c in enumerate(csvs):
    d = parse(c.name)
    if not d: continue
    x, y, p, t = load_events(c)
    dur = (t.max() - t.min()) / 1e6
    rate[d["activity"]].append(len(t) / max(dur, 1e-6) / 1e3); cnt[d["activity"]] += 1
    s = slice(None, None, max(1, len(t)//50000))
    np.add.at(occ, (np.clip(y[s],0,479), np.clip(x[s],0,639)), 1.0)
    if i % 30 == 0: print(f"  {i}/{len(csvs)}", flush=True)

acts = sorted(rate); names = [ACTIVITY_NAMES[a] for a in acts]
plt.figure(figsize=(9,4)); plt.bar(names, [cnt[a] for a in acts], color="steelblue")
plt.ylabel("nb enregistrements"); plt.title("Enregistrements par activite")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(FIG/"01_counts_by_activity.png", dpi=150); plt.close()

plt.figure(figsize=(10,5)); plt.boxplot([rate[a] for a in acts], labels=names, showfliers=False)
plt.ylabel("taux (k ev/s)"); plt.title("Densite d'evenements par activite")
plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(FIG/"02_eventrate_by_activity.png", dpi=150); plt.close()

plt.figure(figsize=(7,5)); plt.imshow(np.log1p(occ), cmap="magma", aspect="auto")
plt.colorbar(label="log(1+events)"); plt.title("Occupation spatiale (640x480)")
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
plt.savefig(FIG/"03_spatial_occupancy.png", dpi=150); plt.close()
print("OK -> figures/")
