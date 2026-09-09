from pathlib import Path
import tarfile, tempfile, shutil, io, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spikingjelly.datasets as sjds

ARCHIVE = Path("/users/abdekess61/raouf/SResNet/data/DvsGesture.tar.gz")
OUT = Path("/users/abdekess61/raouf/SResNet/figs_rapport"); OUT.mkdir(parents=True, exist_ok=True)
T = 40
FRAME_IDS = list(np.linspace(0, T-1, 6, dtype=int))          # 6 frames par ligne
TARGET = {0: "0 · Hand Clapping", 7: "7 · Arm Roll", 8: "8 · Air Drums"}   # label 0-based

def events_to_frames(ev, start_t, end_t, frames_number=T, H=128, W=128):
    m = (ev["t"] >= start_t) & (ev["t"] < end_t)
    x = ev["x"][m].astype(np.int64); y = ev["y"][m].astype(np.int64); p = ev["p"][m].astype(np.int64)
    fr = np.zeros((frames_number, 2, H, W), dtype=np.float32)
    if len(x) == 0: return fr
    p = (p > 0).astype(np.int64)
    b = np.linspace(0, len(x), frames_number + 1, dtype=np.int64)
    for t in range(frames_number):
        s, e = b[t], b[t+1]
        if e <= s: continue
        xs, ys, ps = x[s:e], y[s:e], p[s:e]
        v = (xs>=0)&(xs<W)&(ys>=0)&(ys<H)
        np.add.at(fr[t], (ps[v], ys[v], xs[v]), 1)
    return fr

def signed(fr): return fr[:,1] - fr[:,0]
def norm(img):
    vmax = np.percentile(np.abs(img), 99); vmax = vmax if vmax > 0 else 1.0
    return np.clip(img / vmax, -1, 1)

collected = {}
tmp = Path(tempfile.mkdtemp(prefix="dvsfig_", dir="/tmp"))
try:
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        for lf in sorted(n for n in members if n.endswith("_labels.csv")):
            if all(k in collected for k in TARGET): break
            aed = lf.replace("_labels.csv", "") + ".aedat"
            if aed not in members: continue
            csv = np.loadtxt(io.BytesIO(tar.extractfile(members[lf]).read()),
                             dtype=np.uint32, delimiter=",", skiprows=1)
            if not any((int(r[0])-1) in TARGET and (int(r[0])-1) not in collected for r in csv): continue
            tar.extract(members[aed], path=tmp)
            ev = sjds.load_aedat_v3(str(tmp/aed))
            for r in csv:
                lab = int(r[0]) - 1
                if lab in TARGET and lab not in collected:
                    collected[lab] = events_to_frames(ev, int(r[1]), int(r[2]))
            os.remove(tmp/aed)

    labels = [k for k in (0, 7, 8) if k in collected]
    fig, axes = plt.subplots(len(labels), len(FRAME_IDS),
        figsize=(2.1*len(FRAME_IDS), 2.4*len(labels)),
        gridspec_kw={"left":0.11,"right":0.99,"bottom":0.04,"top":0.86,"wspace":0.05,"hspace":0.22})
    axes = np.atleast_2d(axes)
    for row, lab in enumerate(labels):
        sf = signed(collected[lab])
        for col, t in enumerate(FRAME_IDS):
            ax = axes[row, col]
            ax.imshow(norm(sf[t]), cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"t = {t}", fontsize=11)
            if col == 0: ax.set_ylabel(TARGET[lab], fontsize=12)
    fig.suptitle("Les trois gestes primitifs après accumulation en images successives",
                 fontsize=15, y=0.955)
    out = OUT/"figure10_gestes_primitifs.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved:", out)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("done")
