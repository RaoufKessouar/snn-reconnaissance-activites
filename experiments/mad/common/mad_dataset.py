import sys
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mad_io import load_events
from mad_naming import parse


def active_window(t, W_us, bin_us=50_000):
    """Fenetre glissante de duree W_us qui maximise le nombre d'evenements."""
    t0, t1 = int(t.min()), int(t.max())
    if t1 - t0 <= W_us:
        return t0, t1
    counts = np.bincount(((t - t0) // bin_us).astype(np.int64),
                         minlength=(t1 - t0) // bin_us + 1)
    w = max(1, W_us // bin_us)
    if w >= len(counts):
        return t0, t1
    csum = np.concatenate([[0], np.cumsum(counts)])
    wsum = csum[w:] - csum[:-w]
    sb = int(np.argmax(wsum))
    ws = t0 + sb * bin_us
    return ws, ws + W_us


def make_frames(x, y, p, t, ws, we, T, H, W, crop):
    """Crop spatial + downsample -> T frames a 2 canaux de polarite."""
    cx0, cx1, cy0, cy1 = crop
    m = (t >= ws) & (t < we) & (x >= cx0) & (x < cx1) & (y >= cy0) & (y < cy1)
    x, y, p, t = x[m], y[m], p[m], t[m]
    frames = np.zeros((T, 2, H, W), dtype=np.float32)
    if len(t) == 0:
        return frames
    xi = np.clip(((x - cx0).astype(np.float64) / (cx1 - cx0) * W).astype(np.int64), 0, W - 1)
    yi = np.clip(((y - cy0).astype(np.float64) / (cy1 - cy0) * H).astype(np.int64), 0, H - 1)
    span = max(int(we - ws), 1)
    fi = np.clip(((t - ws).astype(np.int64) * T // span), 0, T - 1)
    pc = np.clip(p.astype(np.int64), 0, 1)
    np.add.at(frames, (fi, pc, yi, xi), 1.0)
    return frames


class MADDataset(Dataset):
    def __init__(self, extract_dir, cache_dir, participants, activities=None,
                 T=40, W_s=3.0, H=128, W=128, crop=(120, 520, 80, 440)):
        self.T, self.W_s, self.H, self.W, self.crop = T, W_s, H, W, crop
        self.W_us = int(W_s * 1e6)
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        acts = set(activities) if activities else set(range(1, 10))
        parts = set(participants)
        items = []
        for c in sorted(Path(extract_dir).rglob("*.csv")):
            d = parse(c.name)
            if not d or d["sensor"] != 1:
                continue
            if d["participant"] not in parts or d["activity"] not in acts:
                continue
            items.append((c, d["activity"]))
        self.classes = sorted({a for _, a in items})
        self.a2i = {a: i for i, a in enumerate(self.classes)}
        self.items = items
        self.tag = f"T{T}_W{W_s}_H{H}_c{'-'.join(map(str, crop))}"

    def __len__(self):
        return len(self.items)

    def frames_for(self, csv):
        cp = self.cache_dir / f"{Path(csv).stem}_{self.tag}.npz"
        if cp.exists():
            return np.load(cp)["frames"]
        x, y, p, t = load_events(csv)
        ws, we = active_window(t, self.W_us)
        fr = make_frames(x, y, p, t, ws, we, self.T, self.H, self.W, self.crop)
        np.savez_compressed(cp, frames=fr)
        return fr

    def __getitem__(self, i):
        csv, act = self.items[i]
        return torch.from_numpy(self.frames_for(csv)), self.a2i[act]


def active_window_late(t, W_us, bin_us=50_000, thr_frac=0.15):
    """Fenetre de duree W_us ANCREE SUR LA FIN de l'activite.
    Vise le geste (dernier mouvement significatif avant immobilite),
    et non la marche d'approche (dense mais anterieure)."""
    t0, t1 = int(t.min()), int(t.max())
    if t1 - t0 <= W_us:
        return t0, t1
    counts = np.bincount(((t - t0) // bin_us).astype(np.int64),
                         minlength=(t1 - t0) // bin_us + 1)
    if counts.max() == 0:
        return t1 - W_us, t1
    thr = thr_frac * counts.max()            # seuil = fraction du pic
    active = np.where(counts >= thr)[0]
    last = int(active[-1]) if len(active) else len(counts) - 1
    we = min(t0 + (last + 1) * bin_us, t1)   # fin = juste apres le dernier bin actif
    ws = we - W_us
    if ws < t0:                              # clamp si l'activite finit tot
        ws, we = t0, t0 + W_us
    return ws, we
