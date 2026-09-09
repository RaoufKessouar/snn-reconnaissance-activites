import sys, itertools, hashlib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mad_io import load_events
from mad_naming import parse
from mad_dataset import active_window, active_window_late, make_frames
class MADChainDataset(Dataset):
    def __init__(self, extract_dir, cache_dir, participants, activities=(1,3,9), seq_len=3,
                 T=40, W_s=3.0, H=128, W=128, crop=(120,520,80,440),
                 samples_per_class=3, seed=0, split_name="train", augment=False,
                 window_mode="maxdensity"):
        self.T,self.W_s,self.H,self.W,self.crop = T,W_s,H,W,crop
        self.W_us=int(W_s*1e6); self.augment=augment
        self.window_mode=window_mode
        self.aw_fn = active_window_late if window_mode=="late" else active_window
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.activities=list(activities)
        self.class_list=list(itertools.product(self.activities, repeat=seq_len))
        aw_marker = "" if window_mode=="maxdensity" else f"_aw{window_mode}"
        self.tag=f"chain_a{'-'.join(map(str,self.activities))}_L{seq_len}_T{T}_W{W_s}_H{H}{aw_marker}_{split_name}"
        clips={}; parts=set(participants)
        for c in sorted(Path(extract_dir).rglob("*.csv")):
            d=parse(c.name)
            if not d or d["sensor"]!=1 or d["participant"] not in parts: continue
            if d["activity"] in set(self.activities):
                clips.setdefault((d["participant"],d["activity"]),[]).append(c)
        rng=np.random.default_rng(seed)
        base=T/seq_len; mn=max(1,int(0.5*base))
        self.samples=[]
        for p in participants:
            for ci,seq in enumerate(self.class_list):
                if any((p,a) not in clips for a in seq): continue
                for _ in range(samples_per_class):
                    chosen=[clips[(p,a)][rng.integers(len(clips[(p,a)]))] for a in seq]
                    parts_len=rng.multinomial(T-seq_len*mn,[1/seq_len]*seq_len)+mn
                    self.samples.append((ci, chosen, parts_len.tolist()))
        self.classes=list(range(len(self.class_list)))
    def __len__(self): return len(self.samples)
    def _build(self, chosen, flens):
        segs=[]
        for path,F in zip(chosen,flens):
            x,y,p,t=load_events(path)
            ws,we=self.aw_fn(t,self.W_us)
            segs.append(make_frames(x,y,p,t,ws,we,F,self.H,self.W,self.crop))
        return np.concatenate(segs,axis=0).astype(np.float32)
    def _aug(self, fr):
        # miroir horizontal uniquement (train only)
        if torch.rand(1).item() < 0.5:
            fr = torch.flip(fr, dims=[3])
        return fr
    def __getitem__(self,i):
        ci,chosen,flens=self.samples[i]
        key=self.tag+"|"+"|".join(Path(c).stem for c in chosen)+"|"+"-".join(map(str,flens))
        h=hashlib.md5(key.encode()).hexdigest()[:16]
        cp=self.cache_dir/f"{self.tag}_{ci}_{h}.npz"
        if cp.exists(): fr=np.load(cp)["frames"]
        else:
            fr=self._build(chosen,flens); np.savez_compressed(cp,frames=fr)
        fr=torch.from_numpy(fr)
        if self.augment: fr=self._aug(fr)
        return fr, ci
