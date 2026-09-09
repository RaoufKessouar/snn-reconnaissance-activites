import sys, hashlib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mad_io import load_events
from mad_naming import parse
from mad_dataset import active_window, make_frames


class MADFallDataset(Dataset):
    """Sequences de seq_len activites tirees au hasard parmi 9 ; ~p_fall contiennent
    UNE chute a une position aleatoire. Labels PAR FRAME (detection + localisation)."""
    def __init__(self, extract_dir, cache_dir, participants,
                 activities=(1,2,3,4,5,6,7,8,9), fall_activity=9, seq_len=5,
                 T=40, W_s=3.0, H=128, W=128, crop=(120,520,80,440),
                 samples_per_participant=60, p_fall=0.5, seed=0, split_name="train"):
        self.T,self.W_s,self.H,self.W,self.crop = T,W_s,H,W,crop
        self.W_us=int(W_s*1e6)
        self.cache_dir=Path(cache_dir); self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.activities=list(activities); self.fall=fall_activity
        self.nonfall=[a for a in self.activities if a!=fall_activity]
        self.a2i={a:i for i,a in enumerate(self.activities)}
        self.fall_idx=self.a2i[fall_activity]
        self.seq_len=seq_len
        self.tag=f"fall_L{seq_len}_T{T}_W{W_s}_H{H}_{split_name}"
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
            avail_nonfall=[a for a in self.nonfall if (p,a) in clips]
            has_fall=(p,self.fall) in clips
            if len(avail_nonfall)<seq_len: continue
            for _ in range(samples_per_participant):
                want_fall = has_fall and (rng.random()<p_fall) and len(avail_nonfall)>=seq_len-1
                if want_fall:
                    others=list(rng.choice(avail_nonfall,size=seq_len-1,replace=False))
                    pos=int(rng.integers(seq_len))
                    seq=others[:pos]+[self.fall]+others[pos:]
                else:
                    seq=list(rng.choice(avail_nonfall,size=seq_len,replace=False))
                chosen=[clips[(p,a)][rng.integers(len(clips[(p,a)]))] for a in seq]
                flens=(rng.multinomial(T-seq_len*mn,[1/seq_len]*seq_len)+mn).tolist()
                self.samples.append((seq, chosen, flens))
    def __len__(self): return len(self.samples)
    def _build(self, chosen, flens):
        segs=[]
        for path,F in zip(chosen,flens):
            x,y,p,t=load_events(path)
            ws,we=active_window(t,self.W_us)
            segs.append(make_frames(x,y,p,t,ws,we,F,self.H,self.W,self.crop))
        return np.concatenate(segs,axis=0).astype(np.float32)
    def __getitem__(self,i):
        seq,chosen,flens=self.samples[i]
        key=self.tag+"|"+"|".join(Path(c).stem for c in chosen)+"|"+"-".join(map(str,flens))
        h=hashlib.md5(key.encode()).hexdigest()[:16]
        cp=self.cache_dir/f"{self.tag}_{h}.npz"
        if cp.exists(): fr=np.load(cp)["frames"]
        else:
            fr=self._build(chosen,flens); np.savez_compressed(cp,frames=fr)
        labels=np.concatenate([np.full(F,self.a2i[a],dtype=np.int64)
                               for a,F in zip(seq,flens)])
        return torch.from_numpy(fr), torch.from_numpy(labels)
