import numpy as np, pandas as pd

def load_events(path, max_events=None):
    with open(path) as fh:
        first = fh.readline().strip()
    toks = [t.strip() for t in first.split(',')]
    has_header = not all(t.lstrip('-').replace('.', '', 1).isdigit() for t in toks if t)
    df = pd.read_csv(path, header=0 if has_header else None,
                     names=None if has_header else ['x', 'y', 'p', 't'], nrows=max_events)
    ren = {df.columns[i]: n for i, n in enumerate(['x', 'y', 'p', 't'])}
    df = df.rename(columns=ren)
    # robustesse : forcer numerique et jeter les lignes corrompues
    for c in ['x', 'y', 'p', 't']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['x', 'y', 'p', 't'])
    return (df['x'].to_numpy(np.int32), df['y'].to_numpy(np.int32),
            df['p'].to_numpy(np.int8), df['t'].to_numpy(np.int64))

def to_frames(x, y, p, t, T, H=480, W=640, ds=1):
    Hs, Ws = H // ds, W // ds
    xi = np.clip(x // ds, 0, Ws - 1); yi = np.clip(y // ds, 0, Hs - 1)
    t0, t1 = t.min(), t.max(); span = max(int(t1 - t0), 1)
    idx = np.clip(((t - t0).astype(np.int64) * T) // span, 0, T - 1)
    pc = np.clip(p.astype(np.int64), 0, 1)
    frames = np.zeros((T, 2, Hs, Ws), dtype=np.float32)
    np.add.at(frames, (idx, pc, yi, xi), 1.0)
    return frames
