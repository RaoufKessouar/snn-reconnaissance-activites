import re, statistics as st
from pathlib import Path

log = Path(__file__).resolve().parent / "train_chain_T40aug.log"
pat = re.compile(
    r"Epoch : (\d+) \| Train Loss : ([\d.]+) \| Train Accuracy : ([\d.]+) \| "
    r"Val\(std\) Loss : ([\d.]+) \| Val\(std\) Accuracy : ([\d.]+) \| "
    r"Val\(ada\) Loss : ([\d.]+) \| Val\(ada\) Accuracy : ([\d.]+)")
rows = []
for line in open(log):
    m = pat.search(line)
    if m:
        g = m.groups()
        rows.append((int(g[0]), float(g[1]), float(g[2]), float(g[3]), float(g[4]), float(g[5]), float(g[6])))

print(f"{'ep':>3} {'trLoss':>7} {'trAcc':>6} {'vsLoss':>7} {'vsAcc':>6} {'vaLoss':>7} {'vaAcc':>6}")
for r in rows:
    print(f"{r[0]:3d} {r[1]:7.3f} {r[2]:6.3f} {r[3]:7.3f} {r[4]:6.3f} {r[5]:7.3f} {r[6]:6.3f}")

def stats(name, x):
    imax = x.index(max(x))
    print(f"{name}: max={max(x):.3f} (ep{rows[imax][0]})  min={min(x):.3f}  "
          f"mean={st.mean(x):.3f}  std_global={st.pstdev(x):.3f}  std_20dern={st.pstdev(x[-20:]):.3f}")

ta  = [r[2] for r in rows]
vsa = [r[4] for r in rows]
vaa = [r[6] for r in rows]
print("\n--- fluctuations (accuracy) ---")
stats("train   ", ta)
stats("val_std ", vsa)
stats("val_ada ", vaa)

# amplitude des sauts epoch-a-epoch sur la val
def jumps(x): return st.mean([abs(x[i]-x[i-1]) for i in range(1, len(x))])
print(f"\nsaut moyen epoch-a-epoch : val_std={jumps(vsa):.3f}  val_ada={jumps(vaa):.3f}")
print(f"nb epochs : {len(rows)}")
