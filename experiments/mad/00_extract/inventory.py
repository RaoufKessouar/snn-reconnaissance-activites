import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from mad_naming import parse, ACTIVITY_NAMES
from mad_paths import EXTRACT_DIR

csvs = sorted(EXTRACT_DIR.rglob("*.csv"))
print(f"CSV trouves : {len(csvs)}")
bad = [c.name for c in csvs if parse(c.name) is None]
print(f"non parses  : {len(bad)}  {bad[:5]}")

by_act = defaultdict(lambda: [0, 0])
parts, subs = set(), set()
for c in csvs:
    d = parse(c.name)
    if not d:
        continue
    by_act[d["activity"]][0] += 1
    by_act[d["activity"]][1] += c.stat().st_size
    parts.add(d["participant"]); subs.add((d["activity"], d["sub"]))

print(f"participants : {sorted(parts)}")
print(f"sous-activites distinctes : {len(subs)} (attendu 29)")
print("\nact | nom               |  n | taille_moy(MB)")
for a in sorted(by_act):
    n, b = by_act[a]
    print(f"{a:2d}  | {ACTIVITY_NAMES[a]:16s} | {n:2d} | {b/n/1e6:6.1f}")
