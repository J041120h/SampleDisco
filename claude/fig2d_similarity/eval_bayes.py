"""Did GP-EI acquisition beat its own linspace seed points?

search_bayesian probes 5 equally-spaced alphas, then makes 10 Expected-
Improvement picks. If the running best never improves after eval 5, those 10
extra objective evaluations bought nothing and a plain 5-point grid would have
returned the same alpha for a third of the compute.
"""
import glob, os, re, sys
R = "/dcs07/hongkai/data/harry/result"
rows = []
for f in sorted(glob.glob(f"{R}/**/autotune_record.txt", recursive=True)):
    txt = open(f, errors="replace").read()
    i = txt.find("Full chronological trace")
    if i < 0:
        continue
    steps = []
    for line in txt[i:].split("\n"):
        p = line.split()
        if len(p) == 3:
            try:
                steps.append((int(p[0]), float(p[1]), float(p[2])))
            except ValueError:
                pass
    if len(steps) < 6:
        continue
    steps.sort()
    seed = steps[:5]
    seed_best = max(s[2] for s in seed)
    seed_a = [s[1] for s in seed if s[2] == seed_best][0]
    all_best = max(s[2] for s in steps)
    all_a = [s[1] for s in steps if s[2] == all_best][0]
    gain = (all_best - seed_best) / abs(seed_best) * 100 if seed_best else float("nan")
    name = os.path.relpath(f, R).replace("/sample_embedding/autotune_record.txt", "")
    rows.append((name, len(steps), seed_a, seed_best, all_a, all_best, gain))

print(f"{'dataset':<50}{'n':>3}{'seed a':>9}{'seed sc':>9}{'EI a':>9}{'EI sc':>9}{'gain%':>8}")
print("-" * 97)
for r in rows:
    print(f"{r[0][-49:]:<50}{r[1]:>3}{r[2]:>9.3f}{r[3]:>9.4f}{r[4]:>9.3f}{r[5]:>9.4f}{r[6]:>8.2f}")
if rows:
    g = sorted(r[6] for r in rows)
    z = sum(1 for x in g if x < 1e-9)
    print("-" * 97)
    print(f"n={len(rows)} runs   EI gained NOTHING in {z}/{len(rows)}   "
          f"median gain {g[len(g)//2]:.2f}%   max {g[-1]:.2f}%")
