"""Fit the tail discount from evals/results/tail_deep: left-out moves valued one move later (v1) and deep (vd)."""
import json
from pathlib import Path

import numpy as np

test = {json.loads(l)["game"] for l in open("evals/results/truncation.jsonl")}      # kept out of the fit
rows = [json.loads(l) for f in sorted(Path("evals/results/tail_deep").glob("*.jsonl")) for l in open(f)]
rows = [r for r in rows if r["game"] not in test]
CUTS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
print(f"{len(rows)} board states. Points for the player to move, relative to the board state's value head.\n")
print("| Cut-off | Board states | Weight left out | Left-out moves, one move later | Left-out moves, deep | All moves, one move later | All moves, deep |")
print("| --- | --- | --- | --- | --- | --- | --- |")
fit = {"v1": [], "vd": []}
for cut in CUTS:
    d = {"v1": [], "vd": []}
    al = {"v1": [], "vd": []}
    w = []
    for r in rows:
        p = np.array(r["p"])
        m = p < cut
        if m.all() or not m.any():
            continue
        w.append(p[m].sum())
        for k in d:
            v = np.array(r[k])
            d[k].append((p[m] * v[m]).sum() / p[m].sum() - r["head"])
            al[k].append((p * v).sum() / p.sum() - r["head"])
    ci = lambda x: 196 * np.std(x, ddof=1) / len(x) ** 0.5
    print(f"| {cut:g} | {len(w)} | {np.mean(w) * 100:.1f}% | {np.mean(d['v1']) * 100:+.1f} ± {ci(d['v1']):.1f} | {np.mean(d['vd']) * 100:+.1f} ± {ci(d['vd']):.1f} | "
          f"{np.mean(al['v1']) * 100:+.1f} | {np.mean(al['vd']) * 100:+.1f} ± {ci(al['vd']):.1f} |")
    for k in d:
        fit[k].append(-np.mean(d[k]) * 100)
x = np.log10(1 / np.array(CUTS))
for k, name in (("v1", "one move later"), ("vd", "deep")):
    b, a = np.polyfit(x, fit[k], 1)
    res = np.sqrt(np.mean((a + b * x - np.array(fit[k])) ** 2))
    print(f"\n{name}: discount = {a:.2f} + {b:.2f} * log10(1 / cut-off)   (rms residual {res:.2f})")
