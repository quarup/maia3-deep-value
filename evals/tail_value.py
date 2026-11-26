"""How much are the moves a node leaves out really worth, next to that node's own value head?

For sampled board states, every legal move is evaluated. Moves under a policy cut-off are "the tail". From
the mover's side: the tail's policy-weighted mean value (value head one move later), the same for the moves
kept, and the board state's own value head.

    python evals/tail_value.py --positions 400
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import Maia  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=400)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
games = [json.loads(l) for l in open("evals/data/games.jsonl")]
rng = random.Random(11)
maia = Maia(device=a.device)
rows = []
for g in rng.sample(games, a.positions):
    moves = g["moves"].split()
    ply = rng.randrange(8, len(moves))
    b = chess.Board()
    for u in moves[:ply]:
        b.push_uci(u)
    me, opp = (g["white_elo"], g["black_elo"]) if b.turn else (g["black_elo"], g["white_elo"])
    e = maia.evaluate([b], me, opp)[0]
    kids = []
    for uci, p in e.policy:
        nb = b.copy()
        nb.push_uci(uci)
        kids.append(nb)
    ev = maia.evaluate(kids, opp, me)
    val = []                       # each move's value for the mover of the root
    for nb, x in zip(kids, ev):
        if x is None:
            val.append(1.0 if nb.is_checkmate() else 0.5)
        else:
            val.append(1.0 - x.expected)
    rows.append({"head": e.expected, "p": [p for _, p in e.policy], "v": val})
json.dump(rows, open("evals/results/tail_value.json", "w"))
print(f"{len(rows)} board states. Values are for the player to move, in points.\n")
print("| Tail = moves with policy under | Tail weight | Value head here | Kept moves | Tail moves | Tail minus value head | All moves (one-hop) minus value head |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for cut in (0.2, 0.1, 0.05, 0.02, 0.01):
    hs, ks, ts, ws, ones = [], [], [], [], []
    for r in rows:
        kept = [(p, v) for p, v in zip(r["p"], r["v"]) if p >= cut]
        tail = [(p, v) for p, v in zip(r["p"], r["v"]) if p < cut]
        if not kept or not tail:
            continue
        wk, wt = sum(p for p, _ in kept), sum(p for p, _ in tail)
        hs.append(r["head"]); ws.append(wt)
        ks.append(sum(p * v for p, v in kept) / wk); ts.append(sum(p * v for p, v in tail) / wt)
        ones.append(sum(p * v for p, v in zip(r["p"], r["v"])))
    d = [t - h for t, h in zip(ts, hs)]
    se = st.stdev(d) / len(d) ** 0.5
    print(f"| {cut:g} | {st.mean(ws) * 100:.0f}% | {st.mean(hs) * 100:.1f} | {st.mean(ks) * 100:.1f} | {st.mean(ts) * 100:.1f} | "
          f"{st.mean(d) * 100:+.1f} ± {1.96 * se * 100:.1f} | {st.mean(o - h for o, h in zip(ones, hs)) * 100:+.1f} |")
