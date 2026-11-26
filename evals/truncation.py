"""What do the limits on moves per node cost, and could `eps` be the only setting? evals/results/truncation.jsonl

On a sample of board states, compares four ways of choosing which moves to follow from a node, each at
several thresholds, against a reference that follows every legal move at a fine threshold:

  default      the likeliest moves up to 97% of the policy, at most 8, renormalised
  all moves    every legal move (no limit, nothing to renormalise)
  no renorm    the default's moves with the policy's own weights; the moves left out count with the node's
               value-head value
  eps only     a move is followed only if its own path probability is at least eps; the moves left out
               count with the node's value-head value. No other setting.
  eps only, renormalised   the same moves, with their weights renormalised instead

For each: the mean gap to the reference (points of value) and the model evaluations used.

    python evals/truncation.py --positions 40 --device mps
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import Maia  # noqa: E402
from variants import DeepValue  # noqa: E402

VARIANTS = {
    "default": dict(),
    "all moves": dict(mass=1.0, max_children=None),
    "no renorm": dict(renormalise=False),
    "eps only": dict(mass=1.0, max_children=None, renormalise=False, strict=True),
    "eps only, renormalised": dict(mass=1.0, max_children=None, renormalise=True, strict=True),
}
EPS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002)
FINER = (0.001, 0.0005, 0.0002)        # for the two "eps only" variants, whose trees are much smaller


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--ref-eps", type=float, default=0.002)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--out", default="evals/results/truncation.jsonl")
    a = ap.parse_args()
    games = [json.loads(l) for l in open("evals/data/games.jsonl")]
    rng = random.Random(7)
    picks = rng.sample(games, a.positions)
    done = {json.loads(l)["game"] for l in open(a.out)} if Path(a.out).exists() else set()
    maia = Maia(a.model, device=a.device)
    for g in picks:
        if g["id"] in done:
            continue
        moves = g["moves"].split()
        ply = random.Random(g["id"]).randrange(8, len(moves))
        b = chess.Board()
        for u in moves[:ply]:
            b.push_uci(u)
        we, be = g["white_elo"], g["black_elo"]
        ref = DeepValue(maia, eps=a.ref_eps, mass=1.0, max_children=None)
        r = ref.value(b, we, be)
        row = {"game": g["id"], "ply": ply, "ref": round(r.value, 5), "ref_nodes": r.nodes, "zero_hop": round(float(r.zero_hop[0] + r.zero_hop[1] / 2), 5), "variants": {}}
        for name, kw in VARIANTS.items():
            for eps in EPS + (FINER if name.startswith("eps only") else ()):
                d = DeepValue(maia, eps=eps, **kw)
                d.cache = ref.cache                  # evaluations are shared: only the tree differs
                before = len(ref.cache)
                v = d.value(b, we, be)
                row["variants"][f"{name}|{eps:g}"] = {"value": round(v.value, 5), "nodes": v.nodes, "new": len(ref.cache) - before}
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(g["id"], ply, "ref", row["ref"], row["ref_nodes"], {k: (v["value"], v["nodes"]) for k, v in row["variants"].items() if k.endswith("|0.01")}, flush=True)


if __name__ == "__main__":
    main()
