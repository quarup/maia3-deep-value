"""The value head averaged k moves ahead, for k = 0, 1, 2, ..., next to the deep value.

V_k of a board state is the average of the value head over the board states k plies later, each weighted by
the probability the policy gives the moves that lead to it (a game that ends earlier keeps its result). V_0 is
the value head alone and V_1 is one move ahead with every legal move. All are for the player to move at the
sampled board state, at the two players' real ratings.

  * V_0, V_1, V_2 are exact: every legal move, then every reply.
  * V_3 ... V_K are estimated by playing --rollouts lines on from the board states two plies ahead, each chosen
    with its exact probability, sampling every further move from the policy. The estimate is unbiased; its
    standard error is kept per board state. As a check the same lines also give an estimate of V_2 ("mc2"
    is the exact V_2 resampled: the mean over the sampled starting board states).

The board states are the ones of evals/tail_deep.py (same games, same plies), so each also has the deep value
of every legal move. Rows: {"game", "ply", "white_elo", "black_elo", "exact": [V0, V1, V2], "mc2", "mc": [V3..VK],
"se": [...], "deep": {eps: value}, "evals"}.

    python evals/lookahead.py --positions 300 --device mps
"""
import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia, scalar, terminal_wdl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--rollouts", type=int, default=500)
ap.add_argument("--device", default="cpu")
ap.add_argument("--shards", default="0/1")
ap.add_argument("--out", default="evals/results/lookahead")
a = ap.parse_args()
mine, of = a.shards.split("/")
mine, of = {int(x) for x in mine.split(",")}, int(of)
games = [json.loads(l) for l in open("evals/data/games.jsonl")]
picks = random.Random(23).sample(games, a.positions)
Path(a.out).mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["game"] for f in Path(a.out).glob("*.jsonl") if " " not in f.name for l in open(f)}
out = Path(a.out) / f"shard-{a.shards.replace('/', '-of-').replace(',', '_')}.jsonl"
maia = Maia(device=a.device)
EPS = (0.002, 0.01, 0.05, 0.2)
t0, n = time.time(), 0
for i, g in enumerate(picks):
    if i % of not in mine or g["id"] in done:
        continue
    start = maia.evals
    moves = g["moves"].split()
    ply = random.Random(g["id"] + "tail").randrange(8, len(moves))
    root = chess.Board()
    for u in moves[:ply]:
        root.push_uci(u)
    we, be = g["white_elo"], g["black_elo"]
    rng = np.random.default_rng(zlib.crc32(f"{g['id']}/{ply}".encode()))
    mover = root.turn

    def read(boards):
        """(value for the root's mover, policy or None) per board; a finished game has its result."""
        se = [we if b.turn == chess.WHITE else be for b in boards]
        oe = [be if b.turn == chess.WHITE else we for b in boards]
        res = []
        for b, e in zip(boards, maia.evaluate(boards, se, oe)):
            if e is None:
                w = scalar(terminal_wdl(b))
                res.append((w if mover == chess.WHITE else 1 - w, None))
            else:
                res.append((e.expected if b.turn == mover else 1 - e.expected, e.policy))
        return res

    # exact: level 0, 1, 2 as lists of (board, path probability, value, policy)
    level = [(root, 1.0) + read([root])[0]]
    exact = [level[0][2]]
    for _ in range(2):
        nxt, keep = [], []
        for b, q, v, pol in level:
            if pol is None:
                keep.append((b, q, v, None))
                continue
            for u, p in pol:
                nb = b.copy()
                nb.push_uci(u)
                nxt.append((nb, q * p))
        level = keep + [(b, q) + r for (b, q), r in zip(nxt, read([b for b, _ in nxt]))]
        exact.append(sum(q * v for _, q, v, _ in level))

    # lines played on from level 2
    q = np.array([x[1] for x in level])
    idx = rng.choice(len(level), size=a.rollouts, p=q / q.sum())
    lines = [[level[j][0], level[j][2], level[j][3]] for j in idx]          # board, value, policy
    mc2 = float(np.mean([l[1] for l in lines]))
    mc, se = [], []
    for _ in range(3, a.depth + 1):
        live = [l for l in lines if l[2] is not None]
        for l in live:
            ucis, ps = zip(*l[2])
            ps = np.array(ps)
            nb = l[0].copy()
            nb.push_uci(ucis[rng.choice(len(ucis), p=ps / ps.sum())])
            l[0] = nb
        for l, (v, pol) in zip(live, read([l[0] for l in live])):
            l[1], l[2] = v, pol
        vals = np.array([l[1] for l in lines])
        mc.append(float(vals.mean()))
        se.append(float(vals.std(ddof=1) / len(vals) ** 0.5))

    est = DeepValue(maia, eps=EPS[0]).value(root, we, be, also=[(e, 40) for e in EPS[1:]])
    side = lambda wdl: scalar(wdl) if mover == chess.WHITE else 1 - scalar(wdl)
    deep = {str(EPS[0]): round(side(est.wdl), 5)}
    deep.update({str(e): round(side(est.also[(e, 40)]), 5) for e in EPS[1:]})
    row = {"game": g["id"], "ply": ply, "white_elo": we, "black_elo": be, "result": g.get("result"),
           "white_to_move": mover == chess.WHITE, "exact": [round(x, 5) for x in exact], "mc2": round(mc2, 5),
           "mc": [round(x, 5) for x in mc], "se": [round(x, 5) for x in se], "deep": deep, "evals": maia.evals - start}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    n += 1
    print(f"{n} positions  {maia.evals} evals  {maia.evals / (time.time() - t0):.0f} evals/s  {(time.time() - t0) / 60:.1f} min", flush=True)
