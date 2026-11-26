"""How much are the moves a node leaves out worth when each is valued by its own deep traversal?

Like evals/tail_value.py, but every legal move of each sampled board state gets a deep value ("eps only"
tree at --eps below the move) next to its value-head value. Rows keep every move, so any curve can be
fitted afterwards: {"game", "ply", "head", "deep", "p": [...], "v1": [...], "vd": [...]}, values for the mover
("head" and "deep" are the board state's own value head and deep value; v1 and vd the same after each move).

    python evals/tail_deep.py --positions 300 --device mps --shards 1,2/3
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia, scalar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--device", default="cpu")
ap.add_argument("--shards", default="0/1")
ap.add_argument("--out", default="evals/results/tail_deep")
a = ap.parse_args()
mine, of = a.shards.split("/")
mine, of = {int(x) for x in mine.split(",")}, int(of)
games = [json.loads(l) for l in open("evals/data/games.jsonl")]
picks = random.Random(23).sample(games, a.positions)
Path(a.out).mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["game"] for f in Path(a.out).glob("*.jsonl") for l in open(f)}
out = Path(a.out) / f"shard-{a.shards.replace('/', '-of-').replace(',', '_')}.jsonl"
maia = Maia(device=a.device)
t0, n = time.time(), 0
for i, g in enumerate(picks):
    if i % of not in mine or g["id"] in done:
        continue
    moves = g["moves"].split()
    ply = random.Random(g["id"] + "tail").randrange(8, len(moves))
    b = chess.Board()
    for u in moves[:ply]:
        b.push_uci(u)
    we, be = g["white_elo"], g["black_elo"]
    white = b.turn == chess.WHITE
    e = maia.evaluate([b], we if white else be, be if white else we)[0]
    kids = []
    for uci, _ in e.policy:
        nb = b.copy()
        nb.push_uci(uci)
        kids.append((nb, we, be))
    deep = DeepValue(maia, eps=a.eps)
    est = deep.value_many(kids)
    mover = lambda wdl: scalar(wdl) if white else 1.0 - scalar(wdl)
    root = DeepValue(maia, eps=a.eps).value(b, we, be)          # the board state's own deep value, same eps
    row = {"game": g["id"], "ply": ply, "head": round(e.expected, 5), "deep": round(mover(root.wdl), 5),
           "p": [round(p, 7) for _, p in e.policy],
           "v1": [round(mover(x.zero_hop), 5) for x in est], "vd": [round(mover(x.wdl), 5) for x in est],
           "evals": deep.evals}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    n += 1
    print(f"{n} positions  {deep.evals} evals  {(time.time() - t0) / 60:.1f} min", flush=True)
