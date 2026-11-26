"""Every board state of whole games, valued every way: evals/results/<name>/*.jsonl.

evals/run_eval.py samples five board states per game, which says nothing about how an estimate behaves from
one move to the next. This takes the first `--games` games (same balanced order) and values every ply, so
that consecutive values can be compared (evals/consistency.py). One row per board state.

    python evals/run_games.py games --games 60 --eps 0.002 --device mps --shards 0,1/3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia  # noqa: E402
from run_eval import ALSO, SCORE, balanced  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shards", default="0/1")
    ap.add_argument("--data", default="evals/data/games.jsonl")
    a = ap.parse_args()
    mine, of = a.shards.split("/")
    mine, of = {int(x) for x in mine.split(",")}, int(of)
    games = balanced([json.loads(l) for l in open(a.data)], a.games)
    games = [g for i, g in enumerate(games) if i % of in mine]
    out_dir = Path("evals/results") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard-{'_'.join(map(str, sorted(mine)))}-of-{of}.jsonl"
    (out_dir / "run.json").write_text(json.dumps({"eps": a.eps, "model": a.model, "games": a.games, "every_ply": True}))
    done = {json.loads(l)["game"] for f in out_dir.glob("*.jsonl") for l in open(f)}
    games = [g for g in games if g["id"] not in done]
    also = [(e, 40) for e in ALSO if e > a.eps]
    maia = Maia(a.model, device=a.device)
    r4 = lambda t: [round(float(x), 4) for x in t]
    t0 = time.time()
    for n, g in enumerate(games, 1):
        moves = g["moves"].split()
        b = chess.Board()
        boards = [b.copy()]
        for u in moves:
            b.push_uci(u)
            boards.append(b.copy())
        deep = DeepValue(maia, eps=a.eps)                       # one cache per game: its plies share most of their work
        est = deep.value_many([(x, g["white_elo"], g["black_elo"]) for x in boards[:-1]], also=also)
        with open(out, "a") as f:
            for i, (x, e) in enumerate(zip(boards, est)):
                f.write(json.dumps({
                    "game": g["id"], "ply": i, "plies": len(moves), "white_to_move": x.turn,
                    "white_elo": g["white_elo"], "black_elo": g["black_elo"], "result": SCORE[g["result"]],
                    "termination": g["termination"], "played": x.san(chess.Move.from_uci(moves[i])),
                    "zero_hop": r4(e.zero_hop), "one_hop": r4(e.one_hop), "deep": r4(e.wdl),
                    "eps": {f"{s[0]:g}": r4(v) for s, v in e.also.items()},
                    "moves": [{"san": m["san"], "p": m["p"], "wdl": r4(m["wdl"])} for m in e.moves],
                    "nodes": e.nodes, "depth": e.depth}) + "\n")
        dt = time.time() - t0
        print(f"{n}/{len(games)} games  {len(moves)} plies  {maia.evals} evals  {maia.evals / dt:.0f} evals/s  {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
