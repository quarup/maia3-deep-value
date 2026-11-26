"""Estimate sampled positions of the evaluation games every way at once: evals/results/<name>/*.jsonl.

For each game, `--positions` plies are drawn at random (seeded by the game id, from ply 8 on) and each
position is scored once with the tree at `--eps`. That one tree also gives the value head alone (zero-hop),
its average one move later (one-hop), and the tree at every looser threshold in `--also` and every depth
cap in `--depths`. One row per position is appended as it is done, so a stopped run resumes.

    python evals/run_eval.py main --games 2000 --eps 0.002
    python evals/run_eval.py main --games 2000 --eps 0.002 --shards 0,1,2/5 --device mps     # split across processes
    python evals/run_eval.py sweep --games 100 --eps 0.0005
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia  # noqa: E402

ALSO = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
DEPTHS = (2, 3, 4, 6, 8)
FIRST_PLY = 8
SCORE = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


def balanced(games: list[dict], n: int) -> list[dict]:
    """The first `n` games taking the rating bins in turn, so that any prefix is balanced."""
    bins: dict[int, list] = {}
    for g in games:
        bins.setdefault((g["white_elo"] + g["black_elo"]) // 2 // 200, []).append(g)
    out = []
    for row in zip(*bins.values()):
        out.extend(row)
    return out[:n]


def sampled_plies(game: dict, k: int) -> list[int]:
    n = len(game["moves"].split())
    return sorted(random.Random(game["id"]).sample(range(FIRST_PLY, n), min(k, n - FIRST_PLY)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the run: results go to evals/results/<name>/")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--positions", type=int, default=5, help="positions sampled per game")
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--model", default="maia3-79m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shards", default="0/1", help="which slices of the games this process takes, e.g. 0,1/5")
    ap.add_argument("--chunk", type=int, default=4, help="games scored together")
    ap.add_argument("--data", default="evals/data/games.jsonl")
    a = ap.parse_args()

    mine, of = a.shards.split("/")
    mine, of = {int(x) for x in mine.split(",")}, int(of)
    games = balanced([json.loads(l) for l in open(a.data)], a.games)
    games = [g for i, g in enumerate(games) if i % of in mine]
    out_dir = Path("evals/results") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard-{'_'.join(map(str, sorted(mine)))}-of-{of}.jsonl"
    (out_dir / "run.json").write_text(json.dumps({"eps": a.eps, "model": a.model, "games": a.games,
                                                   "positions_per_game": a.positions}))
    done = {json.loads(l)["game"] for f in out_dir.glob("*.jsonl") for l in open(f)}
    games = [g for g in games if g["id"] not in done]
    also = [(e, 40) for e in ALSO if e > a.eps] + [(a.eps, d) for d in DEPTHS]
    maia = Maia(a.model, device=a.device)
    print(f"{len(games)} games to do -> {out}", flush=True)

    t0, n_pos = time.time(), 0
    for s in range(0, len(games), a.chunk):
        deep = DeepValue(maia, eps=a.eps)     # a fresh cache per chunk: games share nothing
        jobs = []
        for g in games[s:s + a.chunk]:
            moves = g["moves"].split()
            plies = set(sampled_plies(g, a.positions))
            b = chess.Board()
            for i, u in enumerate(moves):
                if i in plies:
                    jobs.append((g, i, b.copy()))
                b.push_uci(u)
        scores = deep.value_many([(b, g["white_elo"], g["black_elo"]) for g, _, b in jobs], also=also)
        r4 = lambda t: [round(float(x), 4) for x in t]
        with open(out, "a") as f:
            for (g, i, b), sc in zip(jobs, scores):
                f.write(json.dumps({
                    "game": g["id"], "ply": i, "plies": len(g["moves"].split()), "white_to_move": b.turn,
                    "white_elo": g["white_elo"], "black_elo": g["black_elo"], "result": SCORE[g["result"]],
                    "termination": g["termination"], "clock": g["clocks"][i - 2] if i >= 2 else -1,
                    "zero_hop": r4(sc.zero_hop), "one_hop": r4(sc.one_hop), "deep": r4(sc.wdl),
                    "eps": {f"{e:g}": r4(v) for (e, d), v in sc.also.items() if d == 40},
                    "depth_cap": {str(d): r4(v) for (e, d), v in sc.also.items() if d != 40},
                    "eps_nodes": {f"{e:g}": n for (e, d), n in sc.also_nodes.items() if d == 40},
                    "depth_cap_nodes": {str(d): n for (e, d), n in sc.also_nodes.items() if d != 40},
                    "nodes": sc.nodes, "depth": sc.depth}) + "\n")
        n_pos += len(jobs)
        dt = time.time() - t0
        print(f"{s + a.chunk:5d}/{len(games)} games  {n_pos} positions  {maia.evals} evals  "
              f"{maia.evals / dt:.0f} evals/s  {maia.evals / max(n_pos, 1):.0f} evals/position  {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
