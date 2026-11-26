"""Stockfish's evaluation of the board states charted as the example (EXAMPLE in evals/consistency.py).

Only a check that the example is a level game: no Stockfish number enters any estimate or metric. Needs a
`stockfish` binary on the PATH. Writes evals/results/example_stockfish.json: {"engine", "depth", "game",
"pawns": {ply: evaluation for White}}.

    python evals/example_stockfish.py --depth 24
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consistency import EXAMPLE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=24)
a = ap.parse_args()
gid, first, last = EXAMPLE
game = next(g for g in map(json.loads, open("evals/data/games.jsonl")) if g["id"] == gid)
engine = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
board, pawns = chess.Board(), {}
for ply, uci in enumerate(game["moves"].split()):
    if first <= ply < last:
        score = engine.analyse(board, chess.engine.Limit(depth=a.depth))["score"].white()
        pawns[ply] = round(score.score(mate_score=10000) / 100, 2)
        print(ply, pawns[ply], flush=True)
    board.push_uci(uci)
out = {"engine": engine.id.get("name"), "depth": a.depth, "game": gid, "pawns": pawns}
engine.quit()
Path("evals/results/example_stockfish.json").write_text(json.dumps(out, indent=1) + "\n")
