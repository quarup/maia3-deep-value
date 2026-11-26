"""The deep value of every board state of a game, or of one position.

    deep-value game.pgn                          # every position, at the players' own ratings
    deep-value game.pgn --plies 24,25 --eps 0.01
    deep-value game.pgn --ratings 2000           # the same game as two 2000s would play it on
    deep-value --fen "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1" --ratings 1500

With -o the result is written as JSON after every position, and a stopped run resumes from the file.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import chess
import chess.pgn

from . import tree
from .engine import Maia, default_threads


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="?", help="a PGN file (its first game), or - for stdin")
    ap.add_argument("--fen", help="one position instead of a game (no move history)")
    ap.add_argument("--ratings", help="one rating for both players, or WHITE,BLACK (default: the PGN's)")
    ap.add_argument("--plies", help="comma-separated plies (default: every board state of the game)")
    ap.add_argument("--eps", type=float, default=tree.EPS, help=f"path-probability threshold (default {tree.EPS})")
    ap.add_argument("--max-depth", type=int, default=tree.MAX_DEPTH)
    ap.add_argument("--model", default="maia3-79m", help="maia3-5m, maia3-23m or maia3-79m")
    ap.add_argument("-o", "--out", help="write JSON here")
    a = ap.parse_args(argv)
    if bool(a.pgn) == bool(a.fen):
        ap.error("give a PGN file or --fen")

    elos = [int(x) for x in a.ratings.split(",")] if a.ratings else None
    if a.fen:
        if not elos:
            ap.error("--fen needs --ratings")
        boards, moves, tag = [chess.Board(a.fen)], [], a.fen
    else:
        text = sys.stdin.read() if a.pgn == "-" else Path(a.pgn).read_text()
        game = chess.pgn.read_game(io.StringIO(text))
        moves = list(game.mainline_moves())
        b = game.board()
        boards = [b.copy()]
        for m in moves:
            b.push(m)
            boards.append(b.copy())
        elos = elos or [int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])]
        tag = game.headers.get("Site", a.pgn)
    we, be = elos[0], elos[-1]

    out = Path(a.out) if a.out else None
    doc = json.loads(out.read_text()) if out and out.exists() else {}
    head = {"game": tag, "model": a.model, "ratings": [we, be], "eps": a.eps, "max_depth": a.max_depth}
    if {k: doc.get(k) for k in head} != head:
        doc = {**head, "what": "Before each ply's move: the deep value for White, as (win, draw, loss) and as one number "
               "(win + draw / 2), by following "
               "Maia-3's policy for both sides: a move is followed when its path's probability from that position "
               "is at least eps, the moves left out count with their board state's value head, mate and draws "
               "are exact. zero_hop is the value head alone, one_hop "
               "its average one move later; moves are the likeliest moves with the deep value after each.",
               "plies": {}}
    deep = tree.DeepValue(Maia(a.model, threads=default_threads()), eps=a.eps, max_depth=a.max_depth)
    want = [int(x) for x in a.plies.split(",")] if a.plies else range(len(boards))
    for i in want:
        if str(i) in doc["plies"]:
            continue
        t0 = time.time()
        b = boards[i]
        r = deep.value(b, we, be).to_json()
        label = "end"
        if i < len(moves):
            san = b.san(moves[i])
            hit = next((m for m in r["moves"] if m["san"] == san), None)
            r["played"] = {"san": san, "value": hit["value"] if hit else round(deep.after(b, san, we, be).value, 4)}
            label = f"{b.fullmove_number}{'.' if b.turn else '...'}{san}"
        doc["plies"][str(i)] = r
        doc["seconds"] = round(doc.get("seconds", 0) + time.time() - t0, 1)
        if out:
            out.write_text(json.dumps(doc, indent=1))
        w, d, l = r["wdl"]
        print(f"ply {i:3d} {label:9s} win {w:.3f} draw {d:.3f} loss {l:.3f} (White) | deep {r['value']:.3f}  "
              f"one-hop {tree.scalar(r['one_hop']):.3f}  zero-hop {tree.scalar(r['zero_hop']):.3f} | "
              f"depth {r['depth']:2d}  nodes {r['nodes']:5d}", flush=True)


if __name__ == "__main__":
    main()
