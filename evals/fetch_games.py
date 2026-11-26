"""Sample evaluation games from the Lichess open database (CC0): evals/data/games.jsonl.

Streams the start of one monthly dump (the file is tens of GB; only what is needed is downloaded) and keeps
rated blitz games between humans, the kind Maia-3 was trained on, from a month after its training data
ends (July 2025). Games are binned by the players' average rating and each bin is filled to the same size,
so weak and strong play count equally.

    python evals/fetch_games.py --month 2026-10 --per-bin 250
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
from pathlib import Path

import chess.pgn
import zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
BINS = list(range(800, 2400, 200))          # 800-999, ..., 2200-2399: eight bins
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")


def keep(h: dict) -> bool:
    if not h.get("Event", "").startswith("Rated Blitz"):
        return False
    if h.get("Termination") not in ("Normal", "Time forfeit") or h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    if "BOT" in (h.get("WhiteTitle"), h.get("BlackTitle")):
        return False
    return h.get("WhiteElo", "?").isdigit() and h.get("BlackElo", "?").isdigit()


def games(lines):
    """(headers, movetext) of each game in a PGN stream, without parsing the moves."""
    heads, body = {}, []
    for line in lines:
        if line.startswith("["):
            if body:
                yield heads, " ".join(body)
                heads, body = {}, []
            m = re.match(r'\[(\w+) "(.*)"\]', line)
            if m:
                heads[m[1]] = m[2]
        elif line.strip():
            body.append(line.strip())
    if body:
        yield heads, " ".join(body)


def moves_of(movetext: str) -> tuple[list[str], list[int]]:
    """Moves in UCI, and the mover's clock in seconds after each move (-1 where the PGN has none)."""
    moves, clocks = [], []
    for node in chess.pgn.read_game(io.StringIO(movetext)).mainline():
        moves.append(node.move.uci())
        m = CLK.search(node.comment)
        clocks.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else -1)
    return moves, clocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default="2026-10")
    ap.add_argument("--per-bin", type=int, default=250)
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--max-rating-gap", type=int, default=400)
    ap.add_argument("--out", default="evals/data/games.jsonl")
    a = ap.parse_args()

    got: dict[int, list[dict]] = {b: [] for b in BINS}
    seen = 0
    with urllib.request.urlopen(URL.format(month=a.month), timeout=60) as r:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r), encoding="utf-8")
        for h, movetext in games(text):
            seen += 1
            if seen % 50000 == 0:
                print(f"{seen} games read; bins {[len(v) for v in got.values()]}", flush=True)
            if not keep(h):
                continue
            we, be = int(h["WhiteElo"]), int(h["BlackElo"])
            b = (we + be) // 2 // 200 * 200
            if b not in got or len(got[b]) >= a.per_bin or abs(we - be) > a.max_rating_gap:
                continue
            moves, clocks = moves_of(movetext)
            if len(moves) < a.min_plies or -1 in clocks:
                continue
            got[b].append({"id": h["Site"].rsplit("/", 1)[-1], "white_elo": we, "black_elo": be,
                           "result": h["Result"], "termination": h["Termination"],
                           "time_control": h.get("TimeControl", ""), "date": h.get("UTCDate", ""),
                           "moves": " ".join(moves), "clocks": clocks})
            if all(len(v) >= a.per_bin for v in got.values()):
                break
    print(f"{seen} games read; bins {[len(v) for v in got.values()]}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [g for b in BINS for g in got[b]]
    out.write_text("".join(json.dumps(g) + "\n" for g in rows))
    print(f"{len(rows)} games -> {out}")


if __name__ == "__main__":
    main()
