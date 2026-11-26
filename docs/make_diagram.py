"""Draw the tree behind one deep value as docs/img/tree.svg (the picture in the README), and print the
arithmetic of the example.

The position is a real one from the evaluation games (evals/data/games.jsonl): Black, rated 1365, is about to
take a bishop with the queen. To keep the picture small the tree is grown with a very loose threshold (eps
0.2: a move is followed when at least 20% of games from the root would reach it; the default is 0.002). The
moves a board state leaves out are drawn as one dashed box: they count with that board state's value head.

    python docs/make_diagram.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import chess
import chess.svg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia, scalar  # noqa: E402

GAME, PLY = "iG5QM5lW", 41
EPS = 0.2

SHOW_REST = 0.05       # a board state's left-out moves are drawn when they hold at least this much of its policy

INK, MUTED, FAINT, SURFACE, CARD = "#0b0b0b", "#52514e", "#c9c8c2", "#fcfcfb", "#ffffff"
FOLLOW, STOP, OVER = "#1baf7a", "#2a78d6", "#0b0b0b"
W, H, GAP, ROW = 124, 50, 14, 104
BOARD = 232


def pct(x: float) -> str:
    return f"{x * 100:.0f}%" if x >= 0.0095 else "<1%"


def main():
    game = next(g for g in map(json.loads, open("evals/data/games.jsonl")) if g["id"] == GAME)
    board = chess.Board()
    for u in game["moves"].split()[:PLY]:
        board.push_uci(u)
    elo = game["white_elo"], game["black_elo"]
    deep = DeepValue(Maia(), eps=EPS)
    nodes = deep.explain(board, *elo)
    score = deep.value(board, *elo)
    fine = DeepValue(deep.evaluate, eps=0.002).value(board, *elo)

    kids: dict[int, list[int]] = {}
    for i, n in enumerate(nodes):
        if n["parent"] is not None:
            kids.setdefault(n["parent"], []).append(i)
    # the drawn tree: every board state in the tree, and one dashed box per node for the moves it leaves out
    draw = [{**nodes[0], "i": 0, "parent": None}]
    order = [0]
    while order:
        d = order.pop(0)
        if "fold" in draw[d]:
            continue
        ks = kids.get(draw[d]["i"], [])
        for k in ks:
            draw.append({**nodes[k], "i": k, "parent": d})
            order.append(len(draw) - 1)
        if ks and draw[d]["rest"] >= SHOW_REST:
            p = draw[d]["rest"]
            draw.append({"parent": d, "fold": True, "p": p, "q": draw[d]["q"] * p, "depth": draw[d]["depth"] + 1,
                         "avg": scalar(draw[d]["head"]), "kind": "leaf"})
    dk: dict[int, list[int]] = {}
    for j, n in enumerate(draw):
        if n["parent"] is not None:
            dk.setdefault(n["parent"], []).append(j)
    # layout: each row is packed from the left, in the order of its parents, followed moves first; rows are
    # short because only a few lines are ever followed, so the picture stays narrow
    rows: dict[int, list[int]] = {}
    for j, n in enumerate(draw):
        rows.setdefault(n["depth"], []).append(j)
    draw[0]["x"] = 0.0
    for d in sorted(rows)[1:]:
        rows[d].sort(key=lambda j: (draw[draw[j]["parent"]]["x"], "fold" in draw[j], -draw[j]["p"]))
        for x, j in enumerate(rows[d]):
            draw[j]["x"] = float(x)
    label_x, left, top = 24 + BOARD + 36, 24 + BOARD + 36 + 126, 132
    for n in draw:
        n["cx"] = left + n["x"] * (W + GAP) + W / 2
        n["cy"] = top + n["depth"] * ROW + H / 2
    depth = max(n["depth"] for n in draw)
    width = max(int(left + max(len(r) for r in rows.values()) * (W + GAP) + 24), 1000)
    height = top + depth * ROW + H + 28

    mover = "White" if board.turn else "Black"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {width} {height}" '
         f'width="{width}" height="{height}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{width}" height="{height}" rx="10" fill="{SURFACE}"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">Deep value of one board state: follow the likely moves, apply the value head to the rest</text>',
         f'<text x="24" y="58" font-size="13" fill="{MUTED}">A real game, White rated {elo[0]}, Black {elo[1]}. Black is about to take a bishop with the queen; '
         f'the queen is then lost, or the king mated, on the g-file.</text>',
         f'<text x="24" y="76" font-size="13" fill="{MUTED}">For illustration this diagram uses a loose threshold, eps = {EPS} (a move is followed when its '
         f'path probability is at least {pct(EPS)}). The default is eps = 0.002 (0.2%).</text>']
    # the root position
    last = board.peek() if board.move_stack else None
    first = chess.Move.from_uci(game["moves"].split()[PLY])
    svg = chess.svg.board(board, size=BOARD, lastmove=last, coordinates=False,
                          arrows=[chess.svg.Arrow(first.from_square, first.to_square, color=FOLLOW)],
                          colors={"square light": "#eceae4", "square dark": "#b9b6ab", "square light lastmove": "#dfe6c8",
                                  "square dark lastmove": "#a9b48c"})
    inner = re.sub(r"^<svg[^>]*>", "", svg).rsplit("</svg>", 1)[0]
    vb = re.search(r'viewBox="([^"]+)"', svg)[1]
    o.append(f'<svg x="24" y="{top}" width="{BOARD}" height="{BOARD}" viewBox="{vb}">{inner}</svg>')
    ty = top + BOARD + 24
    o.append(f'<text x="24" y="{ty}" font-size="13" font-weight="700" fill="{INK}">Root board state, {mover} to move</text>')
    lines = [("Value for White (win + draw / 2)", None),
             ("Value head (zero-hop)", scalar(score.zero_hop)), ("One move ahead (one-hop)", scalar(score.one_hop)),
             (f"This tree (eps {EPS})", score.value),
             ("Default tree (eps 0.002)", fine.value)]
    for k, (name, v) in enumerate(lines):
        y = ty + 26 + k * 20
        if v is None:
            o.append(f'<text x="24" y="{y}" font-size="12.5" fill="{MUTED}">{name}:</text>')
            continue
        o.append(f'<text x="24" y="{y}" font-size="12.5" fill="{INK}">{name}</text>')
        o.append(f'<text x="{24 + BOARD}" y="{y}" font-size="12.5" font-weight="700" fill="{INK}" text-anchor="end">{pct(v)}</text>')
    o.append(f'<text x="24" y="{ty + 26 + len(lines) * 20 + 6}" font-size="12.5" fill="{MUTED}">The game: White won.</text>')
    # legend, under the numbers
    ly = ty + 26 + len(lines) * 20 + 40
    legend = [(FOLLOW, 2.2, "", ["Some of its moves are followed", f"(path probability at least {pct(EPS)})."]),
              *([(STOP, 1.6, "", [f"None of its moves reaches {pct(EPS)}:", "its value head is its value."])]
                if any(n["kind"] == "leaf" and "fold" not in n for n in draw) else []),
              (OVER, 1.6, "", ["The game is over: the exact outcome", "is used."]),
              (STOP, 1.2, "4 3", [f"Moves left out (path under {pct(EPS)}).", "They count with the value head", "of the box above. Not drawn", f"when under {pct(SHOW_REST)} of the policy."])]
    for col, sw, dash, text in legend:
        o.append(f'<rect x="24" y="{ly - 11}" width="22" height="14" rx="4" fill="{CARD}" stroke="{col}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        for k, t in enumerate(text):
            o.append(f'<text x="54" y="{ly + k * 17}" font-size="12" fill="{INK}">{t}</text>')
        ly += len(text) * 17 + 12
    o.append(f'<text x="24" y="{ly + 2}" font-size="12" fill="{MUTED}">Percentages on the lines: the policy.</text>')
    # row labels
    side = [mover + " to move", ("Black" if board.turn else "White") + " to move"]
    for d in range(depth + 1):
        y = top + d * ROW + H / 2
        row = "Root" if d == 0 else f"{d} {'move' if d == 1 else 'moves'} later"
        o.append(f'<text x="{label_x}" y="{y - 3:.0f}" font-size="12" font-weight="600" fill="{INK}">{row}</text>')
        last_row = all(draw[j]["kind"] == "over" for j in rows[d])
        o.append(f'<text x="{label_x}" y="{y + 13:.0f}" font-size="12" fill="{MUTED}">{"checkmate" if last_row else side[d % 2]}</text>')
    # edges first: width shows how likely the whole path is
    for n in draw:
        if n["parent"] is None:
            continue
        p = draw[n["parent"]]
        x1, y1, x2, y2 = p["cx"], p["cy"] + H / 2, n["cx"], n["cy"] - H / 2
        col = FAINT if "fold" in n else FOLLOW
        o.append(f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{y1 + 30:.1f} {x2:.1f},{y2 - 30:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{col}" '
                 f'stroke-width="{1.2 + 9 * n["q"]:.1f}" stroke-linecap="round"/>')
    for n in draw:
        if n["parent"] is None:
            continue
        o.append(f'<text x="{n["cx"] + 12:.1f}" y="{n["cy"] - H / 2 - 8:.1f}" font-size="11.5" fill="{MUTED}" '
                 f'paint-order="stroke" stroke="{SURFACE}" stroke-width="4">{pct(n["p"])}</text>')
        if "fold" not in n:
            o.append(f'<text x="{n["cx"] - 12:.1f}" y="{n["cy"] - H / 2 - 8:.1f}" font-size="11.5" font-weight="600" fill="{FOLLOW}" text-anchor="end" '
                     f'paint-order="stroke" stroke="{SURFACE}" stroke-width="4">path {pct(n["q"])}</text>')
    # nodes
    for n in draw:
        x, y = n["cx"] - W / 2, n["cy"] - H / 2
        if "fold" in n:
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{W}" height="{H}" rx="8" fill="{SURFACE}" stroke="{STOP}" stroke-width="1.2" stroke-dasharray="4 3"/>')
            o.append(f'<text x="{n["cx"]:.1f}" y="{n["cy"] - 4:.1f}" font-size="12.5" fill="{MUTED}" text-anchor="middle">moves left out</text>')
            o.append(f'<text x="{n["cx"]:.1f}" y="{n["cy"] + 14:.1f}" font-size="12" fill="{INK}" font-weight="600" text-anchor="middle">box above: {pct(n["avg"])}</text>')
            continue
        col, sw = {"inner": (FOLLOW, 2.2), "leaf": (STOP, 1.6), "over": (OVER, 1.6)}[n["kind"]]
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{W}" height="{H}" rx="8" fill="{CARD}" stroke="{col}" stroke-width="{sw}"/>')
        title = "Root" if n["parent"] is None else n["san"]
        # every box shows the value head on that board state (or the outcome); darker where it is all of the value
        what, v, used = {"inner": ("value head", n["head"], False), "leaf": ("value head", n["head"], True),
                         "over": ("outcome", n["wdl"], True)}[n["kind"]]
        o.append(f'<text x="{n["cx"]:.1f}" y="{n["cy"] - 4:.1f}" font-size="14" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')
        o.append(f'<text x="{n["cx"]:.1f}" y="{n["cy"] + 14:.1f}" font-size="12" fill="{INK if used else MUTED}" '
                 f'font-weight="{600 if used else 400}" text-anchor="middle">{what}: {pct(scalar(v))}</text>')
    o.append("</svg>")
    out = Path("docs/img/tree.svg")
    out.write_text("\n".join(o))

    # the arithmetic, for the README
    print(out, f"{len(draw)} boxes")
    print("value head at followed nodes:", {("root" if j == 0 else n["san"]): round(scalar(n["head"]), 3) for j, n in enumerate(draw) if n["kind"] == "inner"})
    print(f"zero-hop {scalar(score.zero_hop):.4f}  one-hop {scalar(score.one_hop):.4f}  tree(eps {EPS}) {score.value:.4f} "
          f"[{score.nodes} nodes]  default {fine.value:.4f} [{fine.nodes} nodes, depth {fine.depth}]  wdl {fine.wdl}")
    for j, n in enumerate(draw):
        if j in dk:
            terms = " + ".join(f"{draw[k]['p']:.3f} x {(draw[k]['avg'] if 'fold' in draw[k] else scalar(draw[k]['wdl'])):.3f}" for k in dk[j])
            print(f"  {'root' if j == 0 else n['san']:8s} depth {n['depth']} path {n['q']:.3f}: {terms} = {scalar(n['wdl']):.3f}")


if __name__ == "__main__":
    main()
