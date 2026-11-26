"""Does discounting the moves a node leaves out bring the "eps only" tree closer to the reference tree?

Two ways to value the moves left out at a node (those whose path probability is under eps):
  head   the node's own value head
  log    the value head less a + b * log10(1 / cut) points for the mover, cut = eps / the node's path
         probability (the fit from evals/tail_value.py)

Uses the board states and reference values of evals/results/truncation.jsonl. One tree per board state, grown
at the finest eps, answers every looser eps.

    python evals/tail_discount.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_value import DeepValue, Maia, scalar  # noqa: E402

EPS = (0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002)
TAILS = {"head": None, "log": (5.4, 5.1), "deep log": (3.2, 4.3)}     # deep log: fitted by evals/tail_deep_fit.py


def back_up(deep, nodes, eps, tail):
    """White's value of the root for this eps and this way of valuing the moves left out."""
    val = [0.0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        n = nodes[i]
        own = scalar(deep._white_wdl(n["k"], n["b"]))
        kids = [k for k in n["kids"] or () if nodes[k]["q"] >= eps]
        if not kids or (i and n["q"] < eps):
            val[i] = own
            continue
        rest = 1.0 - sum(nodes[k]["p"] for k in kids)
        left = own
        if tail:
            off = (tail[0] + tail[1] * math.log10(n["q"] / eps)) / 100          # for the mover
            left = min(1.0, max(0.0, own - off if n["b"].turn == chess.WHITE else own + off))
        val[i] = sum(nodes[k]["p"] * val[k] for k in kids) + rest * left
    return val[0]


def main():
    games = {g["id"]: g for g in map(json.loads, open("evals/data/games.jsonl"))}
    rows = [json.loads(l) for l in open("evals/results/truncation.jsonl")]
    maia = Maia(device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
    out = []
    for r in rows:
        g = games[r["game"]]
        b = chess.Board()
        for u in g["moves"].split()[: r["ply"]]:
            b.push_uci(u)
        deep = DeepValue(maia, eps=min(EPS))
        deep.value(b, g["white_elo"], g["black_elo"])
        nodes = deep._trees[0]
        row = {"game": r["game"], "ply": r["ply"], "ref": r["ref"], "zero_hop": r["zero_hop"], "values": {}, "nodes": {}}
        for eps in EPS:
            row["nodes"][f"{eps:g}"] = sum(1 for n in nodes if n["q"] >= eps)
            for name, tail in TAILS.items():
                row["values"][f"{name}|{eps:g}"] = round(back_up(deep, nodes, eps, tail), 5)
            was = r["variants"].get(f"eps only|{eps:g}")
            if was and abs(was["value"] - row["values"][f"head|{eps:g}"]) > 2e-5:
                print("MISMATCH", r["game"], eps, was["value"], row["values"][f"head|{eps:g}"])
        out.append(row)
        print(r["game"], len(nodes), flush=True)
    json.dump(out, open("evals/results/tail_discount.json", "w"))
    gap = lambda row, key: abs(row["values"][key] - row["ref"]) * 100
    ci = lambda d: 1.96 * st.stdev(d) / len(d) ** 0.5
    print(f"\n{len(out)} board states. Gap = |value - reference| in points.\n")
    print("| eps | Board states evaluated | head | log | deep log | log minus head (95%) | deep log minus head (95%) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for eps in EPS:
        e = f"{eps:g}"
        h, l, f = ([gap(r, f"{t}|{e}") for r in out] for t in TAILS)
        dl, df = [x - y for x, y in zip(l, h)], [x - y for x, y in zip(f, h)]
        print(f"| {e} | {st.mean(r['nodes'][e] for r in out):.0f} | {st.mean(h):.2f} | {st.mean(l):.2f} | {st.mean(f):.2f} | "
              f"{st.mean(dl):+.2f} ± {ci(dl):.2f} | {st.mean(df):+.2f} ± {ci(df):.2f} |")
    sg = lambda r: 1 if r["ply"] % 2 == 0 else -1
    print("\nError for the player to move (value - reference, points)\n\n| eps | " + " | ".join(TAILS) + " |\n| --- | --- | --- | --- |")
    for eps in EPS:
        cells = []
        for t in TAILS:
            d = [sg(r) * (r["values"][f"{t}|{eps:g}"] - r["ref"]) * 100 for r in out]
            cells.append(f"{st.mean(d):+.2f} ± {ci(d):.2f}")
        print(f"| {eps:g} | " + " | ".join(cells) + " |")
    z = [abs(r["zero_hop"] - r["ref"]) * 100 for r in out]
    print(f"\nValue head alone: {st.mean(z):.2f}")


main()
