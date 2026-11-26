"""Is an estimate consistent with itself from one move to the next?

A value is a forecast of the result. If it is a sound forecast then, before a move, it equals the average of
what it will say after the move, over the moves the player might make. This script measures how far the value
head alone, the value head averaged k moves ahead, and the deep value are from that, without using game
results, and puts the accuracy against real outcomes next to it.

Reads   evals/results/lookahead   the value head averaged k = 0..8 moves ahead (evals/lookahead.py)
        evals/results/tail_deep   every legal move valued afresh (evals/tail_deep.py)
        evals/results/games       whole games, every board state (evals/run_games.py)
        evals/results/main        the main run against real outcomes (evals/run_eval.py)
Writes  docs/consistency.md and docs/img/zigzag_by_k.png, zigzag_example.png, game_trace.png

    python evals/consistency.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import C, INK, MUTED, SURFACE, brier, by_game, exp, interval, load, plt, table  # noqa: E402

RNG = np.random.default_rng(0)
IMG = Path("docs/img")
EXAMPLE = ("klZArXVF", 63, 74)      # game and plies of the stretch charted as the example: a level middlegame
                                    # where the value head reverses direction on every ply (chosen to illustrate)
PCT = [50, 80, 90, 95]
PCT_HEAD = ["Median", "80th percentile", "90th", "95th", "Mean"]


def pct(x) -> list[str]:
    return [f"{v:.1f}" for v in np.percentile(np.abs(x), PCT)] + [f"{np.abs(x).mean():.1f}"]


def ci(x) -> str:
    return f"{x.mean():+.2f} ± {1.96 * x.std(ddof=1) / len(x) ** 0.5:.2f}"


def spread(x) -> list[str]:
    """10th percentile, median and 90th percentile, signed."""
    return [f"{v:+.1f}" for v in np.percentile(x, [10, 50, 90])]


def share(x) -> str:
    return f"{np.mean(x) * 100:.1f}%"


def mid(r) -> np.ndarray:
    """The midpoint of the value head and one move ahead."""
    return (np.array(r["zero_hop"]) + np.array(r["one_hop"])) / 2


def rows_of(name: str) -> list[dict]:
    files = sorted(Path(f"evals/results/{name}").glob("*.jsonl"))
    return [json.loads(line) for f in files if " " not in f.name for line in open(f)]


def whole_games() -> list[list[dict]]:
    by: dict[str, list] = {}
    for r in load("games"):
        by.setdefault(r["game"], []).append(r)
    games = [sorted(g, key=lambda r: r["ply"]) for g in by.values()]
    return [g for g in games if len(g) == g[0]["plies"]]


def moves_average(rows: list[dict], own: str, kid: str) -> np.ndarray:
    """Per board state: the policy-weighted average of the values after each legal move, minus its own value."""
    return np.array([np.dot(r["p"], r[kid]) / sum(r["p"]) - r[own] for r in rows]) * 100


# --- the value head averaged k moves ahead ---------------------------------------------------------------------
def k_moves_ahead() -> str:
    rows = rows_of("lookahead")
    V = np.array([r["exact"] + r["mc"] for r in rows]) * 100
    deep = np.array([r["deep"]["0.002"] for r in rows]) * 100
    ks = range(V.shape[1])
    noise = np.mean([np.hypot(a, b) for r in rows for a, b in zip(r["se"][:-1], r["se"][1:])]) * 100
    check = abs(np.mean([r["mc2"] - r["exact"][2] for r in rows])) * 100
    md = [f"## The value head averaged k moves ahead: {len(rows)} board states\n\n`evals/lookahead.py`. V_k is the value head k "
          f"moves later, weighted by the policy along every line; k = 0 is the value head alone and k = 1 is one move ahead. "
          f"k = 0, 1, 2 are exact. k = 3 and up are sampled: 500 lines per board state, each move drawn from the policy, which "
          f"is unbiased; the noise on one step averages {noise:.1f} points, and the same draws give k = 2 within {check:.2f} "
          f"points of the exact value on average. Points, for the player to move at the sampled board state; ± is a 95% "
          f"interval.\n\n![The value head averaged k moves ahead, minus the value head alone](img/zigzag_by_k.png)\n"]
    md.append(table(["k moves ahead", "Minus the value head alone (k = 0): 10th percentile", "Median", "90th", "Average",
                     "Step from k − 1: 10th percentile", "Median", "90th", "Average", "Average minus the deep value (eps 0.002)"],
                    [[k] + ([""] * 8 if k == 0 else spread(V[:, k] - V[:, 0]) + [ci(V[:, k] - V[:, 0])] +
                            spread(V[:, k] - V[:, k - 1]) + [ci(V[:, k] - V[:, k - 1])]) + [ci(V[:, k] - deep)] for k in ks]))
    own = moves_average([r for r in rows_of("tail_deep") if "deep" in r and len(r["p"]) > 1], "deep", "vd")
    md.append(f"\nThe deep value's own step from k = 0 to 1 (its value against the average of its moves' deep values, each from a "
              f"tree of its own, eps 0.01): 10th percentile {spread(own)[0]}, median {spread(own)[1]}, 90th {spread(own)[2]}.")

    # where the lean is: odd k minus even k per board state, by how far into the game it is
    lean = V[:, 1::2].mean(1) - V[:, 0::2].mean(1)
    ply = np.array([r["ply"] for r in rows])
    src = {g["id"]: g for g in map(json.loads, open("evals/data/games.jsonl"))}
    clock = np.array([src[r["game"]]["clocks"][r["ply"] - 2] for r in rows])          # the mover's clock, seconds
    cuts = [("First 20 moves of the game", ply < 20), ("Moves 21 to 40", (ply >= 20) & (ply < 40)),
            ("Moves 41 to 70", (ply >= 40) & (ply < 70)), ("After move 70", ply >= 70),
            ("Mover has 2 minutes or more", clock >= 120), ("1 to 2 minutes", (clock >= 60) & (clock < 120)),
            ("30 to 60 seconds", (clock >= 30) & (clock < 60)), ("Under 30 seconds", clock < 30)]
    md.append("\n### Where the lean is\n\nPer board state: the average of V_k over odd k minus the average over even k, points. "
              "Moves are counted for both players together. The model does not see the clock; it is shown because it goes with "
              "the stage of the game.\n")
    md.append(table(["Board states", "Number", "10th percentile", "Median", "90th", "Average"],
                    [["All", len(rows)] + spread(lean) + [f"{lean.mean():+.2f}"]] +
                    [[name, int(m.sum())] + spread(lean[m]) + [f"{lean[m].mean():+.2f}"] for name, m in cuts]))
    md.append("\n### How large the swing is, board state by board state\n\nAbsolute values, points.\n")
    out = [[f"k = {k - 1} to {k}"] + pct(V[:, k] - V[:, k - 1]) for k in ks if k]
    out.append(["Highest minus lowest over all k"] + pct(V.max(1) - V.min(1)))
    td = {(r["game"], r["ply"]): r for r in rows_of("tail_deep") if "deep" in r and len(r["p"]) > 1}
    same = [td[(r["game"], r["ply"])] for r in rows if (r["game"], r["ply"]) in td]
    out.append([f"Deep value (eps 0.01): own value to the average of its moves' deep values ({len(same)} board states)"]
               + pct(moves_average(same, "deep", "vd")))
    md.append(table(["Step"] + PCT_HEAD, out))

    md.append("\n### Gap to the deep value (eps 0.002)\n\nAbsolute values, points.\n")
    out = [[f"{k} moves ahead"] + pct(V[:, k] - deep) for k in ks]
    out.append(["Midpoint of k = 0 and k = 1"] + pct((V[:, 0] + V[:, 1]) / 2 - deep))
    out += [[f"Deep value, eps {e}"] + pct(np.array([r["deep"][e] for r in rows]) * 100 - deep) for e in ("0.2", "0.05", "0.01")]
    md.append(table(["Estimate"] + PCT_HEAD, out))

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    lo, m, hi = np.percentile(V - V[:, :1], [10, 50, 90], axis=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(list(ks), lo, hi, color=C["zero_hop"], alpha=0.14, lw=0, label="10th to 90th percentile")
    ax.plot(list(ks), m, "-o", color=C["zero_hop"], lw=1.6, ms=6, mec=SURFACE, mew=1, label="median")
    ax.set_xticks(list(ks))
    ax.set_xlabel("k, moves looked ahead (0 is the value head alone)")
    ax.set_ylabel("change from the value head alone, points")
    ax.set_title(f"The value head averaged k moves ahead, minus the value head alone ({len(rows)} board states)", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=9)
    ax.set_ylim(-14, 10)
    ax.set_yticks(range(-14, 11, 2))
    fig.tight_layout()
    fig.savefig(IMG / "zigzag_by_k.png", dpi=170)
    plt.close(fig)
    return "\n".join(md)


# --- one move, every legal move valued afresh -------------------------------------------------------------------
def one_move() -> str:
    rows = [r for r in rows_of("tail_deep") if "deep" in r and len(r["p"]) > 1]
    out, gaps = [], {}
    for name, own, kid in (("Value head alone", "head", "v1"), ("Deep value (eps 0.01)", "deep", "vd")):
        gap = gaps[own] = moves_average(rows, own, kid)
        best = np.array([max(r[kid]) - r[own] for r in rows])
        out.append([name] + spread(gap) + [ci(gap), f"{np.abs(gap).mean():.2f}", share(np.abs(gap) >= 5), share(np.abs(gap) >= 10),
                                           share(best < 0), share(best < -0.02)])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True)
    bins = np.arange(-25, 16, 1)
    for ax, (own, label, key) in zip(axes, (("head", "value head alone", "zero_hop"), ("deep", "deep value", "deep"))):
        g = gaps[own]
        ax.hist(np.clip(g, -25, 15), bins=bins, weights=np.full(len(g), 100 / len(g)), color=C[key], alpha=0.85)
        ax.axvline(0, color=MUTED, lw=1)
        top = ax.get_ylim()[1] * 1.25
        ax.set_ylim(0, top)
        for q, name, side in zip(np.percentile(g, [10, 50, 90]), ("10th percentile", "median", "90th percentile"), ("right", "center", "left")):
            ax.axvline(q, color=INK, lw=1, ls=(0, (4, 3)))
            if name != "median":                       # the median is too close to the other two to label in the lower panel
                ax.text(q + {"right": -0.3, "left": 0.3}[side], top * 0.93, f"{name}: {q:+.1f}", ha=side, va="top", fontsize=9, color=INK)
        ax.set_title(f"{label} (median {np.median(g):+.1f})", loc="left", color=INK, fontsize=10)
        ax.set_ylabel("% of board states\nper 1-point bin")
    axes[1].set_xlabel("average value after the player's move, minus the value before it (points, for that player)")
    fig.tight_layout()
    fig.savefig(IMG / "one_move_step.png", dpi=170)
    plt.close(fig)
    return (f"## One move: {len(rows)} board states, every legal move valued afresh\n\nEach legal move gets its own tree "
            f"(`evals/tail_deep.py`), so the deep value of a board state is compared with deep values it was not built from. "
            f"If every legal move lowers the value, the value before the move was one the player could not keep. Points, for "
            f"the player to move; ± is a 95% interval.\n\n![The step from a value to the average of the values after each move]"
            f"(img/one_move_step.png)\n\n" +
            table(["Estimate", "Value after the move, averaged over the moves, minus the value before: 10th percentile", "Median",
                   "90th", "Average (should be 0)", "Mean absolute", "5 points or more either way", "10 points or more",
                   "Every move lowers the value", "Every move lowers it by over 2 points"], out))


# --- along whole games ------------------------------------------------------------------------------------------
def along_the_game(games: list[list[dict]]) -> str:
    est = [("Value head alone", lambda r: r["zero_hop"]), ("One move ahead", lambda r: r["one_hop"]),
           ("Deep value, eps 0.05", lambda r: r["eps"]["0.05"]), ("Deep value, eps 0.01", lambda r: r["eps"]["0.01"]),
           ("Deep value, eps 0.002", lambda r: r["deep"])]
    out = []
    for name, f in est:
        per = []                                                           # per game: sums for each statistic
        for g in games:
            w = exp([f(r) for r in g])
            mover = np.array([1 if r["white_to_move"] else -1 for r in g])
            d = np.diff(w)                                                 # change over each played move but the last
            full = np.diff(np.append(w, g[0]["result"]))                   # with the last step, to the result
            per.append([(d * mover[:-1]).sum(), len(d), np.abs(d).sum(), (d[1:] * d[:-1]).sum(), (d[:-1] ** 2).sum(),
                        (full ** 2).sum(), (g[0]["result"] - w[0]) ** 2])
        per = np.array(per, float)

        def stat(fn, scale=1.0, sign="+"):
            m = fn(per.sum(axis=0))
            boots = [fn(per[RNG.integers(0, len(per), len(per))].sum(axis=0)) for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return f"{m * scale:{sign}.2f} ({lo * scale:{sign}.2f} to {hi * scale:{sign}.2f})"
        out.append([name, stat(lambda s: s[0] / s[1], 100), stat(lambda s: s[2] / s[1], 100, ""),
                    stat(lambda s: s[3] / s[4]), stat(lambda s: s[5] / s[6], sign="")])
    swing = {}
    for name, f in (est[0], est[1], est[-1]):
        swing[name] = np.concatenate([np.diff(exp([f(r) for r in g])) for g in games]) * 100
    extra = swing["Value head alone"] - swing["Deep value, eps 0.002"]
    sizes = [[name] + pct(x)[:-1] + [share(np.abs(x) >= 5), share(np.abs(x) >= 10)] for name, x in
             list(swing.items()) + [("Value head's change minus the deep value's, same move", extra)]]
    sizes = ("\n\n### How large the change from one move to the next is\n\nAbsolute change in the value over one played move, "
             "points, all moves of all games.\n\n" +
             table(["Estimate"] + PCT_HEAD[:-1] + ["5 points or more", "10 points or more"], sizes))
    return (f"## Along the game: {len(games)} games, {sum(len(g) for g in games)} board states\n\nEvery board state of "
            f"each game, valued at the players' ratings (`evals/run_games.py`); a change is the value after the move that was "
            f"played minus the value before it. Points; 95% intervals resample games.\n\n" +
            table(["Estimate", "A move changes the mover's value by (should be 0)", "Mean absolute change per move",
                   "Correlation of one change with the next (should be 0)",
                   "Squared changes over the game / squared distance start to result (should be 1)"], out) + sizes)


# --- accuracy and movement side by side -------------------------------------------------------------------------
def side_by_side(games: list[list[dict]]) -> str:
    main = load("main")
    y, gm = np.array([r["result"] for r in main]), by_game(main)
    est = [("Value head alone", "1", lambda r: r["zero_hop"]), ("One move ahead", "about 32", lambda r: r["one_hop"]),
           ("Midpoint of the two", "about 32", mid)]
    est += [(f"Deep value, eps {e}", n, (lambda e: lambda r: r["eps"][e])(e)) for e, n in
            (("0.5", "2"), ("0.2", "4"), ("0.1", "8"), ("0.05", "16"), ("0.01", "80"))]
    est.append(("Deep value, eps 0.002", "390", lambda r: r["deep"]))
    b_head, b_mid = brier([r["zero_hop"] for r in main], y), brier([mid(r) for r in main], y)

    def gain(m, lo, hi):
        return f"{-m * 1000:.1f} ({-hi * 1000:.1f} to {-lo * 1000:.1f})"
    out = []
    for name, n, f in est:
        b = brier([f(r) for r in main], y)
        per = []
        for g in games:
            w = exp([f(r) for r in g]) * 100
            d = np.diff(w)
            per.append([w.var(), d.var(), ((w[1:-1] - (w[:-2] + w[2:]) / 2) ** 2).mean(), (d[1:] * d[:-1]).sum(), (d[:-1] ** 2).sum()])
        per = np.array(per)
        out.append([name, n, f"{b.mean():.4f}", "" if f is est[0][2] else gain(*interval(b - b_head, gm)),
                    "" if f is mid else gain(*interval(b - b_mid, gm)), f"{per[:, 0].mean() ** 0.5:.1f}",
                    f"{per[:, 1].mean() ** 0.5:.1f}", f"{per[:, 2].mean() ** 0.5:.1f}", f"{per[:, 3].sum() / per[:, 4].sum():+.2f}"])
    return (f"## Accuracy and movement side by side\n\nMSE of value on the {len(main)} board states of the main run; an improvement "
            f"is the other estimate's MSE minus this one's, times 1,000, with a 95% interval from resampling games. The last four "
            f"columns are from the {len(games)} whole games: standard deviations in points, pooled over games, and the correlation "
            f"of one change with the next. The spread of values over a game is dominated by the real course of the game, so it "
            f"does not separate the estimates; the change per move does.\n\n" +
            table(["Estimate", "Model evaluations", "MSE of value", "Improvement over the value head", "Improvement over the midpoint",
                   "Spread of values over a game", "Change per move", "Gap to the average of the two neighbouring board states",
                   "Correlation of successive changes"], out))


# --- charts of single games -------------------------------------------------------------------------------------
def chart_example(games: list[list[dict]]) -> str:
    """The stretch named in EXAMPLE: value head and deep value ply by ply. Returns it as a table."""
    gid, a, b = EXAMPLE
    g = next(g for g in games if g[0]["game"] == gid)[a:b]
    x = np.arange(len(g))
    lines = [("zero_hop", "value head alone", 1.8, lambda r: r["zero_hop"]), ("deep", "deep value (eps 0.002)", 2.4, lambda r: r["deep"])]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.axhline(50, color=MUTED, lw=1)
    for key, label, lw, f in lines:
        ax.plot(x, exp([f(r) for r in g]) * 100, "-o", color=C[key], lw=lw, ms=4.5, mec=SURFACE, mew=1, label=label)
    ax.set_xticks(x, [f"{r['ply'] // 2 + 1}{'.' if r['white_to_move'] else '...'}{r['played']}" for r in g], fontsize=8)
    ax.set_xlabel("the move about to be played (each value is from before it)")
    ax.set_ylabel("White's win chance, % (win + half of draws)")
    ax.set_ylim(30, 70)
    ax.set_title(f"One stretch of a game ({g[0]['white_elo']} against {g[0]['black_elo']}), move by move", loc="left", color=INK)
    handles, labels = ax.get_legend_handles_labels()
    sf = Path("evals/results/example_stockfish.json")
    if sf.exists():                                    # a check that the game is level, on its own axis
        sf = json.loads(sf.read_text())
        right = ax.twinx()
        right.plot(x, [sf["pawns"][str(r["ply"])] for r in g], "D", color=MUTED, ms=5, mec=SURFACE, mew=1, ls=(0, (2, 3)), lw=1,
                   label=f"{sf['engine']}, depth {sf['depth']} (right axis)")
        right.set_ylim(-5, 5)                          # 0 pawns sits on the 50-point line
        right.set_ylabel("Stockfish evaluation for White, pawns")
        right.grid(False)
        right.spines["right"].set_visible(True)
        h2, l2 = right.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, frameon=False, ncols=3, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "zigzag_example.png", dpi=170)
    plt.close(fig)

    def v(r, f):
        return f"{exp([f(r)])[0] * 100:.0f}"
    return table(["To move", "Value head alone", "Deep value", "Move then played"],
                 [["White" if r["white_to_move"] else "Black", v(r, lines[0][3]), v(r, lines[1][3]),
                   f"{r['ply'] // 2 + 1}{'.' if r['white_to_move'] else '...'}{r['played']}"] for r in g])


def chart_game(games: list[list[dict]]):
    """A whole game: the game of 50 to 90 plies whose value-head movement per ply is the median one."""
    pick = [g for g in games if 50 <= len(g) <= 90]
    move = [np.abs(np.diff(exp([r["zero_hop"] for r in g]))).mean() for g in pick]
    g = pick[int(np.argsort(move)[len(move) // 2])]
    x = np.arange(len(g)) / 2 + 1
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.axhline(50, color=MUTED, lw=1)
    ax.plot(x, exp([r["zero_hop"] for r in g]) * 100, "-", color=C["zero_hop"], lw=1.4, label="value head alone")
    ax.plot(x, exp([r["deep"] for r in g]) * 100, "-", color=C["deep"], lw=2.2, label="deep value (eps 0.002)")
    ax.set_xlabel("move number")
    ax.set_ylabel("White's win chance, % (win + half of draws)")
    ax.set_ylim(0, 100)
    res = {1.0: "White won", 0.0: "Black won", 0.5: "draw"}[g[0]["result"]]
    ax.set_title(f"One game, every board state ({g[0]['white_elo']} against {g[0]['black_elo']}, {res})", loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "game_trace.png", dpi=170)
    plt.close(fig)


PREAMBLE = """# Self-consistency

Generated by `evals/consistency.py`; rerun it rather than editing here. The fit to real outcomes is in
[evals.md](evals.md), and how these measures were arrived at is in [experiment-log.md](experiment-log.md).

A value is a forecast of the result. A sound forecast, before a move, equals the average of what it will say
after the move, over the moves the player might make. Things that follow, each measurable without knowing who
won:

- **Looking further ahead should not change the answer on average.** The value head on a board state, and the
  value head averaged over the board states k moves later, are forecasts of the same result.
- **No drift.** A move should not change the mover's value on average.
- **No zig-zag.** One change should not predict the next. A correlation of -1 between successive changes means
  every drop is taken back on the next move.
- **No contradiction.** The value before a move should lie between the values after the best and the worst
  move.
- **No excess movement.** Over a game, the squared changes should add up to the squared distance from the first
  value to the result. A ratio above 1 means the estimate moves more than the information it gains.

The deep value is built as an average over its moves, so it should do well here. The checks are still not
circular: the value after a move comes from a separate tree with its own budget, which reaches board states
the first tree never saw.
"""

if __name__ == "__main__":
    games = whole_games()
    example = chart_example(games)
    chart_game(games)
    md = [PREAMBLE, k_moves_ahead(), one_move(), along_the_game(games), side_by_side(games),
          "## Two games\n\nA stretch chosen to show the pattern clearly: a level middlegame (Stockfish 18 at depth 24 rates all "
          "eleven board states between -0.36 and +0.44 pawns) where the value head reverses direction on every move. White's "
          "value, points.\n\n![One stretch of a game, move by move](img/zigzag_example.png)\n\n" + example +
          "\n\nAnd a whole game chosen by rule, not by eye: of the games of 50 to 90 moves, the one whose value-head movement per "
          "move is the median.\n\n![White's value along one game: value head alone and deep value](img/game_trace.png)"]
    Path("docs/consistency.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))
