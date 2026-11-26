"""Tables and charts from the evaluation runs: docs/evals.md and docs/img/*.png.

Every value estimate is compared with the real game outcome. The first two metrics are the Maia-3 paper's:

  * value accuracy (the paper's "game outcome prediction accuracy"): how often the likeliest of win, draw,
    loss is the real outcome;
  * value loss: the cross-entropy the value head is trained on, minus the log of the probability given to
    the real outcome among win, draw, loss (lower is better);
  * Brier score: the squared error summed over the three outcomes, (p_win - 1)^2 + p_draw^2 + p_loss^2 when
    White won, and likewise for the others (lower is better);
  * MSE of value: the squared gap between the value as one number (win + draw / 2) and the outcome as 1,
    1/2 or 0. It ignores how probability is split between a draw and a decisive game.

Differences from zero-hop come with a 95% interval from resampling whole games (board states of one game
share their outcome, so games, not board states, are the unit). The charts and the per-slice tables use
the MSE of value. Everything is computed from the stored rows in evals/results/<run>/*.jsonl, which keep
the full (win, draw, loss) of every estimate, so other metrics can be added without rerunning the model.

    python evals/report.py                    # reads evals/results/main and evals/results/sweep
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# colours: three series in a fixed order (validated for colour-blind separation), neutral ink and grid
C = {"zero_hop": "#2a78d6", "one_hop": "#eb6834", "deep": "#1baf7a"}
NAME = {"zero_hop": "value head alone (zero-hop)", "one_hop": "one move ahead (one-hop)", "deep": "deep value"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})
RNG = np.random.default_rng(0)

PREAMBLE = """# Evals

This page is written by `evals/report.py` from the stored results; rerun it rather than editing here. The
short version is in the [README](../README.md#results), and the reasons behind the algorithm's design are in
[design-notes.md](design-notes.md). How consistent each estimate is with itself from move to move is on its
own page, [consistency.md](consistency.md).

## Method

- **Games.** Rated blitz games from the [Lichess open database](https://database.lichess.org/), October 2026,
  which is after the period Maia-3 was trained on. Games between humans that ended on the board, by
  resignation or on time, at least 20 moves long (a move is one player's turn, so 10 by each), with the players within 400 rating points of each other.
  300 games were sampled per 200-point rating band from 800 to 2399 (`evals/fetch_games.py`); the runs take
  the bands in turn, so any prefix of the games is balanced by rating. How this population differs from the
  Maia-3 paper's test set is in the [README](../README.md#the-evaluation-games).
- **Board states.** Five per game, drawn at random after the first 8 moves, with a seed fixed by the game id.
- **Estimates.** For each board state, at the two players' real ratings: the value head alone (zero-hop),
  its policy-weighted average one move later (one-hop), and the deep value. One tree gives all three, and
  the deep value at every looser threshold and at several depth caps as well.
- **Truth.** How the game ended: 1, 1/2 or 0 for White.
- **Metrics.** Value accuracy (the Maia-3 paper's "game outcome prediction accuracy": how often the likeliest
  of win, draw, loss is what happened), value loss (the cross-entropy the value head is trained on), the
  Brier score over the three outcomes, and the MSE of value (the squared gap between win + draw / 2 and the
  result). Higher accuracy is better; lower is better for the other three.
- **Intervals.** Board states of one game share its result, so 95% intervals come from resampling whole
  games (2,000 resamples), on the paired difference from zero-hop.
- **Raw data.** `evals/data/games.jsonl` holds the games and `evals/results/<run>/*.jsonl` one row per board
  state with every estimate's full (win, draw, loss), so any other metric can be computed without the model.

"""


def load(name: str) -> list[dict]:
    files = [f for f in sorted(Path("evals/results", name).glob("*.jsonl")) if " " not in f.name]   # not a sync tool's copies
    rows = [json.loads(l) for f in files for l in open(f)]
    seen, out = set(), []
    for r in rows:
        if (r["game"], r["ply"]) not in seen:
            seen.add((r["game"], r["ply"]))
            out.append(r)
    return out


def run_eps(name: str) -> float:
    return json.loads(Path("evals/results", name, "run.json").read_text())["eps"]


def exp(wdl) -> np.ndarray:
    wdl = np.asarray(wdl, dtype=float)
    return wdl[:, 0] + 0.5 * wdl[:, 1]


def brier(wdl, y) -> np.ndarray:
    return (exp(wdl) - y) ** 2


def logloss(wdl, y) -> np.ndarray:
    wdl = np.clip(np.asarray(wdl, dtype=float), 1e-4, 1)
    wdl = wdl / wdl.sum(axis=1, keepdims=True)
    col = np.where(y == 1, 0, np.where(y == 0.5, 1, 2))
    return -np.log(wdl[np.arange(len(y)), col])


def brier3(wdl, y) -> np.ndarray:
    """The three-class Brier score: squared error summed over win, draw, loss."""
    wdl = np.asarray(wdl, dtype=float)
    hot = np.stack([y == 1, y == 0.5, y == 0], axis=1).astype(float)
    return ((wdl - hot) ** 2).sum(axis=1)


def correct(wdl, y) -> np.ndarray:
    col = np.where(y == 1, 0, np.where(y == 0.5, 1, 2))
    return (np.argmax(np.asarray(wdl, dtype=float), axis=1) == col).astype(float)


def by_game(rows) -> np.ndarray:
    ids = {g: i for i, g in enumerate(dict.fromkeys(r["game"] for r in rows))}
    return np.array([ids[r["game"]] for r in rows])


def interval(per_position: np.ndarray, game: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    """Mean of a per-position quantity with a 95% interval from resampling games."""
    g = game.max() + 1
    tot = np.bincount(game, weights=per_position, minlength=g)
    cnt = np.bincount(game, minlength=g).astype(float)
    pick = RNG.integers(0, g, size=(n, g))
    boot = tot[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return float(per_position.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fmt(m, lo, hi, scale=1000, digits=1) -> str:
    return f"{m * scale:+.{digits}f} ({lo * scale:+.{digits}f} to {hi * scale:+.{digits}f})"


def table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(" --- " for _ in head) + "|"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def slices(rows, y) -> list[tuple[str, np.ndarray]]:
    ply = np.array([r["ply"] for r in rows])
    left = np.array([r["plies"] - r["ply"] for r in rows])
    elo = np.array([(r["white_elo"] + r["black_elo"]) / 2 for r in rows])
    clock = np.array([r["clock"] for r in rows])
    term = np.array([r["termination"] for r in rows])
    gap = np.abs(exp([r["deep"] for r in rows]) - exp([r["zero_hop"] for r in rows]))
    return [
        ("All board states", np.ones(len(rows), bool)),
        ("Move numbers 5 to 15", ply < 30), ("Move numbers 16 to 30", (ply >= 30) & (ply < 60)), ("Move number 31 on", ply >= 60),
        ("Last 10 moves of the game", left <= 10), ("More than 10 moves from the end", left > 10),
        ("Rated 800 to 1199", elo < 1200), ("Rated 1200 to 1599", (elo >= 1200) & (elo < 1600)),
        ("Rated 1600 to 1999", (elo >= 1600) & (elo < 2000)), ("Rated 2000 to 2399", elo >= 2000),
        ("Mover has 30 s or more", clock >= 30), ("Mover has under 30 s", (clock >= 0) & (clock < 30)),
        ("Game ended on the board or by resignation", term == "Normal"), ("Game ended on time", term == "Time forfeit"),
        ("Deep value and zero-hop differ by 5 points or more", gap >= 0.05),
        ("Deep value and zero-hop differ by under 5 points", gap < 0.05),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main")
    ap.add_argument("--sweep", default="sweep")
    a = ap.parse_args()
    img = Path("docs/img")
    img.mkdir(parents=True, exist_ok=True)
    md: list[str] = [PREAMBLE]

    rows = load(a.main)
    y = np.array([r["result"] for r in rows])
    game = by_game(rows)
    P = {k: np.array([r[k] for r in rows]) for k in C}
    n_games = game.max() + 1
    md.append(f"## Main run\n\n{len(rows)} board states from {n_games} games; tree threshold (eps) "
              f"{run_eps(a.main):g}; mean tree size "
              f"{np.mean([r['nodes'] for r in rows]):.0f} board states, mean deepest line "
              f"{np.mean([r['depth'] for r in rows]):.1f} moves.\n")

    # --- headline table --------------------------------------------------------------------------
    base_b = brier(P["zero_hop"], y)
    const = float(np.mean((0.5 - y) ** 2))
    metrics = [("Value accuracy", correct, 100, "points", lambda v: f"{v * 100:.1f}%"),
               ("Value loss (cross-entropy)", logloss, 1000, "x1000", lambda v: f"{v:.4f}"),
               ("Brier score (3 outcomes)", brier3, 1000, "x1000", lambda v: f"{v:.4f}"),
               ("MSE of value", brier, 1000, "x1000", lambda v: f"{v:.4f}")]
    head = ["Value estimate"]
    out = [[NAME[k]] for k in C]
    for title, fn, scale, unit, show in metrics:
        head += [title, f"vs zero-hop, {unit} (95%)"]
        base = fn(P["zero_hop"], y)
        for row, k in zip(out, C):
            m = fn(P[k], y)
            row += [show(m.mean()), "" if k == "zero_hop" else fmt(*interval(m - base, game), scale=scale)]
    md.append(table(head, out))

    # --- the population, and the same scores under the Maia-3 paper's position filters -------------
    src = {g["id"]: g for g in map(json.loads, open("evals/data/games.jsonl"))}
    mine = [src[i] for i in dict.fromkeys(r["game"] for r in rows)]
    count = lambda key: ", ".join(f"{v} {k}" for k, v in Counter(g[key] for g in mine).most_common(4))
    like = np.array([r["ply"] >= 20 and r["clock"] >= 30 and min(src[r["game"]]["clocks"][:r["ply"]]) >= 30 for r in rows])
    md.append(f"\n### The games, and the Maia-3 paper's filters\n\nHow the {n_games} games ended: {count('termination')}; "
              f"results: {count('result')}; time controls (seconds + increment): {count('time_control')}. "
              f"{np.mean([r['clock'] < 30 for r in rows]) * 100:.0f}% of the board states have the mover under 30 seconds and "
              f"{np.mean([r['ply'] < 20 for r in rows]) * 100:.0f}% are within the first 20 moves.\n\nThe paper scores the value head on "
              f"positions after \"the first 10 moves\" (read here as 10 by each player) and before either player first drops under 30 seconds. The same two filters "
              f"keep {like.sum()} of these board states. The other differences from the paper's test set (the checkpoint, the "
              f"year, the rating mix, the number of positions per game) remain, so the rows below are not a reproduction of "
              f"its numbers.\n")
    md.append(table(["Value estimate", "Value accuracy, all", "with the filters", "Value loss, all", "with the filters"],
                    [[NAME[k], f"{correct(P[k], y).mean() * 100:.1f}%", f"{correct(P[k], y)[like].mean() * 100:.1f}%",
                      f"{logloss(P[k], y).mean():.4f}", f"{logloss(P[k], y)[like].mean():.4f}"] for k in C]))
    md.append(f"Value accuracy and value loss are the Maia-3 paper's metrics; higher accuracy is better, lower is "
              f"better for the other three. Always predicting 50% for White gives an MSE of value of {const:.4f}. "
              f"Outcomes in this sample: White won {np.mean(y == 1) * 100:.1f}%, draw {np.mean(y == 0.5) * 100:.1f}%, "
              f"Black won {np.mean(y == 0) * 100:.1f}%.\n")

    # --- slices ------------------------------------------------------------------------------------
    sl = slices(rows, y)
    out, plot = [], []
    for name, m in sl:
        if m.sum() < 30:
            continue
        g = by_game([r for r, keep in zip(rows, m) if keep])
        one = interval((brier(P["one_hop"], y) - base_b)[m], g)
        deep = interval((brier(P["deep"], y) - base_b)[m], g)
        out.append([name, int(m.sum()), f"{base_b[m].mean():.4f}", fmt(*one), fmt(*deep)])
        plot.append((name, int(m.sum()), one, deep))
    md.append("### MSE of value by slice (difference from zero-hop, x1000; negative is better)\n")
    md.append(table(["Slice", "Board states", "Zero-hop MSE", "One-hop", "Deep value"], out))

    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(plot) + 1.9))
    for i, (name, n, one, deep) in enumerate(plot):
        yy = len(plot) - 1 - i
        for k, (m_, lo, hi), off in (("one_hop", one, 0.14), ("deep", deep, -0.14)):
            ax.plot([lo * 1000, hi * 1000], [yy + off] * 2, color=C[k], lw=2, solid_capstyle="round")
            ax.plot(m_ * 1000, yy + off, "o", color=C[k], ms=6, mec=SURFACE, mew=1, label=NAME[k] if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(plot)), [f"{n}  (n={c})" for n, c, _, _ in plot][::-1])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MSE of value minus zero-hop's, x1000 (left of the line: closer to the real outcomes)")
    fig.suptitle("Is the deep value closer to the real game outcomes than the value head alone?",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(img / "mse_by_slice.png", dpi=170)
    plt.close(fig)
    md.append("![MSE of value by slice, difference from zero-hop](img/mse_by_slice.png)\n")

    # --- calibration -------------------------------------------------------------------------------
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)))
    cal = []
    for k in C:
        e = exp(P[k])
        b = np.clip(np.digitize(e, edges) - 1, 0, 9)
        xs = [e[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ax.plot(xs, ys, "-o", color=C[k], lw=2, ms=5, mec=SURFACE, mew=1, label=NAME[k])
        cal.append([NAME[k]] + [f"{y[b == i].mean():.2f}" if (b == i).sum() >= 20 else "" for i in range(10)])
    cal.append(["board states (zero-hop)"] + [int((np.clip(np.digitize(exp(P["zero_hop"]), edges) - 1, 0, 9) == i).sum()) for i in range(10)])
    ax.set_xlabel("Predicted value for White (win + draw / 2)")
    ax.set_ylabel("White's real average score")
    ax.set_title("Calibration: do board states valued at 70% score 70%?")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(img / "calibration.png", dpi=170)
    plt.close(fig)
    md.append("![Calibration of the three estimates](img/calibration.png)\n")
    md.append("### Calibration (White's real average score by predicted value)\n")
    md.append(table(["Predicted value"] + [f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}%" for i in range(10)], cal))

    # --- when the tree moves the number, does the truth follow? -------------------------------------
    z, d = exp(P["zero_hop"]), exp(P["deep"])
    move = d - z
    slope = lambda m, r: float((m * r).sum() / (m * m).sum())
    s_all = slope(move, y - z)
    boots = []
    for _ in range(1000):
        gs = RNG.integers(0, n_games, n_games)
        w = np.bincount(gs, minlength=n_games)[game]
        boots.append(float((w * move * (y - z)).sum() / (w * move * move).sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    md.append("### When the tree moves the value, does the real outcome follow?\n")
    md.append(f"Regressing (outcome minus zero-hop) on (deep value minus zero-hop), through the origin: slope "
              f"**{s_all:.2f}** (95% {lo:.2f} to {hi:.2f}). A slope of 1 means the tree's correction is fully "
              f"right on average, 0 means it is noise, and in between means part of it is right. "
              f"Mean absolute move: {np.abs(move).mean() * 100:.1f} points; "
              f"{(np.abs(move) >= 0.05).mean() * 100:.0f}% of board states move by 5 points or more, "
              f"{(np.abs(move) >= 0.10).mean() * 100:.0f}% by 10 or more.\n")
    edges_m = np.array([-1, -0.15, -0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 0.15, 1])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs, ys, es, ns = [], [], [], []
    for i in range(len(edges_m) - 1):
        m = (move >= edges_m[i]) & (move < edges_m[i + 1])
        if m.sum() < 25:
            continue
        xs.append(move[m].mean() * 100)
        ys.append((y - z)[m].mean() * 100)
        es.append((y - z)[m].std() / np.sqrt(m.sum()) * 100 * 1.96)
        ns.append(int(m.sum()))
    lim = max(abs(min(xs)), abs(max(xs))) * 1.25
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls=(0, (4, 3)), label="the tree's correction is fully right")
    ax.axhline(0, color=MUTED, lw=1, label="the value head was right")
    ax.errorbar(xs, ys, yerr=es, fmt="o", color=C["deep"], ms=6, mec=SURFACE, mew=1, lw=1.5)
    md.append(table(["Deep value minus zero-hop (points, bin mean)"] + [f"{x:+.1f}" for x in xs],
                    [["Real outcome minus zero-hop (points)"] + [f"{v:+.1f}" for v in ys],
                     ["Board states"] + ns]))
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("How far the tree moved the value (deep value minus zero-hop, points)")
    ax.set_ylabel("Real outcome minus zero-hop (points)")
    ax.set_title("When the tree moves the value, does the real outcome follow?")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(img / "correction.png", dpi=170)
    plt.close(fig)
    md.append("![Real outcome against how far the tree moved the value](img/correction.png)\n")

    # --- the draw probability on its own --------------------------------------------------------------
    d_edges = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
    out = []
    for k in ("zero_hop", "deep"):
        pd = P[k][:, 1]
        b = np.clip(np.digitize(pd, d_edges) - 1, 0, len(d_edges) - 2)
        out.append([NAME[k] + ": mean predicted draw"] + [f"{pd[b == i].mean() * 100:.1f}%" if (b == i).sum() >= 20 else "" for i in range(len(d_edges) - 1)])
        out.append([NAME[k] + ": real draw rate"] + [f"{(y[b == i] == 0.5).mean() * 100:.1f}%" if (b == i).sum() >= 20 else "" for i in range(len(d_edges) - 1)])
        out.append([NAME[k] + ": board states"] + [int((b == i).sum()) for i in range(len(d_edges) - 1)])
    md.append("### Calibration of the draw probability\n")
    md.append(table(["Predicted draw probability"] + [f"{d_edges[i] * 100:g}-{d_edges[i + 1] * 100:g}%" for i in range(len(d_edges) - 1)], out))

    # --- thresholds and depth caps -------------------------------------------------------------------
    sweep_rows = load(a.sweep) if Path("evals/results", a.sweep).exists() else []
    # tree sizes per threshold: from the rows themselves, else (main run) from the sweep's trees
    sweep_size = {e: np.mean([r["eps_nodes"][e] for r in sweep_rows]) for e in (sweep_rows[0]["eps_nodes"] if sweep_rows else {})}

    def sweep_table(rows, title, own):
        y = np.array([r["result"] for r in rows])
        game = by_game(rows)
        zero = brier([r["zero_hop"] for r in rows], y)
        finest = exp([r["deep"] for r in rows])
        eps_keys = sorted(rows[0]["eps"], key=float, reverse=True)
        series = [("zero-hop", [r["zero_hop"] for r in rows], 1), ("one-hop", [r["one_hop"] for r in rows], None)]
        series += [(f"eps {e}", [r["eps"][e] for r in rows],
                    np.mean([r["eps_nodes"][e] for r in rows]) if "eps_nodes" in rows[0] else sweep_size.get(e)) for e in eps_keys]
        series.append((f"eps {own:g}", [r["deep"] for r in rows], np.mean([r["nodes"] for r in rows])))
        out, pts = [], []
        for name, wdl, nodes in series:
            b = brier(wdl, y)
            ch = np.abs(exp(wdl) - finest)
            out.append([name, f"{b.mean():.4f}", "" if name == "zero-hop" else fmt(*interval(b - zero, game)),
                        f"{correct(wdl, y).mean() * 100:.1f}%", f"{ch.mean() * 100:.2f}", f"{np.percentile(ch, 95) * 100:.1f}",
                        "" if nodes is None else f"{nodes:.0f}"])
            pts.append((name, interval(b - zero, game), ch.mean() * 100, nodes))
        md.append(f"### {title}\n")
        md.append(table(["Value estimate", "MSE of value", "vs zero-hop, x1000 (95%)", "Value accuracy",
                         "Mean gap to the finest tree (points)", "95th percentile gap", "Mean tree size (model evaluations)"], out))
        return pts, eps_keys

    main_pts, main_keys = sweep_table(rows, f"Thresholds, main run ({len(rows)} board states)", run_eps(a.main))
    md.append("Every row comes from the same tree per board state. Tree sizes at the looser thresholds are means over "
              "the deep sweep's board states.\n\n![Closeness to real outcomes by threshold](img/mse_by_threshold.png)\n")
    y_m = y
    out = []
    for dcap in sorted(rows[0]["depth_cap"], key=int):
        b = brier([r["depth_cap"][dcap] for r in rows], y_m)
        ch = np.abs(exp([r["depth_cap"][dcap] for r in rows]) - exp(P["deep"]))
        out.append([f"at most {dcap} moves", f"{b.mean():.4f}", fmt(*interval(b - base_b, game)), f"{ch.mean() * 100:.2f}"])
    md.append("### The same tree cut at a depth (main run)\n")
    md.append(table(["Tree", "MSE of value", "vs zero-hop, x1000 (95%)", "Mean gap to the full tree (points)"], out))

    if sweep_rows:
        md.append(f"## Deep sweep\n\nA smaller run with a finer tree, to see how much further the value moves and "
                  f"what it costs. {len(sweep_rows)} board states from {by_game(sweep_rows).max() + 1} games, mean tree "
                  f"size {np.mean([r['nodes'] for r in sweep_rows]):.0f} board states, mean deepest line "
                  f"{np.mean([r['depth'] for r in sweep_rows]):.1f} moves.\n")
        sw_pts, sw_keys = sweep_table(sweep_rows, f"Thresholds, deep sweep ({len(sweep_rows)} board states)", run_eps(a.sweep))
        # how far each threshold's reading is from the finest tree, and what it costs: two small charts
        own = run_eps(a.sweep)
        names = [p[0] for p in sw_pts if p[0].startswith("eps")]
        eps_x = [float(n.split()[1]) for n in names]
        gaps = [p[2] for p in sw_pts if p[0] in names]
        sizes = [p[3] for p in sw_pts if p[0] in names]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        a1.plot(eps_x, gaps, "-o", color=C["deep"], lw=2, ms=6, mec=SURFACE, mew=1)
        a1.set_xscale("log"); a1.invert_xaxis()
        a1.set_ylim(bottom=0)
        a1.set_xlabel("Path-probability threshold (eps), finer to the right")
        a1.set_ylabel("Points")
        a1.set_title(f"Mean gap to the finest tree (eps {own:g})")
        if all(s is not None for s in sizes):
            a2.plot(eps_x, sizes, "-o", color=C["deep"], lw=2, ms=6, mec=SURFACE, mew=1)
            a2.set_xscale("log"); a2.set_yscale("log"); a2.invert_xaxis()
            a2.set_xlabel("Path-probability threshold (eps), finer to the right")
            a2.set_ylabel("Model evaluations")
            a2.set_title("Cost: mean tree size")
        fig.tight_layout()
        fig.savefig(img / "threshold_sweep.png", dpi=170)
        plt.close(fig)
        md.append("![Gap to the finest tree and cost, by threshold](img/threshold_sweep.png)\n")

    # accuracy against truth by threshold, main run
    pts = [p for p in main_pts if p[0] != "zero-hop"]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    xs = range(len(pts))
    for x, (name, (m_, lo, hi), _, _) in zip(xs, pts):
        col = C["one_hop"] if name == "one-hop" else C["deep"]
        ax.plot([x, x], [lo * 1000, hi * 1000], color=col, lw=2, solid_capstyle="round")
        ax.plot(x, m_ * 1000, "o", color=col, ms=7, mec=SURFACE, mew=1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(list(xs), [p[0].replace("eps ", "") for p in pts])
    ax.tick_params(axis="x", colors=INK)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("One-hop, then the tree at each threshold (eps), finer to the right")
    ax.set_ylabel("MSE of value minus zero-hop's, x1000")
    ax.set_title("Closeness to real game outcomes, by how deep the tree goes")
    fig.tight_layout()
    fig.savefig(img / "mse_by_threshold.png", dpi=170)
    plt.close(fig)

    # --- the earlier way of choosing moves, on the games both runs cover ------------------------------
    if Path("evals/results/top8").exists():
        old = {(r["game"], r["ply"]): r for r in load("top8")}
        both = [(old[(r["game"], r["ply"])], r) for r in rows if (r["game"], r["ply"]) in old]
        ra, rb = [x for x, _ in both], [x for _, x in both]
        yb, gb = np.array([r["result"] for r in rb]), by_game(rb)
        est = [("zero-hop (value head)", [r["zero_hop"] for r in rb], 1),
               ("top 8 moves, renormalised, eps 0.01", [r["deep"] for r in ra], np.mean([r["nodes"] for r in ra])),
               ("this repo, eps 0.01", [r["eps"]["0.01"] for r in rb], sweep_size.get("0.01")),
               ("this repo, eps 0.002", [r["deep"] for r in rb], np.mean([r["nodes"] for r in rb]))]
        ref = est[1][1]
        out = []
        for name, wdl, size in est:
            row = [name, "" if size is None else f"{size:.0f}"]
            for _, fn, scale, _, show in metrics:
                row.append(show(fn(wdl, yb).mean()))
            row.append("" if wdl is ref else fmt(*interval(brier(wdl, yb) - brier(ref, yb), gb)))
            out.append(row)
        md.append(f"## The earlier way of choosing moves\n\nBefore the current rule, a board state followed its likeliest "
                  f"moves up to 97% of the policy, at most 8, with renormalised weights ([design-notes.md](design-notes.md), "
                  f"section 2). {len(rb)} board states from {gb.max() + 1} games that both runs cover.\n")
        md.append(table(["Value estimate", "Model evaluations", "Value accuracy", "Value loss", "Brier score", "MSE of value",
                         "MSE vs top 8, x1000 (95%)"], out))

    Path("docs/evals.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
