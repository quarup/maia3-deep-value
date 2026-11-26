# Design notes: why the algorithm is what it is

This file records the choices behind `deep_value/tree.py` and the measurements each one rests on. The [README](../README.md) describes the algorithm and [evals.md](evals.md) reports how it does against real games; this is the reasoning in between. All of it was measured in November 2026 with the `maia3-79m` checkpoint.

## The decisions in one table

| Question | Decision | Main evidence |
| --- | --- | --- |
| Which moves does a board state follow? | Every move whose path probability is at least `eps`. No limit on the number of moves, no share of the policy to reach. | Same closeness to real outcomes as the earlier "top 8 moves" rule, with fewer model evaluations ([section 2](#2-which-moves-to-follow)) |
| What weight do the followed moves get? | The policy's own probabilities, not renormalised. | Renormalising after cutting by `eps` roughly doubles the gap to a reference tree ([section 3](#3-renormalise-or-not)) |
| What are the moves left out worth? | The value head of the board state they leave. | Three corrections were tried; none beat it, and two made shallow trees worse ([section 4](#4-what-the-moves-left-out-are-worth)) |
| Default threshold | `eps` = 0.002 | Closer thresholds stop paying against real outcomes around 0.005 to 0.002 ([section 6](#6-the-default-threshold)) |
| What is stored | The full (win, draw, loss) of every estimate, per board state | Any metric can be recomputed without the model ([section 7](#7-metrics)) |

## 1. Two yardsticks

Design choices were judged two ways, because they answer different questions.

**Gap to a reference tree.** For 30 board states (random games and points in them from the evaluation set) a reference value was computed with every legal move followed at `eps` 0.002, about 11,900 model evaluations per board state. A variant's *gap* is the absolute difference between its value and the reference's, in points (percentage points of win + draw / 2), averaged over the 30. This measures how much a shortcut changes the answer, with no noise from game results. It is cheap enough to compare many variants, but 30 board states give error bars of 0.15 to 0.9 points, and the reference is itself an approximation: its own value still moves by about 0.25 points between `eps` 0.005 and 0.002. For scale, the value head is 3.75 points from the reference.

**Real outcomes.** The value is compared with how the game ended, over thousands of board states ([evals.md](evals.md)). This is the yardstick that matters, but a single game result is very noisy, so only differences of about 0.2 x 1000 in MSE or more are detectable even with 10,000 board states.

Paired differences are reported as mean ± 1.96 standard errors over board states (reference-tree studies) or as a 95% interval from resampling whole games (real outcomes).

## 2. Which moves to follow

The first version, inherited from the video project this code came from, had four settings: follow the likeliest moves up to 97% of the policy, at most 8 of them, renormalise their weights, and stop a line when its path probability falls under `eps`. The 8 and the 97% were picked by hand. The question was whether `eps` alone could do the job.

Variants compared (`evals/truncation.py`, results in `evals/results/truncation.jsonl`, table from `evals/truncation_report.py`):

| Variant | What it follows |
| --- | --- |
| top 8 | likeliest moves up to 97% of the policy, at most 8, renormalised; a board state is expanded while its own path probability is at least `eps` |
| all moves | every legal move; same expansion rule |
| top 8, not renormalised | top 8's moves with the policy's own weights; moves left out count with the board state's value head |
| eps only | a move is followed only if the path *to it* has probability at least `eps`; moves left out count with the board state's value head |
| eps only, renormalised | the same moves, weights renormalised |

Mean gap to the reference and mean model evaluations, 30 board states:

| eps | top 8 | all moves | top 8, not renormalised | eps only | eps only, renormalised |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 1.26 (50) | 1.19 (237) | 1.20 (44) | 1.71 (8) | 3.75 (17) |
| 0.05 | 0.96 (100) | 0.88 (483) | 0.99 (88) | 1.40 (16) | 2.74 (36) |
| 0.02 | 0.74 (250) | 0.65 (1,156) | 0.73 (209) | 1.31 (38) | 1.58 (82) |
| 0.01 | 0.57 (514) | 0.42 (2,389) | 0.57 (421) | 0.98 (77) | 1.48 (181) |
| 0.005 | 0.55 (1,006) | 0.25 (4,739) | 0.50 (806) | 0.85 (152) | 1.37 (356) |
| 0.002 | 0.50 (2,496) | reference (11,872) | 0.40 (1,936) | 0.57 (382) | 1.01 (893) |
| 0.001 | | | | 0.48 (759) | 0.83 (1,767) |
| 0.0005 | | | | 0.39 (1,521) | 0.85 (3,499) |
| 0.0002 | | | | 0.37 (3,769) | 0.80 (8,719) |

Each cell is gap in points, then evaluations in brackets.

Why "eps only" is so much smaller at the same `eps`: in the other variants a board state above the threshold expands all its chosen moves, so most of the tree is a fringe of board states that are themselves under the threshold. Under "eps only" no board state under the threshold is ever evaluated, so the tree has at most 1/`eps` board states per depth.

The fair comparison is therefore at equal cost, not equal `eps`:

| Comparison | Evaluations | Difference in gap (first minus second) |
| --- | --- | --- |
| eps only at 0.01 vs top 8 at 0.01 | 77 vs 514 | +0.41 ± 0.22 |
| eps only at 0.002 vs top 8 at 0.01 | 382 vs 514 | +0.00 ± 0.14 |
| eps only at 0.002 vs top 8 at 0.002 | 382 vs 2,496 | +0.07 ± 0.16 |
| eps only at 0.0005 vs top 8 at 0.002 | 1,521 vs 2,496 | −0.11 ± 0.16 |
| all moves at 0.01 vs top 8 at 0.01 | 2,389 vs 514 | −0.14 ± 0.18 |

Reading: "eps only" reaches the same gap as "top 8" for fewer evaluations at every level tried. "Top 8" stops improving below `eps` 0.01 (0.57, 0.55, 0.50) because the moves it never follows put a floor under it. Following every legal move costs about five times more than "top 8" for a gain inside the error bar.

**Against real outcomes**, on the 962 games (4,810 board states) that a "top 8" run at `eps` 0.01 and an "eps only" run at 0.002 both cover:

| Value estimate | Evaluations | Value accuracy | Value loss | Brier score | MSE of value | MSE vs top 8, x1000 (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| Value head | 1 | 64.8% | 0.7169 | 0.4437 | 0.1910 | +3.0 (+1.4 to +4.6) |
| top 8, eps 0.01 | 501 | 65.7% | 0.7068 | 0.4383 | 0.1880 | |
| eps only, eps 0.01 | 80 | 65.6% | 0.7081 | 0.4388 | 0.1884 | +0.4 (+0.0 to +0.8) |
| eps only, eps 0.002 | 389 | 65.6% | 0.7074 | 0.4384 | 0.1880 | −0.0 (−0.3 to +0.2) |

"Eps only" at 0.002 is indistinguishable from "top 8" at 0.01 with 22% fewer evaluations, and at 0.01 it gives up 0.4 x 1000 of MSE for a sixth of the cost. The "top 8" run was stopped at 962 of 2,000 games once this was clear; its rows are kept in `evals/results/top8/`.

**Decision.** "Eps only" is the algorithm. One setting, no hand-picked constants, an exact bound on the tree's width, and no loss against either yardstick. `max_depth` (40 moves) remains as a safety stop, not as a tuning setting.

## 3. Renormalise or not

When some moves are left out, the weights of the followed moves no longer sum to 1. There are two ways to close the gap: scale the followed moves up (renormalise), or give the missing weight to something else (here, the board state's own value head).

Renormalising says "the moves left out are worth the same as the average of the moves followed". That is wrong in a consistent direction, because the moves left out are the unlikely ones, and unlikely moves are worse moves ([section 4](#4-what-the-moves-left-out-are-worth)). The data:

| Comparison | Difference in gap |
| --- | --- |
| eps only, renormalised vs eps only, at 0.01 | +0.50 ± 0.43 |
| eps only, renormalised vs eps only, at 0.002 | +0.44 ± 0.27 |
| eps only, renormalised vs eps only, at 0.0005 | +0.46 ± 0.22 |
| top 8, not renormalised vs top 8, at 0.01 | +0.01 ± 0.16 |
| top 8, not renormalised vs top 8, at 0.002 | −0.09 ± 0.14 |

With "eps only" the weight left out is large near the leaves (a board state at path probability just above `eps` follows almost nothing), so renormalising does real damage: the gap roughly doubles and it uses more than twice the evaluations, because scaled-up weights push more lines over the threshold. With "top 8" only 3% or so is left out at each board state, so it makes no measurable difference either way.

**Decision.** Do not renormalise. The followed moves keep the policy's probabilities and the rest of the weight goes to the board state's value head.

## 4. What the moves left out are worth

The moves a board state leaves out count with that board state's own value head. That is an approximation, and an obvious one to question: the moves left out are the unpopular ones, so they should be worth less to the player to move than the board state as a whole. Three corrections were tried.

### 4.1 Deriving their value from the value head

If the value head equals the policy-weighted average of all moves, the left-out moves' value follows from the followed ones: `tail = (head − sum of followed weight x value) / weight left out`. Worked example: head 0.5, followed moves 90% of the weight at 0.7, so the remaining 10% would be worth (0.5 − 0.63) / 0.1 = −1.3, which is not a value. The formula divides the value head's error by a small weight, so it amplifies noise; clipped to the valid range it was about twice as far from the reference as the plain rule (for example +0.87 ± 0.53 points worse at `eps` 0.002). These numbers are from the working session; the raw output of this variant was not kept.

### 4.2 Measuring their value

`evals/tail_value.py` (400 board states) and `evals/tail_deep.py` (300 board states, every legal move given its own deep value with an "eps only" tree at `eps` 0.01) measure what the moves under a policy cut-off are worth. Values are for the player to move, in points, relative to the board state's own value head; table from `evals/tail_deep_fit.py` on 299 board states:

| Cut-off | Weight left out | Left-out moves, value head one move later | Left-out moves, deep value | All moves, value head one move later | All moves, deep value |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 32.0% | −8.0 ± 1.3 | −6.3 ± 1.2 | −2.2 | −1.1 ± 0.7 |
| 0.1 | 19.3% | −9.8 ± 1.3 | −7.9 ± 1.2 | −2.4 | −1.3 ± 0.7 |
| 0.05 | 11.1% | −10.5 ± 1.3 | −8.6 ± 1.1 | −2.4 | −1.2 ± 0.7 |
| 0.02 | 4.9% | −12.5 ± 1.4 | −10.3 ± 1.2 | −2.4 | −1.2 ± 0.7 |
| 0.01 | 2.8% | −14.3 ± 1.5 | −12.2 ± 1.3 | −2.4 | −1.3 ± 0.7 |
| 0.005 | 1.4% | −15.6 ± 1.5 | −13.3 ± 1.3 | −2.3 | −1.2 ± 0.7 |
| 0.002 | 0.6% | −17.2 ± 1.5 | −14.8 ± 1.3 | −2.3 | −1.2 ± 0.7 |
| 0.001 | 0.3% | −19.0 ± 1.5 | −16.4 ± 1.4 | −2.3 | −1.2 ± 0.7 |

So the suspicion is right: the left-out moves are worth 6 to 16 points less than the board state's value head, more as the cut-off tightens, and the spread from one board state to the next is large (a standard deviation of 12 to 15 points).

Two straight-line fits against log10(1 / cut-off), with no theory behind the shape:

- **one move later:** discount = 5.4 + 5.1 x log10(1 / cut-off) points (first measurement, 400 board states)
- **deep:** discount = 3.2 + 4.3 x log10(1 / cut-off) points (cut-offs of 0.2 and under, residual 0.2 points; the one game shared with the 30 test board states left out of the fit)

### 4.3 Applying the discount

`evals/tail_discount.py` values the left-out moves at the board state's value head minus the fitted discount, with the cut-off at each board state being `eps` divided by its path probability. Gap to the reference on the 30 board states:

| eps | Evaluations | Value head (the rule in use) | One-move-later discount | Deep discount | Deep discount minus value head |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 4 | 1.84 | 2.82 | 2.43 | +0.58 ± 0.63 |
| 0.15 | 6 | 1.72 | 2.52 | 2.21 | +0.49 ± 0.58 |
| 0.1 | 8 | 1.71 | 2.39 | 1.97 | +0.26 ± 0.53 |
| 0.07 | 11 | 1.54 | 2.44 | 2.15 | +0.60 ± 0.46 |
| 0.05 | 16 | 1.40 | 1.98 | 1.73 | +0.34 ± 0.46 |
| 0.03 | 26 | 1.39 | 1.64 | 1.46 | +0.07 ± 0.40 |
| 0.02 | 38 | 1.31 | 1.31 | 1.24 | −0.07 ± 0.36 |
| 0.015 | 51 | 1.16 | 1.23 | 1.15 | −0.01 ± 0.29 |
| 0.01 | 77 | 0.98 | 0.88 | 0.87 | −0.11 ± 0.25 |
| 0.007 | 109 | 0.95 | 0.76 | 0.79 | −0.15 ± 0.20 |
| 0.005 | 152 | 0.85 | 0.77 | 0.78 | −0.07 ± 0.20 |
| 0.003 | 253 | 0.69 | 0.70 | 0.67 | −0.02 ± 0.17 |
| 0.002 | 382 | 0.57 | 0.63 | 0.59 | +0.02 ± 0.16 |

A flat 8-point discount behaved the same way (worse above `eps` 0.03, no different below).

The discounted trees are biased against the player to move when the tree is shallow. Signed error from the mover's side, in points:

| eps | Value head | One-move-later discount | Deep discount |
| --- | --- | --- | --- |
| 0.2 | −0.13 ± 0.92 | −1.45 ± 1.29 | −1.04 ± 1.10 |
| 0.1 | +0.00 ± 0.75 | −1.05 ± 1.06 | −0.76 ± 0.91 |
| 0.05 | −0.36 ± 0.61 | −0.86 ± 0.92 | −0.75 ± 0.78 |
| 0.01 | −0.43 ± 0.41 | −0.34 ± 0.39 | −0.38 ± 0.36 |
| 0.002 | −0.21 ± 0.25 | −0.22 ± 0.27 | −0.23 ± 0.24 |

### 4.4 Why a correct discount does not help

This is an interpretation that fits the tables above; it was not tested directly.

- The value head runs a little high for whoever is to move ([section 5](#5-the-value-head-leans-towards-the-player-to-move)).
- In a shallow tree the leaves are board states where the opponent is to move, so their value heads lean towards the opponent: the followed part of the tree is too low for the root's mover.
- The plain rule values the left-out moves at the root's own value head, which is too high for the mover. The two errors offset.
- A correct discount removes one of the two errors and leaves the other exposed. The one-move-later fit also carried the opponent's lean inside it, which is why it over-corrected; the deep fit removed about a third of the damage.
- In a deep tree both players have many moves, the leans alternate and cancel, and the left-out weight per board state is under 1%. The rule for the left-out moves then stops mattering, which is what the rows at `eps` 0.02 and below show.

**Decision.** Keep the board state's own value head for the moves left out. It needs no fitted constants, and nothing tried beats it at the thresholds the algorithm runs at. A cleaner fix for shallow trees would be to correct the leaves (shift each leaf's value head about a point against its mover) rather than the left-out moves; that was not tested, because it only matters above `eps` 0.03.

## 5. The value head leans towards the player to move

Three independent measurements agree in sign and rough size:

| Measurement | Value head | One move ahead | Deep value |
| --- | --- | --- | --- |
| Against real outcomes, 10,000 board states: mean of (value − result) from the mover's side, points | +0.89 (+0.10 to +1.71) | −1.21 (−2.01 to −0.41) | −0.20 (−0.99 to +0.59) |
| Against the deep value of all moves, 299 board states: the value head is higher by | 1.2 ± 0.7 | | |
| One move ahead minus the value head, 299 board states | | −2.3 | |

The value head is about a point too kind to the mover, and its average one move later (where the opponent is to move) is about a point too harsh. The deep value sits between them and is not detectably off. This is a small effect next to the 4-point average size of the tree's correction, but it explains section 4.4, and it is why "one move ahead" is not a safe improvement on the value head.

## 6. The default threshold

From the main run (10,000 board states, one tree at `eps` 0.002 that also answers every looser threshold):

| eps | Model evaluations per board state | MSE of value minus the value head's, x1000 (95%) | Mean gap to the tree at 0.002 (points) |
| --- | --- | --- | --- |
| one move ahead | about 32 | −1.3 (−2.3 to −0.3) | 3.22 |
| 0.5 | 2 | −1.6 (−2.3 to −0.8) | 3.23 |
| 0.2 | 4 | −2.7 (−3.6 to −1.9) | 2.39 |
| 0.1 | 8 | −3.0 (−4.0 to −2.0) | 1.87 |
| 0.05 | 16 | −3.3 (−4.4 to −2.4) | 1.43 |
| 0.02 | 40 | −3.5 (−4.6 to −2.5) | 0.94 |
| 0.01 | 80 | −3.6 (−4.8 to −2.6) | 0.64 |
| 0.005 | 159 | −3.8 (−4.9 to −2.7) | 0.38 |
| 0.002 | 390 | −3.9 (−5.0 to −2.7) | 0 |

Paired on the same board states, the tree at 0.002 beats the tree at 0.01 by 0.2 x 1000 of MSE (−0.4 to −0.0), and 0.01 beats 0.05 by 0.3 (−0.6 to +0.0). Each tightening still helps, by less each time.

Finer than the default, from the deep sweep (500 board states from 100 games, one tree each at `eps` 0.0005):

| eps | Model evaluations | Mean gap to the tree at 0.0005 (points) | 95th percentile gap | MSE of value minus the value head's, x1000 |
| --- | --- | --- | --- | --- |
| 0.01 | 80 | 0.87 | 2.3 | −5.5 |
| 0.005 | 159 | 0.60 | 1.6 | −5.8 |
| 0.002 | 393 | 0.35 | 0.9 | −6.1 |
| 0.001 | 781 | 0.18 | 0.5 | −6.1 |
| 0.0005 | 1,550 | 0 | 0 | −6.2 |

Each halving of `eps` doubles the cost and roughly halves the distance to the finest tree. Below 0.002 the value moves by a third of a point on average and the fit to real outcomes no longer changes (the intervals on 500 board states are about ±5, so this run can only show the absence of a large effect).

**Decision.** Default `eps` = 0.002. Looser thresholds remain a reasonable choice when cost matters: `eps` 0.01 keeps most of the gain for about a fifth of the evaluations.

## 7. Metrics

- **The estimate is a (win, draw, loss) triple,** like the value head's output, and the triple is averaged up the tree. One number (win + draw / 2) is derived from it when needed.
- **Value accuracy and value loss** are the Maia-3 paper's metrics for the value head (its "game outcome prediction accuracy", and the cross-entropy it is trained on), so results can be set next to the paper's.
- **The three-outcome Brier score** scores the whole triple with a bounded penalty.
- **MSE of value** scores only the one number. It ignores how probability is split between a draw and a decisive game, which makes it the least noisy of the four on blitz games, where 5% of games are drawn. Slices and thresholds are reported with it.
- **The unit is the game, not the board state.** Five board states of one game share one result, so intervals come from resampling games.
- **Raw rows are kept** (`evals/results/<run>/*.jsonl`): every estimate's full triple per board state, with ratings, moves played so far (`ply`), clock and how the game ended. `evals/report.py` rebuilds every table from them.

## 8. Checks along the way

| Check | Result |
| --- | --- |
| This repo's engine wrapper against the one the algorithm was developed with | Outputs agree to 8e-8 |
| This repo's tree against the video project's original implementation, same settings | Same values on the board states compared |
| CPU against GPU (`mps`) | Values agree to 4 decimals |
| Speed, `maia3-79m` on an Apple M-series laptop | About 125 board states a second on CPU, 215 on the GPU; more threads or processes on the CPU do not help |
| The simplified tree against the stored rows of the main run | Same values to 4 decimals on the rows compared |
| Evaluation games | Lichess, October 2026: after the period the model was trained on (January 2023 to July 2025) |

Two things that went wrong and were fixed:

- **Tree sizes at looser thresholds were overcounted** in the first version of the "eps only" back-up (it counted every child of a followed board state, not only the children that threshold follows). Values were not affected. The field was removed from the main run's rows, and sizes per threshold in the reports come from the sweep run, which was made with the corrected code. An early reading of "225 evaluations at `eps` 0.01" came from this; the right number is about 80.
- **One-hop in "eps only" trees** is the average over the root's moves that are in the tree; moves under `eps` count with the root's value head. At the default threshold those moves carry about 0.6% of the weight.

## 9. Self-consistency: the larger difference

The fit to real outcomes improves by about 2% of MSE (section 6), which is modest. The larger difference is that the value head zig-zags from move to move and the deep value does not. [consistency.md](consistency.md) has the measurements in full and [experiment-log.md](experiment-log.md) how the measures were chosen, including two that were dropped.

**The value head averaged k moves ahead** (`evals/lookahead.py`, 300 board states, points for the player to move): every even k is 2.4 to 2.8 points above every odd k, out to k = 8, and the deep value sits 1.2 to 1.4 points from each. So no fixed depth removes the lean of section 5.

**Along 60 whole games** (4,425 board states, before every move, `evals/run_games.py`), in points for the player who moved:

| Check | Value head | One move ahead | Deep value, eps 0.05 | eps 0.01 | eps 0.002 |
| --- | --- | --- | --- | --- | --- |
| A move changes the mover's value by (should be 0) | −2.38 | +3.07 | +0.21 | +0.50 | +0.53 |
| Mean absolute change per move | 5.49 | 5.48 | 2.63 | 2.63 | 2.58 |
| Correlation of one change with the next (should be 0) | −0.61 | −0.62 | −0.04 | −0.05 | −0.05 |
| Squared changes over the game / squared distance from first value to result (should be 1) | 2.16 | 2.17 | 0.89 | 0.89 | 0.86 |

**On 299 board states where every legal move got a tree of its own** (`evals/tail_deep.py`, deep value at eps 0.01):

| Check | Value head | Deep value |
| --- | --- | --- |
| Mean absolute gap between the value and the average of its moves' values | 4.77 | 0.58 |
| Every legal move lowers the value | 35.1% | 3.0% |
| Every legal move lowers it by over 2 points | 22.7% | 0.0% |

What to take from it:

- **The cause is the lean of section 5.** The value head rates the player to move too high, so its value falls on their move and recovers on the opponent's. That is the −0.61 correlation, and it doubles the movement over a game.
- **One move ahead flips the lean and keeps the zig-zag,** and so does any fixed depth: every line of a fixed-depth average ends with the same player to move.
- **The deep value keeps a small drift of about +0.5 points per move.** Its leaves are value heads at mixed depths, so the leans mostly cancel, not fully. It is a fifth of the value head's drift and has the other sign.
- **The deep value moves slightly less than the result justifies (0.86 to 0.89; the interval reaches 1).** With 60 games this is not evidence of a fault.
- **Consistency needs far fewer evaluations than the last bit of accuracy:** eps 0.05 (16 evaluations) already has no zig-zag.
- **The midpoint (the average of the value head and one move ahead) also cancels the lean,** at about 32 evaluations, but is less accurate than the tree from 4 evaluations up (experiment log, section 5).
- **This is partly by construction.** The deep value is an average over its moves, so it should agree with them. The checks are still fair, because the value after a move comes from a separate tree with its own budget.

## 10. Not done

- **Leaf correction** for the value head's lean towards the mover (section 4.4): untested.
- **The rest of the "top 8" run:** stopped at 962 of 2,000 games; the comparison was already tight.
- **Games lost on time:** the deep value does no better than the value head there ([evals.md](evals.md)), which is expected, since neither sees the clock. Nothing was tried.
- **Other checkpoints:** everything here is `maia3-79m`.

## Rerunning

```
python evals/truncation.py --positions 30 --device mps      # section 2 and 3 (uses evals/variants.py)
python evals/truncation_report.py
python evals/tail_value.py --positions 400                  # section 4.2, one move later
python evals/tail_deep.py --positions 300 --device mps      # section 4.2, deep
python evals/tail_deep_fit.py
python evals/tail_discount.py mps                           # section 4.3
python evals/report.py                                      # sections 2 (real outcomes), 5 and 6
python evals/lookahead.py --positions 300 --device mps     # section 9
python evals/run_games.py games --games 60 --eps 0.002
python evals/consistency.py
```

`evals/variants.py` is the earlier, more general tree with all the options that were compared. The package itself has only the rule that was chosen.
