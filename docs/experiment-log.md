# Experiment log

What was tried, in order, with the numbers and where the raw output is. Newest entries are at the bottom.
[design-notes.md](design-notes.md) has the earlier experiments behind the algorithm itself (which moves to
follow, the value of the moves left out, the default threshold); this log starts where those end and is the
working record behind the results section of the [README](../README.md).

Conventions: a move is one player's turn (a ply). A value is win + draw / 2, as a percentage. A point is one percentage point of that. "For the mover" means from the side of
the player to move at the board state in question. All runs use `maia3-79m` at the two players' real ratings.

| Estimate | What it is | Model evaluations |
| --- | --- | --- |
| Value head (zero-hop) | The value head on the board state | 1 |
| One move ahead (one-hop) | The value head after each legal move, weighted by the policy | about 32 |
| Midpoint | The plain average of the two rows above | about 32 |
| k moves ahead | The value head k moves later, weighted by the policy along every line | about 30^k |
| Deep value | This repo's tree, at a threshold `eps` | 2 (eps 0.5) to 390 (eps 0.002) |

## 1. Accuracy against real outcomes (2026-11-24)

10,000 board states from 2,000 games (`evals/results/main`, tables in [evals.md](evals.md)).

- Deep value at eps 0.002: MSE of value 0.1911 against the value head's 0.1949. The difference is 0.0039
  (95% interval 0.0028 to 0.0050), about 2%.
- **Reading:** a real but modest gain. This prompted the question below: is the larger benefit consistency
  inside a game, not accuracy?

## 2. First consistency metrics (2026-11-25)

Scripts: `evals/run_games.py` (every board state of 60 games, 4,425 board states, `evals/results/games`) and
`evals/consistency.py` (writes [consistency.md](consistency.md)).

Idea: a sound forecast, before a move, equals the average of what it will say after the move. So a move should
not change the mover's value on average, and one change should not predict the next.

| Check | Value head | One move ahead | Deep value, eps 0.002 |
| --- | --- | --- | --- |
| A move changes the mover's value by, on average (should be 0) | −2.38 | +3.07 | +0.53 |
| Correlation of one change with the next (should be 0) | −0.61 | −0.62 | −0.05 |
| Squared changes over the game / squared distance from first value to result (should be 1) | 2.16 | 2.17 | 0.86 |
| Likeliest move is played and the value drops over 2 points | 42.2% | 12.0% | 4.6% |

On 299 board states where every legal move got its own tree (`evals/results/tail_deep`, eps 0.01): the value
head rates the board state above the result of every legal move in 35.1% of them (by over 2 points in 22.7%);
the deep value in 3.0% (never by over 2 points).

**Objection (Quarup):** the "likeliest move is played and the value drops" row is not a contradiction. The
likeliest move can truly lose value, most of all at low ratings where the likeliest move is often a mistake.
**Accepted.** That row is to be dropped from the README. The rows that do not depend on which move was played
(drift, correlation, every-move-lowers) stand. Two replacement metrics were proposed; sections 3 and 4.

## 3. Metric 2: variance along a game (2026-11-25)

Proposal: take the variance of each estimate's values along a game and average over games; if the deep value
has lower variance and no worse accuracy, it is more consistent at no cost.

From the 60 games; standard deviations in points, pooled over games. The pattern is the same with the first 16
moves left out.

| Estimate | Spread of values over the game | Change from one move to the next | Gap to the average of the two neighbouring board states |
| --- | --- | --- | --- |
| Value head | 15.3 | 7.5 | 6.5 |
| One move ahead | 15.3 | 7.3 | 6.4 |
| Midpoint | 15.0 | 4.7 | 3.4 |
| Deep value, eps 0.05 | 15.2 | 4.9 | 3.5 |
| Deep value, eps 0.002 | 15.0 | 4.7 | 3.4 |

- **The spread of values over the game does not separate the estimates.** It ignores the order of the moves and
  is dominated by the real course of the game. The zig-zag is in it but swamped: 15.3² − 15.0² is about 9, a
  wobble of about 3 points on a 15-point trend.
- **The change per move does separate them,** and is the version of this metric to keep (as a secondary metric).
- **Smoothness alone can be had cheaply.** The midpoint is as smooth as the deep value, because the value head
  leans towards the mover and one move ahead leans towards the opponent, and the average cancels the lean. So
  "smoother and no less accurate" does not single out the deep value; accuracy at equal smoothness does
  (section 5).

These numbers are made by `evals/consistency.py` (the table "Accuracy and movement side by side" in
[consistency.md](consistency.md)).

## 4. Metric 1: the value head averaged k moves ahead (2026-11-25)

Proposal: compare (a) the value head, (b) the weighted average one move ahead, (c) two moves ahead, and so on,
with (z) the deep value. This does not say which is right, but it measures the zig-zag directly, with no game
result and no dependence on which move was played.

Script: `evals/lookahead.py`, output `evals/results/lookahead`. 300 board states, the same ones as
`evals/tail_deep.py`.

- **k = 0, 1, 2 are exact:** every legal move, then every reply (about 1,000 evaluations).
- **k = 3 to 8 are sampled.** Exact is out of reach (about 27,000 evaluations at k = 3 and 800,000 at k = 4).
  500 lines are played on from the board states two moves ahead, each started with its exact path probability,
  each further move drawn from the policy. The estimate is unbiased, and its standard error is kept per board
  state; the noise on one step from k − 1 to k averages 0.8 points. The same draws also estimate k = 2, where
  the exact answer is known: the mean difference over the 300 board states is 0.02 points.
- **Why not average the 1,000 most probable lines:** the lines left out contain the unlikely moves, which are
  mostly mistakes, so they are worth less than the lines kept. Renormalising pretends nobody blunders, and
  valuing the rest with a parent's value head mixes in the other parity, which is the thing being measured.

The run took 91 minutes on the laptop GPU (1.32 million evaluations). Full tables: [consistency.md](consistency.md).

**Averages, 300 board states** (points, for the mover at the sampled board state; ± is a 95% interval):

| k moves ahead | Average minus the deep value (eps 0.002) | Average step from k − 1 |
| --- | --- | --- |
| 0 (value head) | +1.26 ± 0.66 | |
| 1 | −1.17 ± 0.57 | −2.44 ± 0.73 |
| 2 | +1.37 ± 0.45 | +2.54 ± 0.64 |
| 3 | −1.44 ± 0.44 | −2.81 ± 0.62 |
| 4 | +1.17 ± 0.39 | +2.60 ± 0.59 |
| 5 | −1.40 ± 0.38 | −2.56 ± 0.57 |
| 6 | +1.29 ± 0.32 | +2.69 ± 0.59 |
| 7 | −1.43 ± 0.32 | −2.72 ± 0.55 |
| 8 | +1.27 ± 0.31 | +2.71 ± 0.54 |

- **The zig-zag does not fade with more moves ahead.** Every even k is above every odd k by 2.4 to 2.8 points,
  out to 8 moves ahead, and every interval excludes zero. Looking a fixed number of moves ahead does not fix it at any
  depth tried.
- **The deep value sits between the two parities,** 1.2 to 1.4 points from each on average.

**Averages hide the tail** (Quarup asked for percentiles). Absolute swing per board state, points:

| Step | Median | 80th percentile | 90th | 95th | Mean |
| --- | --- | --- | --- | --- | --- |
| k = 0 to 1 (value head to one move ahead) | 3.1 | 7.3 | 12.4 | 15.2 | 4.8 |
| k = 1 to 2 | 2.4 | 5.7 | 10.8 | 13.7 | 4.1 |
| k = 2 to 3 | 2.3 | 6.2 | 9.8 | 13.7 | 4.0 |
| k = 3 to 4 | 2.0 | 5.4 | 9.5 | 13.4 | 3.7 |
| k = 4 to 5 | 1.9 | 5.6 | 9.0 | 13.4 | 3.5 |
| k = 5 to 6 | 1.8 | 5.2 | 8.8 | 13.8 | 3.5 |
| k = 6 to 7 | 1.7 | 5.3 | 9.1 | 12.4 | 3.5 |
| k = 7 to 8 | 1.8 | 5.2 | 8.6 | 13.1 | 3.4 |
| Highest minus lowest over k = 0 to 8 | 6.2 | 12.5 | 16.3 | 21.5 | 8.3 |
| Deep value (eps 0.01): own value to the average of its moves' deep values | 0.4 | 0.9 | 1.3 | 1.8 | 0.6 |

- **In one board state in ten the value head and one move ahead differ by 12 points or more;** the deep value's
  equivalent step is 1.3 points. The last row is the deep value's version of the first: each legal move gets a
  tree of its own (`evals/tail_deep.py`, 299 of the same board states).
- **The tail barely shrinks with depth:** the 95th percentile is 13 to 15 points at every step. (The interim
  reading on 106 board states, that the 90th percentile fell to about 6 by k = 4, did not hold: it is about 9.)
- **The choice of k matters a lot:** the answer moves by 6.2 points or more for half the board states depending
  on which k from 0 to 8 is used, and by 16 points or more for one in ten.
- Steps from k = 2 on include sampling noise of about 0.8 points; the first two rows are exact.

Absolute gap to the deep value at eps 0.002, points:

| Estimate | Median | 80th percentile | 90th | 95th | Mean |
| --- | --- | --- | --- | --- | --- |
| Value head (k = 0) | 2.7 | 6.7 | 9.5 | 11.7 | 4.1 |
| One move ahead (k = 1) | 2.1 | 4.9 | 7.9 | 11.5 | 3.3 |
| k = 2 | 1.8 | 4.1 | 5.9 | 8.5 | 2.8 |
| k = 4 | 1.4 | 3.2 | 5.1 | 6.6 | 2.2 |
| k = 6 | 1.1 | 3.0 | 4.8 | 6.3 | 2.0 |
| k = 8 | 1.2 | 3.1 | 4.6 | 6.4 | 2.0 |
| Midpoint | 1.8 | 4.7 | 7.3 | 8.8 | 2.9 |
| Deep value, eps 0.2 | 1.8 | 4.1 | 5.7 | 7.6 | 2.6 |
| Deep value, eps 0.05 | 1.0 | 2.2 | 3.5 | 4.4 | 1.5 |
| Deep value, eps 0.01 | 0.4 | 1.1 | 1.6 | 2.0 | 0.7 |

- **Looking further ahead closes about half the gap to the deep value by k = 4, then stops,** held at about 2
  points by the parity offset.
- **The midpoint agrees with the deep value on average but not board state by board state:** it is 7 points or
  more away in one in ten.

## 5. Accuracy and zig-zag side by side (2026-11-25)

MSE from the 10,000 board states of the main run; the last two columns from the 60 whole games. The improvement
is the value head's MSE minus the estimate's, times 1,000; intervals are each against the value head.

| Estimate | Evaluations | MSE of value | Improvement over value head, x1000 (95%) | Change per move (points) | Correlation of one change with the next |
| --- | --- | --- | --- | --- | --- |
| Value head | 1 | 0.1949 | 0 | 7.5 | −0.61 |
| One move ahead | about 32 | 0.1937 | 1.3 (0.2 to 2.3) | 7.3 | −0.62 |
| Midpoint | about 32 | 0.1932 | 1.7 (1.2 to 2.3) | 4.7 | 0.00 |
| Deep value, eps 0.5 | 2 | 0.1934 | 1.6 (0.8 to 2.3) | 5.9 | −0.21 |
| Deep value, eps 0.2 | 4 | 0.1922 | 2.7 (1.9 to 3.6) | 5.2 | −0.06 |
| Deep value, eps 0.1 | 8 | 0.1920 | 3.0 (2.0 to 4.0) | 5.0 | −0.03 |
| Deep value, eps 0.05 | 16 | 0.1916 | 3.3 (2.3 to 4.4) | 4.9 | −0.04 |
| Deep value, eps 0.01 | 80 | 0.1913 | 3.6 (2.6 to 4.8) | 4.8 | −0.05 |
| Deep value, eps 0.002 | 390 | 0.1911 | 3.9 (2.8 to 5.0) | 4.7 | −0.05 |

- **This is the accuracy result of section 1 again,** with the zig-zag next to it: the deep value removes the
  zig-zag and gains a little accuracy.
- **The midpoint is not the better cheap option.** It costs about 32 evaluations; a 4-evaluation tree (eps 0.2)
  is more accurate (2.7 against 1.7) with almost no zig-zag (−0.06). The midpoint is smoother per move by a hair
  (4.7 against 4.9 to 5.2 for the small trees).
- **Against the midpoint directly** (improvement in MSE x1000, 95% interval): eps 0.5 −0.1 (−0.7 to 0.4), eps 0.2
  1.0 (0.4 to 1.7), eps 0.1 1.3 (0.6 to 2.0), eps 0.05 1.6 (0.9 to 2.4), eps 0.01 1.9 (1.1 to 2.7), eps 0.002 2.1
  (1.3 to 3.0). From 4 evaluations up the tree is more accurate than the midpoint, and the interval excludes zero.
- One move ahead against the midpoint: −0.4 (−1.0 to 0.1), so the two are not told apart on accuracy.

## 6. Wrap-up (2026-11-25)

- The README was rewritten around four parts (Quarup): the problem with an example, the results (the zig-zag
  by k, the change per move, accuracy), the solution, and the variations explored.
- The likeliest-move row of section 2 was removed from `evals/consistency.py`, the README and the design notes.
- The example in the README's problem section is a stretch of game `klZArXVF` (`ply` 63 to 73 in the data), found by
  searching the 60 whole games for a level position where the value head reverses direction on every move while
  the deep value stays put. Checked with Stockfish 18 at depth 24 (`evals/example_stockfish.py`; drawn on the
  chart's right axis, −5 to +5 pawns with 0 on the 50-point line): the eleven board states are rated −0.36,
  −0.14, −0.12, +0.44, 0.00, 0.00, 0.00, +0.24, +0.18, +0.08 and +0.09 pawns for White, so the game is level
  throughout. (Black later won; White was the one short of time, 54 seconds falling to 29 over the stretch,
  which neither head sees.) It was chosen to illustrate; the whole-game chart in
  [consistency.md](consistency.md) is chosen by rule (median movement).
- **Why our accuracy differs from the Maia-3 paper's** (Quarup's question): different games. The paper scores
  its 5M model at 62.8% value accuracy and 0.736 value loss on 2022 blitz positions after move 10 with time
  pressure removed; ours are `maia3-79m` on October 2026 games balanced by rating, after the first 8 moves, time pressure
  included (63.2% and 0.7360). With the paper's two position filters applied to our board states (6,331 of the
  10,000), the value head scores 65.7% and 0.7029 and the deep value 66.9% and 0.6884. [evals.md](evals.md) has the table.

- **Swing sizes along a game** (for the README's opening statement): over all moves of the 60 games, the value
  head changes by 10 points or more on 16.0% of moves and by 5 or more on 38.8%; the deep value on 4.4% and
  14.5%. The 90th percentile of the change is 12.6 points for the value head and 6.3 for the deep value. The
  value head's change minus the deep value's on the same move is 10 points or more on 14.7% of moves. A
  stricter reading, a change of 10 or more that the next move reverses by 10 or more, covers 8% of moves.

- **Wording:** "move" everywhere means one player's turn, as in the Maia-3 paper, which uses "plies" once (an
  input feature) and never defines its "first 10 moves". Our paper-filter comparison reads that as 10 by each
  player. Code, data fields and the command line keep `ply` for the number of moves played so far.

- **Why the 90th percentile of "k moves ahead minus the value head" does not zig-zag while the 10th does**
  (Quarup's question): the lean is not a uniform shift. Per board state, odd k minus even k has a median of −1.2
  but a 10th percentile of −9.2, and it is the same board states at every k (the step from k = 0 to 1 correlates
  0.78 with the step from 2 to 3). It follows the stage of the game: average −0.0 in the first 20 moves, −0.4 in
  moves 21 to 40, −4.0 in moves 41 to 70 and −11.7 after move 70 (moves of both players together). It does not
  follow how close the game is to ending: board states before move 40 within 10 moves of the end average −0.2.
  The mover's clock goes with it (−0.8 with 2 minutes or more, −10.6 under 30 seconds), though the model has no
  clock input. Not explained. Table "Where the lean is" in [consistency.md](consistency.md).

## 7. The midpoint in the headline results; one set of names (2026-11-26)

- **The midpoint was added to both headline comparisons** (Quarup). On the one-move check its step is exact from
  data already stored: the midpoint is (V0 + V1) / 2, its average after the move is (V1 + V2) / 2, so the step
  is (V2 − V0) / 2. On the 299 board states: 10th percentile −2.0, median +0.1, 90th +2.1; 3.3% of board states
  at 5 points or more and 0.3% at 10 or more. The deep value: −0.9, −0.1, +0.8, and none.
- **Along a game the midpoint moves as little per move as the deep value** (section 5), but on this check its
  step is more than twice as wide. The two measures differ because the along-game one includes the real effect
  of the move played, which is the same for both and hides the difference.
- **Accuracy on the 10,000 board states:** midpoint 63.6%, value loss 0.7276, MSE of value 0.1932 (1.7 better
  than the value head, 1.2 to 2.3); deep value 64.0%, 0.7243, 0.1911 (3.9 better).
- **Names.** Every document and chart now says "value head", "one move ahead", "midpoint" and "deep value".
  "Zero-hop" and "one-hop" remain only as field names in the code and data (`zero_hop`, `one_hop`).
- **The README's results** are now two parts, stability (the one-move check) and accuracy, each comparing the
  value head, the midpoint and the deep value. The along-game table moved to "Variations explored".

## Open items

- Optional check: a "1,000 most probable lines" column on a subset of `evals/lookahead.py`, to show the size of
  its bias.
- Leaf correction for the value head's lean (design notes, section 4.4): still untested.
