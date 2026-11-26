"""The tree against the README's recursion written out naively, and against numbers worked by hand."""
from collections import Counter

import chess
import pytest

from conftest import make_fake, noisy_wdl, peaked_policy, uniform_policy
from deep_value import DeepValue, terminal_wdl

MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1"
ENDGAME = "8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40"


def reference(evaluate, board, eps, max_depth, q=1.0, depth=0):
    """D(n, q) of the README, one board state at a time and with no cache: White's (win, draw, loss)."""
    over = terminal_wdl(board)
    if over is not None:
        return over
    e = evaluate([board], [0], [0])[0]
    own = (e.win, e.draw, e.loss) if board.turn == chess.WHITE else (e.loss, e.draw, e.win)
    total, followed = [0.0, 0.0, 0.0], 0.0
    if depth < max_depth:
        for uci, p in e.policy:
            if q * p >= eps:
                nb = board.copy()
                nb.push(chess.Move.from_uci(uci))
                below = reference(evaluate, nb, eps, max_depth, q * p, depth + 1)
                total = [t + p * x for t, x in zip(total, below)]
                followed += p
    return tuple(t + (1.0 - followed) * o for t, o in zip(total, own))


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, MIDGAME, BACK_RANK, ENDGAME])
@pytest.mark.parametrize("eps,max_depth", [(0.2, 40), (0.03, 40), (0.005, 40), (0.005, 3), (0.0, 2)])
def test_the_tree_equals_the_recursion_written_out(fen, eps, max_depth):
    evaluate = make_fake(peaked_policy, noisy_wdl)
    got = DeepValue(evaluate, eps=eps, max_depth=max_depth).value(chess.Board(fen), 1500, 1500)
    assert got.wdl == pytest.approx(reference(evaluate, chess.Board(fen), eps, max_depth))
    # one-hop as the tree reports it: the root's moves under eps count with the root's value head
    assert got.one_hop == pytest.approx(reference(evaluate, chess.Board(fen), eps, 1))
    assert sum(got.wdl) == pytest.approx(1.0) and min(got.wdl) >= 0.0


def test_one_hop_is_the_policy_weighted_value_head_one_move_later():
    evaluate = make_fake(peaked_policy, noisy_wdl)
    b = chess.Board(MIDGAME)
    got = DeepValue(evaluate, eps=0.0, max_depth=2).value(b, 1500, 1500)     # eps 0: every move is in the tree
    want = reference(evaluate, b, 0.0, 1)
    assert got.one_hop == pytest.approx(want)
    assert got.zero_hop == pytest.approx(reference(evaluate, b, 0.0, 0))


def test_a_tree_worked_by_hand():
    # White: move A 0.9, move B 0.1. After A, Black: two moves at 0.5 each. eps 0.4 follows A (0.9) and both
    # replies (0.45 each); B (0.1) and everything a move further (0.225) are left out.
    start = chess.Board()
    a, b_ = sorted(m.uci() for m in start.legal_moves)[:2]
    after_a = start.copy()
    after_a.push(chess.Move.from_uci(a))
    c, d = sorted(m.uci() for m in after_a.legal_moves)[:2]

    def policy(x):
        if not x.move_stack:
            return [(a, 0.9), (b_, 0.1)]
        return uniform_policy(x, 2)

    def mover_value(x):
        line = [m.uci() for m in x.move_stack]
        return {(): 0.6, (a,): 0.3, (a, c): 1.0, (a, d): 0.6}[tuple(line)]       # after A Black has 0.3: White 0.7

    deep = DeepValue(make_fake(policy, mover_value), eps=0.4)
    got = deep.value(start, 1500, 1500)
    assert got.zero_hop == pytest.approx((0.6, 0.0, 0.4))
    assert got.one_hop[0] == pytest.approx(0.9 * 0.7 + 0.1 * 0.6)                # B counts with the root's value head
    assert got.value == pytest.approx(0.9 * (0.5 * 1.0 + 0.5 * 0.6) + 0.1 * 0.6)
    assert (got.nodes, got.leaves, got.depth) == (4, 2, 2)
    assert [m["p"] for m in got.moves] == [0.9]                                  # only the move that was followed
    assert got.moves[0]["wdl"][0] == pytest.approx(0.8)


@pytest.mark.parametrize("eps", [0.3, 0.05, 0.01, 0.002])
def test_the_tree_is_at_most_one_over_eps_wide_at_every_depth(eps):
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=eps)
    nodes = deep.explain(chess.Board(MIDGAME), 1500, 1500)
    assert all(n["q"] >= eps for n in nodes)
    per_depth, weight = Counter(), Counter()
    for n in nodes:
        per_depth[n["depth"]] += 1
        weight[n["depth"]] += n["q"]
    assert max(per_depth.values()) <= 1 / eps
    assert max(weight.values()) <= 1.0 + 1e-9                  # the board states at one depth are disjoint events
    for n in nodes[1:]:
        assert n["q"] == pytest.approx(nodes[n["parent"]]["q"] * n["p"])
        assert n["depth"] == nodes[n["parent"]]["depth"] + 1


def test_a_looser_threshold_gives_a_smaller_tree():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.002)
    ladder = [(0.002, 40), (0.01, 40), (0.05, 40), (0.2, 40), (1.0, 40)]
    got = deep.value(chess.Board(MIDGAME), 1500, 1500, also=ladder)
    sizes = [got.also_nodes[s] for s in ladder]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == got.nodes and sizes[-1] == 1
    assert got.also[(0.002, 40)] == pytest.approx(got.wdl)
    assert got.also[(1.0, 40)] == pytest.approx(got.zero_hop)  # nothing followed: the value head alone


def test_the_deep_value_stays_inside_the_range_of_the_value_heads():
    lo, hi = 0.35, 0.65
    value = lambda x: lo + (hi - lo) * noisy_wdl(x)[0]         # the mover's value, always inside [lo, hi]
    got = DeepValue(make_fake(peaked_policy, value), eps=0.01).value(chess.Board(MIDGAME), 1500, 1500)
    assert 1 - hi <= got.value <= hi                           # an average of value heads cannot leave their range
    for m in got.moves:
        assert 1 - hi <= m["wdl"][0] <= hi
