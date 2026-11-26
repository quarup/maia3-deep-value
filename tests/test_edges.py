"""Edges of the tree: finished games, perspective, the depth stop, the cache, the JSON form."""
import json

import chess
import pytest

from conftest import make_fake, noisy_wdl, peaked_policy, uniform_policy
from deep_value import DeepValue, Estimate, scalar, terminal_wdl

STALEMATE_IN_ONE = "7k/8/5QK1/8/8/8/8/8 w - - 0 1"            # Qf7 leaves Black with no move and no check


def test_terminal_wdl_is_always_for_white():
    assert terminal_wdl(chess.Board()) is None
    white_mated = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        white_mated.push_san(san)
    assert terminal_wdl(white_mated) == (0.0, 0.0, 1.0)
    assert terminal_wdl(chess.Board("3R2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1")) == (1.0, 0.0, 0.0)
    assert terminal_wdl(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == (0.0, 1.0, 0.0)      # stalemate
    assert terminal_wdl(chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")) == (0.0, 1.0, 0.0)       # bare kings
    assert scalar((0.2, 0.4, 0.4)) == pytest.approx(0.4)


def test_a_draw_inside_the_tree_is_exact():
    forced = lambda x: [("f6f7", 1.0)] if not x.move_stack else uniform_policy(x)
    deep = DeepValue(make_fake(forced, lambda x: 0.95), eps=0.01)
    got = deep.value(chess.Board(STALEMATE_IN_ONE), 1500, 1500)
    assert got.zero_hop == pytest.approx((0.95, 0.0, 0.05))    # the value head sees a won position
    assert got.wdl == (0.0, 1.0, 0.0) and got.value == 0.5     # the policy stalemates
    assert [n["kind"] for n in deep.explain(chess.Board(STALEMATE_IN_ONE), 1500, 1500)] == ["inner", "over"]


def test_values_are_for_white_whoever_is_to_move():
    mover = (0.6, 0.3, 0.1)
    deep = DeepValue(make_fake(uniform_policy, lambda x: mover), eps=1.0)      # nothing followed
    white_to_move, black_to_move = chess.Board(), chess.Board()
    black_to_move.push_san("e4")
    assert deep.value(white_to_move, 1500, 1500).wdl == mover
    assert deep.value(black_to_move, 1500, 1500).wdl == mover[::-1]


def test_a_threshold_no_move_reaches_gives_the_value_head_alone():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.999)
    got = deep.value(chess.Board(), 1500, 1500)
    assert (got.nodes, got.leaves, got.depth, got.moves) == (1, 1, 0, [])
    assert got.wdl == got.zero_hop == got.one_hop
    assert deep.evals == 1


def test_max_depth_stops_a_forced_line():
    forced = lambda x: uniform_policy(x, 1)                    # one move takes the whole policy: no eps stops it
    got = DeepValue(make_fake(forced, lambda x: 0.5), eps=0.5, max_depth=5).value(chess.Board(), 1500, 1500)
    assert (got.depth, got.nodes, got.leaves) == (5, 6, 1)


def test_the_same_board_state_is_never_evaluated_twice():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.05)
    first = deep.value(chess.Board(), 1500, 1600)
    n = deep.evals
    assert n == first.nodes
    again = deep.value(chess.Board(), 1500, 1600)
    assert deep.evals == n and again.wdl == first.wdl
    deep.value(chess.Board(), 1600, 1500)                      # other ratings: other evaluations
    assert deep.evals == 2 * n


def test_history_is_part_of_the_board_state():
    seen = []

    def evaluate(boards, self_elos, oppo_elos):
        seen.extend(len(b.move_stack) for b in boards)
        return make_fake(peaked_policy, noisy_wdl)(boards, self_elos, oppo_elos)

    one, other = chess.Board(), chess.Board()
    for san in ("e4", "e5", "Nf3"):
        one.push_san(san)
    for san in ("Nf3", "e5", "e4"):
        other.push_san(san)
    bare = chess.Board(one.fen())
    assert one.board_fen() == other.board_fen() == bare.board_fen()
    keys = {DeepValue.key(b) for b in (one, other, bare)}
    assert len(keys) == 3                                      # the same position, three different inputs
    deep = DeepValue(evaluate, eps=1.0)
    deep.value_many([(one, 1500, 1500), (other, 1500, 1500), (bare, 1500, 1500)])
    assert sorted(seen) == [0, 3, 3]
    assert DeepValue.key(one) == chess.STARTING_FEN + "|e2e4 e7e5 g1f3"


def test_the_evaluator_gets_boards_with_their_move_stack():
    stacks = []

    def evaluate(boards, self_elos, oppo_elos):
        stacks.extend([m.uci() for m in b.move_stack] for b in boards)
        return make_fake(lambda x: uniform_policy(x, 1), lambda x: 0.5)(boards, self_elos, oppo_elos)

    b = chess.Board()
    b.push_san("d4")
    DeepValue(evaluate, eps=0.5, max_depth=2).value(b, 1500, 1500)
    assert [len(s) for s in stacks] == [1, 2, 3] and all(s[0] == "d2d4" for s in stacks)
    assert len(b.move_stack) == 1                              # the caller's board is left as it was


def test_an_object_with_an_evaluate_method_works_as_the_evaluator():
    class Model:
        evaluate = staticmethod(make_fake(peaked_policy, noisy_wdl))

    a = DeepValue(Model(), eps=0.05).value(chess.Board(), 1500, 1500)
    b = DeepValue(Model.evaluate, eps=0.05).value(chess.Board(), 1500, 1500)
    assert a.wdl == b.wdl


def test_after_values_a_move_the_policy_would_not_follow():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.05)
    b = chess.Board()
    rare = b.san(chess.Move.from_uci(peaked_policy(b)[-1][0]))
    assert rare not in [m["san"] for m in deep.value(b, 1500, 1500).moves]
    nb = b.copy()
    nb.push_san(rare)
    assert deep.after(b, rare, 1500, 1500).wdl == deep.value(nb, 1500, 1500).wdl
    assert not b.move_stack


def test_the_reported_moves_are_the_likeliest_and_carry_the_value_after_them():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.001)
    b = chess.Board()
    got = deep.value(b, 1500, 1500)
    assert len(got.moves) == 6
    assert [m["p"] for m in got.moves] == sorted((m["p"] for m in got.moves), reverse=True)
    top = got.moves[0]
    assert top["p"] == pytest.approx(peaked_policy(b)[0][1], abs=1e-4)
    sub = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.001 / peaked_policy(b)[0][1])
    nb = b.copy()
    nb.push_san(top["san"])
    assert top["wdl"] == pytest.approx(sub.value(nb, 1500, 1500).wdl)          # the subtree is a tree of its own


def test_a_setting_outside_the_tree_is_refused():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.05, max_depth=10)
    with pytest.raises(ValueError):
        deep.value(chess.Board(), 1500, 1500, also=[(0.01, 10)])               # finer than the tree
    with pytest.raises(ValueError):
        deep.value(chess.Board(), 1500, 1500, also=[(0.05, 11)])               # deeper than the tree


def test_an_estimate_goes_to_json_and_back():
    deep = DeepValue(make_fake(peaked_policy, noisy_wdl), eps=0.01)
    got = deep.value(chess.Board(), 1500, 1500, also=[(0.05, 40)])
    doc = json.loads(json.dumps(got.to_json()))
    assert doc["value"] == pytest.approx(got.value, abs=1e-4)
    assert doc["wdl"] == pytest.approx(list(got.wdl), abs=1e-4)
    assert (doc["nodes"], doc["leaves"], doc["depth"]) == (got.nodes, got.leaves, got.depth)
    assert list(doc["also"]) == ["0.05/40"]
    assert all(set(m) == {"san", "p", "wdl", "value"} for m in doc["moves"])
    assert "also" not in Estimate((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)).to_json()
