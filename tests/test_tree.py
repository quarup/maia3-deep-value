"""The tree on a hand-made evaluator: no model needed."""
import chess
import pytest

from conftest import make_fake as fake
from conftest import uniform_policy as uniform
from deep_value import DeepValue


def back_rank(x):
    # Black's likely move walks into a mate in one that White always plays
    if x.turn == chess.BLACK and x.fullmove_number == 1:
        return [("g8h8", 0.8), ("h7h6", 0.2)]                  # likeliest first: Kh8 leaves the back rank mate, h6 makes room
    if x.turn == chess.WHITE and x.piece_at(chess.D1):
        return [("d1d8", 1.0)]
    return uniform(x)


BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1"


def test_a_likely_mate_two_plies_down_is_found_where_the_head_sees_nothing():
    d = DeepValue(fake(back_rank, lambda x: 0.5), eps=0.01)
    got = d.value(chess.Board(BACK_RANK), 1300, 1300)
    assert got.zero_hop == (0.5, 0.0, 0.5)                     # the head calls it level
    assert got.one_hop == (0.5, 0.0, 0.5)                      # and so does one step: the mate is two plies down
    by = {m["san"]: m for m in got.moves}
    assert by["Kh8"]["wdl"] == (1.0, 0.0, 0.0)                 # Rd8 is mate
    assert got.value > 0.85                                    # 0.8 for the mate, the rest near one half
    assert sum(got.wdl) == pytest.approx(1.0)


def test_a_line_stops_when_its_path_probability_falls_under_eps():
    calls = []

    def policy(x):
        calls.append(len(x.move_stack))
        return uniform(x, 2)                                   # every step halves the path probability

    d = DeepValue(fake(policy, lambda x: 0.5), eps=0.2)
    got = d.value(chess.Board(), 1300, 1300)
    # 1 -> 0.5 -> 0.25 are followed (at least 0.2); the moves after those, at 0.125, are never evaluated
    assert got.depth == 2 and got.nodes == 7 and got.leaves == 4 and got.value == 0.5
    assert max(calls) == 2


def test_positions_of_one_game_share_their_evaluations():
    d = DeepValue(fake(lambda x: uniform(x, 2), lambda x: 0.5), eps=0.2)
    b = chess.Board()
    d.value(b, 1300, 1300)
    first = d.evals
    b.push(chess.Move.from_uci(sorted(m.uci() for m in b.legal_moves)[0]))   # a move the tree already followed
    d.value(b, 1300, 1300)
    assert d.evals - first < first                             # most of the second tree was already read


def test_looser_settings_come_from_the_same_tree():
    value = lambda x: 0.3 + 0.4 * (hash(x.fen()) % 97) / 96    # some value that differs by position
    fine = DeepValue(fake(back_rank, value), eps=0.01)
    got = fine.value(chess.Board(BACK_RANK), 1300, 1300, also=[(0.3, 40), (0.05, 40), (0.01, 2), (0.01, 1)])
    for (eps, depth), wdl in got.also.items():
        alone = DeepValue(fake(back_rank, value), eps=eps, max_depth=depth).value(chess.Board(BACK_RANK), 1300, 1300)
        assert wdl == pytest.approx(alone.wdl)
        assert got.also_nodes[(eps, depth)] == alone.nodes
    assert got.also[(0.01, 1)] == pytest.approx(got.one_hop)
    with pytest.raises(ValueError):
        fine.value(chess.Board(BACK_RANK), 1300, 1300, also=[(0.001, 40)])


def test_many_positions_at_once_match_one_at_a_time_and_keep_their_ratings():
    seen = []

    def evaluate(boards, self_elos, oppo_elos):
        seen.extend(zip((b.turn for b in boards), self_elos, oppo_elos))
        return fake(lambda x: uniform(x, 3), lambda x: 0.6)(boards, self_elos, oppo_elos)

    a, b = chess.Board(), chess.Board()
    b.push_san("e4")
    together = DeepValue(evaluate, eps=0.1).value_many([(a, 1000, 2000), (b, 1500, 1700)])
    apart = [DeepValue(evaluate, eps=0.1).value(a, 1000, 2000), DeepValue(evaluate, eps=0.1).value(b, 1500, 1700)]
    assert [s.wdl for s in together] == [pytest.approx(s.wdl) for s in apart]
    # the side to move always gets its own rating first
    assert set(seen) == {(chess.WHITE, 1000, 2000), (chess.BLACK, 2000, 1000),
                         (chess.WHITE, 1500, 1700), (chess.BLACK, 1700, 1500)}


def test_a_finished_game_is_scored_exactly():
    b = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        b.push_san(san)
    got = DeepValue(fake(uniform, lambda x: 0.5)).value(b, 1300, 1300)
    assert got.wdl == got.zero_hop == (0.0, 0.0, 1.0) and got.value == 0.0


def test_explain_gives_the_tree_behind_a_score():
    d = DeepValue(fake(back_rank, lambda x: 0.5), eps=0.01)
    nodes = d.explain(chess.Board(BACK_RANK), 1300, 1300)
    assert nodes[0]["parent"] is None and nodes[0]["kind"] == "inner" and nodes[0]["q"] == 1.0
    mate = next(n for n in nodes if n["san"] == "Rd8#")
    assert mate["kind"] == "over" and mate["wdl"] == (1.0, 0.0, 0.0) and mate["q"] == pytest.approx(0.8)
    assert nodes[mate["parent"]]["san"] == "Kh8"
    assert all(n["q"] >= 0.01 for n in nodes)


def test_no_board_state_under_the_threshold_is_evaluated_and_left_out_moves_count_with_the_value_head():
    value = lambda x: 0.3 + 0.4 * (hash(x.fen()) % 97) / 96
    d = DeepValue(fake(back_rank, value), eps=0.3)             # of Black's two moves only Kh8 (0.8) is followed
    b = chess.Board(BACK_RANK)
    nodes = d.explain(b, 1300, 1300)
    assert all(n["q"] >= 0.3 for n in nodes)
    assert [n["san"] for n in nodes] == [None, "Kh8", "Rd8#"]
    assert nodes[0]["rest"] == pytest.approx(0.2)
    got = d.value(b, 1300, 1300)
    head = 1.0 - value(b)                                      # Black is to move: White's value of the root
    assert got.value == pytest.approx(0.8 * 1.0 + 0.2 * head)  # the mate, and h6 counted with the root's value head
    assert sum(got.wdl) == pytest.approx(1.0)
