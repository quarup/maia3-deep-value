"""The Maia-3 wrapper: how a board becomes model input and model output becomes an Eval.

The first tests need only numpy. The ones marked with `stub_maia` run the real `Maia.evaluate` on a stand-in
network, so they need torch and the maia3 package but no weights. The last one loads the real model and runs
only when DEEP_VALUE_MODEL_TESTS=1.
"""
import os

import chess
import numpy as np
import pytest

from deep_value.engine import HISTORY, Eval, Maia, _mirror, _piece_planes, history_planes

PAWN, KING = 0, 5                                             # channels: the mover's six piece types, then the opponent's


def test_eval_expected_is_win_plus_half_the_draws():
    assert Eval([], 0.5, 0.2, 0.3).expected == pytest.approx(0.6)


def test_mirror_flips_ranks_and_keeps_files_and_promotions():
    assert _mirror("e7e5") == "e2e4" and _mirror("g8f6") == "g1f3"
    assert _mirror("a2a1q") == "a7a8q"
    assert _mirror(_mirror("b7c8n")) == "b7c8n"


def test_planes_are_seen_from_the_side_to_move():
    start = chess.Board()
    white = _piece_planes(start)
    assert white.shape == (64, 12) and white.sum() == 32
    assert white[chess.E2, PAWN] == 1 and white[chess.E1, KING] == 1          # own pieces: channels 0 to 5
    assert white[chess.E7, 6 + PAWN] == 1 and white[chess.E8, 6 + KING] == 1  # the opponent's: 6 to 11
    after_e4 = start.copy()
    after_e4.push_san("e4")
    black = _piece_planes(after_e4)                            # Black to move: the board is turned round
    assert black[chess.E2, PAWN] == 1 and black[chess.E1, KING] == 1          # Black's own pawn and king, as if White's
    assert black[chess.E5, 6 + PAWN] == 1 and black[chess.E7, 6 + PAWN] == 0  # White's e4 pawn, seen from Black's side
    assert after_e4.turn == chess.BLACK                        # the caller's board is not touched


def test_history_planes_hold_the_last_eight_board_states_oldest_first():
    b = chess.Board()
    assert history_planes(b).shape == (64, 12 * HISTORY)
    alone = history_planes(b)
    for i in range(HISTORY):                                   # no history yet: the same board eight times
        assert np.array_equal(alone[:, 12 * i:12 * (i + 1)], _piece_planes(b))
    boards = [b.copy()]
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "Bc5", "Nxf7", "Bxf2+"):
        b.push_san(san)
        boards.append(b.copy())
    x = history_planes(b)
    for i in range(HISTORY):                                   # slot 7 is the board itself, slot 0 is 7 moves back
        assert np.array_equal(x[:, 12 * i:12 * (i + 1)], _piece_planes(boards[len(boards) - HISTORY + i]))
    short = history_planes(boards[2])                          # 2 moves played: padded at the front with the start
    assert np.array_equal(short[:, :12], _piece_planes(boards[0]))
    assert np.array_equal(short[:, 12 * 5:12 * 6], _piece_planes(boards[0]))
    assert np.array_equal(short[:, 12 * 6:12 * 7], _piece_planes(boards[1]))
    assert len(b.move_stack) == 10


def test_a_bare_fen_and_the_same_position_with_its_moves_are_different_inputs():
    b = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        b.push_san(san)
    assert not np.array_equal(history_planes(b), history_planes(chess.Board(b.fen())))
    assert np.array_equal(history_planes(b)[:, -12:], history_planes(chess.Board(b.fen()))[:, -12:])


# --- Maia.evaluate on a stand-in network ---------------------------------------------------------------

@pytest.fixture
def stub_maia():
    torch = pytest.importorskip("torch")
    utils = pytest.importorskip("maia3.utils")
    index = {u: i for i, u in enumerate(utils.get_all_possible_moves())}
    calls = []

    def network(x, self_elo, oppo_elo):
        calls.append({"shape": tuple(x.shape), "self": self_elo.tolist(), "oppo": oppo_elo.tolist(),
                      "think": float(x[..., -1].abs().sum())})
        logits = torch.zeros(x.shape[0], len(index))
        logits[:, index["e2e4"]] = 3.0                         # in the mover's own frame: its king's pawn two squares
        value = torch.tensor([[0.0, 1.0, 2.0]]).repeat(x.shape[0], 1)          # (loss, draw, win) logits
        return logits, value, None

    m = Maia.__new__(Maia)                                     # skip the download: everything evaluate() needs
    m.torch, m.model, m.device, m.batch, m.index, m.evals = torch, network, "cpu", 2, index, 0
    m.calls = calls
    return m


def test_evaluate_gives_a_policy_over_the_legal_moves_likeliest_first(stub_maia):
    b = chess.Board()
    e = stub_maia.evaluate([b], 1500, 1600)[0]
    assert {u for u, _ in e.policy} == {m.uci() for m in b.legal_moves}
    assert sum(p for _, p in e.policy) == pytest.approx(1.0)
    assert [p for _, p in e.policy] == sorted((p for _, p in e.policy), reverse=True)
    assert e.policy[0][0] == "e2e4"
    assert e.policy[0][1] == pytest.approx(np.exp(3) / (np.exp(3) + 19))       # softmax over the 20 legal moves only


def test_evaluate_reads_the_value_head_as_loss_draw_win(stub_maia):
    e = stub_maia.evaluate([chess.Board()], 1500, 1600)[0]
    z = np.exp([0.0, 1.0, 2.0])
    assert (e.loss, e.draw, e.win) == pytest.approx(tuple(z / z.sum()))
    assert e.win + e.draw + e.loss == pytest.approx(1.0)


def test_evaluate_turns_blacks_moves_back_to_the_real_board(stub_maia):
    b = chess.Board()
    b.push_san("d4")
    e = stub_maia.evaluate([b], 1500, 1600)[0]
    assert e.policy[0][0] == "e7e5"                            # the network said e2e4 in Black's own frame
    assert all(chess.Move.from_uci(u) in b.legal_moves for u, _ in e.policy)


def test_evaluate_skips_finished_games_and_batches_the_rest(stub_maia):
    mated = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        mated.push_san(san)
    boards = [chess.Board(), mated, chess.Board(), chess.Board()]
    out = stub_maia.evaluate(boards, [1000, 1100, 1200, 1300], [2000, 2100, 2200, 2300])
    assert [e is None for e in out] == [False, True, False, False]
    assert stub_maia.evals == 3
    assert [c["shape"] for c in stub_maia.calls] == [(2, 64, 12 * HISTORY + 1), (1, 64, 12 * HISTORY + 1)]
    assert [c["self"] for c in stub_maia.calls] == [[1000, 1200], [1300]]      # each board keeps its own ratings
    assert [c["oppo"] for c in stub_maia.calls] == [[2000, 2200], [2300]]
    assert all(c["think"] == 0.0 for c in stub_maia.calls)                     # the think-time channel stays empty


def test_evaluate_takes_one_rating_for_every_board(stub_maia):
    stub_maia.evaluate([chess.Board(), chess.Board()], 1500, 1700)
    assert stub_maia.calls[0]["self"] == [1500, 1500] and stub_maia.calls[0]["oppo"] == [1700, 1700]


# --- the real model -----------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("DEEP_VALUE_MODEL_TESTS") != "1", reason="set DEEP_VALUE_MODEL_TESTS=1 to load Maia-3")
def test_the_real_model_on_positions_with_an_obvious_answer():
    from deep_value import DeepValue

    maia = Maia(os.environ.get("DEEP_VALUE_TEST_MODEL", "maia3-5m"))
    start = chess.Board()
    queen_up = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    recapture = chess.Board()
    for san in ("e4", "d5", "exd5"):
        recapture.push_san(san)
    even, winning, taking = maia.evaluate([start, queen_up, recapture], 1500, 1500)
    assert sum(p for _, p in even.policy) == pytest.approx(1.0)
    assert {even.policy[0][0], even.policy[1][0]} <= {"e2e4", "d2d4", "g1f3", "c2c4"}
    assert 0.4 < even.expected < 0.6 and winning.expected > 0.75
    assert taking.policy[0][0] == "d8d5"                       # Black takes back, in the board's own coordinates
    one_by_one = maia.evaluate([queen_up], 1500, 1500)[0]      # a board's answer does not depend on its batch
    assert one_by_one.expected == pytest.approx(winning.expected, abs=1e-4)
    weak, strong = maia.evaluate([start, start], [1000, 2200], [1000, 2200])
    assert weak.policy != strong.policy                        # the ratings reach the model
    deep = DeepValue(maia, eps=0.05).value(queen_up, 1500, 1500)
    assert deep.value > 0.75 and sum(deep.wdl) == pytest.approx(1.0)
