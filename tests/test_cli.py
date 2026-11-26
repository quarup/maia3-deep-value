"""The command line, with a hand-made evaluator in place of Maia-3."""
import io
import json

import chess
import pytest

from conftest import make_fake, noisy_wdl, uniform_policy
from deep_value import DeepValue, cli

PGN = """[Site "https://lichess.org/abcd1234"]
[WhiteElo "1510"]
[BlackElo "1490"]

1. a3 e5 2. a4 Nf6 *
"""


class FakeMaia:
    made = []

    def __init__(self, model, threads=None):
        self.model, self.boards, self.elos = model, 0, set()
        self._evaluate = make_fake(lambda b: uniform_policy(b, 2), noisy_wdl)
        FakeMaia.made.append(self)

    def evaluate(self, boards, self_elos, oppo_elos):
        self.boards += len(boards)
        self.elos |= {(b.turn, s, o) for b, s, o in zip(boards, self_elos, oppo_elos)}
        return self._evaluate(boards, self_elos, oppo_elos)


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "Maia", FakeMaia)
    FakeMaia.made.clear()
    pgn = tmp_path / "game.pgn"
    pgn.write_text(PGN)

    def go(*args):
        cli.main([a.replace("GAME", str(pgn)) for a in args])
        return capsys.readouterr().out.splitlines()
    go.tmp = tmp_path
    return go


def test_a_game_gives_one_line_per_board_state(run):
    lines = run("GAME", "--eps", "0.2")
    assert len(lines) == 5                                     # before each of the 4 moves, and the final board state
    assert [ln.split()[2] for ln in lines] == ["1.a3", "1...e5", "2.a4", "2...Nf6", "end"]
    assert all("deep" in ln and "one-hop" in ln and "zero-hop" in ln for ln in lines)
    assert FakeMaia.made[0].model == "maia3-79m"
    # the PGN's ratings, each side's own first
    assert FakeMaia.made[0].elos == {(chess.WHITE, 1510, 1490), (chess.BLACK, 1490, 1510)}


def test_the_json_matches_the_library_and_names_the_move_played(run):
    out = run.tmp / "out.json"
    run("GAME", "--eps", "0.2", "-o", str(out))
    doc = json.loads(out.read_text())
    assert doc["game"] == "https://lichess.org/abcd1234" and doc["ratings"] == [1510, 1490] and doc["eps"] == 0.2
    assert list(doc["plies"]) == ["0", "1", "2", "3", "4"]
    deep = DeepValue(make_fake(lambda b: uniform_policy(b, 2), noisy_wdl), eps=0.2)
    b = chess.Board()
    assert doc["plies"]["0"]["wdl"] == pytest.approx(list(deep.value(b, 1510, 1490).wdl), abs=1e-4)
    # 1.a3 is one of the two moves the policy follows, so its value comes from the tree
    first = doc["plies"]["0"]
    assert first["played"] == {"san": "a3", "value": next(m["value"] for m in first["moves"] if m["san"] == "a3")}
    # 1...e5 is not: it is valued with a tree of its own
    b.push_san("a3")
    second = doc["plies"]["1"]
    assert "e5" not in [m["san"] for m in second["moves"]]
    assert second["played"]["value"] == pytest.approx(deep.after(b, "e5", 1510, 1490).value, abs=1e-4)
    assert "played" not in doc["plies"]["4"]                   # the last board state has no move after it


def test_a_stopped_run_resumes_from_the_file(run):
    out = run.tmp / "out.json"
    assert len(run("GAME", "--eps", "0.2", "--plies", "0,1", "-o", str(out))) == 2
    lines = run("GAME", "--eps", "0.2", "-o", str(out))
    assert [ln.split()[1] for ln in lines] == ["2", "3", "4"]  # plies 0 and 1 are not computed again
    assert list(json.loads(out.read_text())["plies"]) == ["0", "1", "2", "3", "4"]
    assert run("GAME", "--eps", "0.2", "-o", str(out)) == [] and FakeMaia.made[-1].boards == 0


def test_another_setting_starts_the_file_again(run):
    out = run.tmp / "out.json"
    run("GAME", "--eps", "0.2", "--plies", "0", "-o", str(out))
    run("GAME", "--eps", "0.5", "--plies", "3", "-o", str(out))
    doc = json.loads(out.read_text())
    assert doc["eps"] == 0.5 and list(doc["plies"]) == ["3"]   # values at two thresholds are never mixed


def test_ratings_can_be_overridden(run):
    run("GAME", "--eps", "0.5", "--plies", "0,1", "--ratings", "2000")
    assert FakeMaia.made[-1].elos == {(chess.WHITE, 2000, 2000), (chess.BLACK, 2000, 2000)}
    run("GAME", "--eps", "0.5", "--plies", "0,1", "--ratings", "1200,1800")
    assert FakeMaia.made[-1].elos == {(chess.WHITE, 1200, 1800), (chess.BLACK, 1800, 1200)}


def test_a_single_position_from_a_fen(run):
    lines = run("--fen", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "--ratings", "1500", "--eps", "0.2", "--model", "maia3-5m")
    assert len(lines) == 1 and lines[0].split()[2] == "end"
    assert FakeMaia.made[-1].model == "maia3-5m"


def test_the_game_can_come_from_stdin(run, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(PGN))
    assert len(run("-", "--eps", "0.5", "--plies", "2")) == 1


@pytest.mark.parametrize("args", [[], ["GAME", "--fen", chess.STARTING_FEN], ["--fen", chess.STARTING_FEN]])
def test_bad_arguments_are_refused(run, args):
    with pytest.raises(SystemExit):                            # nothing to read, two things to read, a FEN without ratings
        run(*args)
    assert FakeMaia.made == []                                 # refused before the model is loaded
