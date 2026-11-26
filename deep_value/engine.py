"""Maia-3 as an evaluator: for each board, the move probabilities and win/draw/loss of the side to move.

This is the only file that touches the model. It talks to the `maia3` package directly:

  * each board is read with its real history (the last 8 positions of `board.move_stack`), which is what
    Maia-3 was trained on and what its UCI engine does with `--use-uci-history`;
  * every board in a batch carries its own pair of ratings, so positions from different games (or the two
    sides of one game) go through the model together.

Anything with the same `evaluate(boards, self_elos, oppo_elos)` signature can stand in for it: the tree in
`tree.py` does not know about Maia-3, and the tests pass a hand-made evaluator.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Sequence

import chess
import numpy as np

HISTORY = 8


@dataclass
class Eval:
    """One position as the model reads it, from the point of view of the side to move."""
    policy: list[tuple[str, float]]     # (uci, probability) over the legal moves, likeliest first
    win: float
    draw: float
    loss: float

    @property
    def expected(self) -> float:
        return self.win + 0.5 * self.draw


def _piece_planes(board: chess.Board) -> np.ndarray:
    """(64, 12) one-hot planes of one board, seen from its side to move (as maia3.dataset.tokenize_board)."""
    if board.turn == chess.BLACK:
        board = board.mirror()
    planes = np.zeros((64, 12), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        planes[sq, piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)] = 1.0
    return planes


def history_planes(board: chess.Board, n: int = HISTORY) -> np.ndarray:
    """(64, 12 * n) planes of the last `n` positions leading to `board`, oldest first, `board` last. A game
    shorter than `n` is padded at the front with its earliest position."""
    b = board.copy()
    planes = [_piece_planes(b)]
    while len(planes) < n and b.move_stack:
        b.pop()
        planes.append(_piece_planes(b))
    planes += [planes[-1]] * (n - len(planes))
    return np.concatenate(planes[::-1], axis=1)


def _mirror(uci: str) -> str:
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


class Maia:
    """A Maia-3 checkpoint (`maia3-5m`, `maia3-23m`, `maia3-79m`), downloaded from Hugging Face on first use."""

    def __init__(self, model: str = "maia3-79m", device: str = "cpu", batch: int = 128, threads: int | None = None):
        import torch
        from maia3 import model_registry, uci, utils

        if threads:
            torch.set_num_threads(threads)
        cfg = types.SimpleNamespace()
        spec = model_registry.resolve_model_spec(model)
        model_registry.apply_model_config(cfg, spec)
        cfg.device = device
        cfg.trust_checkpoint = False
        try:
            cfg.checkpoint_path = model_registry.resolve_checkpoint_path(spec, local_files_only=True)
        except Exception:
            cfg.checkpoint_path = model_registry.resolve_checkpoint_path(spec)
        if cfg.include_time_info or cfg.history != HISTORY:
            raise ValueError(f"{model}: this wrapper expects 8 positions of history and no clock inputs")
        self.name = spec.name
        self.torch = torch
        self.model = uci.load_model(cfg)
        self.device = device
        self.batch = batch
        self.index = {u: i for i, u in enumerate(utils.get_all_possible_moves())}
        self.evals = 0                  # positions read so far

    def evaluate(self, boards: Sequence[chess.Board], self_elos, oppo_elos) -> list[Eval | None]:
        """One Eval per board; None where the game is over. `self_elos` and `oppo_elos` are the ratings of
        the side to move and of its opponent: one number for all boards, or one per board."""
        n = len(boards)
        se = [int(self_elos)] * n if np.isscalar(self_elos) else [int(x) for x in self_elos]
        oe = [int(oppo_elos)] * n if np.isscalar(oppo_elos) else [int(x) for x in oppo_elos]
        out: list[Eval | None] = [None] * n
        live = [i for i, b in enumerate(boards) if not b.is_game_over()]
        torch = self.torch
        for s in range(0, len(live), self.batch):
            idx = live[s:s + self.batch]
            x = np.zeros((len(idx), 64, 12 * HISTORY + 1), dtype=np.float32)     # the last channel is think time: 0
            legal = []
            for row, i in enumerate(idx):
                b = boards[i]
                x[row, :, :-1] = history_planes(b)
                ucis = [m.uci() for m in b.legal_moves]
                frame = ucis if b.turn == chess.WHITE else [_mirror(u) for u in ucis]
                legal.append((ucis, [self.index[u] for u in frame]))
            with torch.no_grad():
                logits, value, _ = self.model(
                    torch.from_numpy(x).to(self.device),
                    torch.tensor([se[i] for i in idx], dtype=torch.long, device=self.device),
                    torch.tensor([oe[i] for i in idx], dtype=torch.long, device=self.device))
                wdl = torch.softmax(value.float(), dim=-1).cpu().numpy()          # (loss, draw, win)
                logits = logits.float().cpu().numpy()
            for row, i in enumerate(idx):
                ucis, cols = legal[row]
                z = logits[row, cols].astype(np.float64)
                p = np.exp(z - z.max())
                p /= p.sum()
                order = np.argsort(-p)
                out[i] = Eval([(ucis[k], float(p[k])) for k in order],
                              float(wdl[row, 2]), float(wdl[row, 1]), float(wdl[row, 0]))
            self.evals += len(idx)
        return out


def default_threads() -> int | None:
    v = os.environ.get("DEEP_VALUE_THREADS")
    return int(v) if v else None
