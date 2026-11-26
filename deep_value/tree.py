"""Deep value: a board state's game outcome prediction (win, draw, loss), computed over a tree of likely play.

Maia-3 has two output heads: a policy head that predicts the move a player of a given rating plays, and a
value head that predicts the game outcome (win, draw, loss). The value head evaluates one board state in one
model evaluation ("zero-hop"). It can be off when something is in the air: a piece hanging, a mate nobody has
seen yet. Averaging the value head over the board states one move later, weighted by the policy ("one-hop"),
only moves the problem a move down. Here the value of a board state is computed by following the policy
head for both sides, move after move, with one setting, `eps`:

  * a move is followed when the probability of the whole path to it, from the root board state, is at least
    `eps`. Pruning is by the probability of the path, not per move, so the tree has at most 1/eps board
    states at any depth: forced lines and endgames (one or two moves take the whole policy) run deep, and
    quiet positions (the policy is spread wide) stay shallow;
  * the moves a board state leaves out keep their policy weight and count with that board state's own
    value head. A board state none of whose moves is followed is a leaf: the value head gives its value;
  * checkmate and draws get their exact outcome;
  * values are then averaged back up the tree, each move weighted by its policy probability. Nobody
    maximises: this is the value of the model's own policy (policy evaluation, not search for the best
    move), "what happens if both sides keep playing like players of these ratings".

Every board state in the tree has path probability at least `eps`, and each is one model evaluation.

Because pruning is by path probability, the tree for a larger `eps` is the top of the tree for a smaller
one. One traversal therefore answers every looser threshold too (`also=`), along with zero-hop and one-hop.

Board states are evaluated with their full move history (the model's input), so the board states of one
game share most of their model evaluations through the cache.

Why this rule and not others (a fixed number of moves per board state, renormalised weights, a discount on
the moves left out): docs/design-notes.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import chess
import numpy as np

from .engine import Eval

EPS = 0.002          # follow a move when the probability of the path to it from the root is at least this
MAX_DEPTH = 40       # plies below the root: a safety stop that the default eps does not reach in practice
ROOT_MOVES = 6       # how many of the root's own moves are reported

Evaluate = Callable[[Sequence[chess.Board], Sequence[int], Sequence[int]], list]


def terminal_wdl(board: chess.Board) -> tuple[float, float, float] | None:
    """White's (win, draw, loss) if the game is over at `board`, else None."""
    if board.is_checkmate():
        return (0.0, 0.0, 1.0) if board.turn == chess.WHITE else (1.0, 0.0, 0.0)
    if board.is_game_over():
        return (0.0, 1.0, 0.0)
    return None


def scalar(wdl) -> float:
    """A (win, draw, loss) value as one number: P(win) + P(draw) / 2."""
    return float(wdl[0] + 0.5 * wdl[1])


@dataclass
class Estimate:
    """The value of one board state. Every triple is (win, draw, loss) for White."""
    wdl: tuple[float, float, float]            # deep value
    zero_hop: tuple[float, float, float]       # the value head on the board state itself
    one_hop: tuple[float, float, float]        # the value head averaged over the board states one move later
                                               # (moves with policy under eps count with the root's value head)
    nodes: int = 1
    leaves: int = 1
    depth: int = 0                             # deepest ply reached below the root
    moves: list[dict] = field(default_factory=list)      # the likeliest moves: {"san", "p", "wdl"}
    also: dict = field(default_factory=dict)   # {(eps, max_depth): wdl} for the looser settings asked for
    also_nodes: dict = field(default_factory=dict)       # {(eps, max_depth): nodes that setting's tree has}

    @property
    def value(self) -> float:
        """The deep value as one number for White: P(win) + P(draw) / 2."""
        return scalar(self.wdl)

    def to_json(self) -> dict:
        r4 = lambda t: [round(float(x), 4) for x in t]
        return {"value": round(self.value, 4), "wdl": r4(self.wdl), "zero_hop": r4(self.zero_hop),
                "one_hop": r4(self.one_hop), "nodes": self.nodes, "leaves": self.leaves, "depth": self.depth,
                "moves": [{**m, "wdl": r4(m["wdl"]), "value": round(scalar(m["wdl"]), 4)} for m in self.moves],
                **({"also": {f"{e:g}/{d}": r4(v) for (e, d), v in self.also.items()}} if self.also else {})}


class DeepValue:
    """Computes deep values with one evaluator (see `engine.Maia`), keeping the model evaluations it has made.

        deep = DeepValue(Maia("maia3-79m"))
        v = deep.value(board, white_elo=1500, black_elo=1480)     # board carries its move stack
        v.wdl, v.value, v.zero_hop, v.one_hop, v.moves
    """

    def __init__(self, evaluate: Evaluate, *, eps: float = EPS, max_depth: int = MAX_DEPTH):
        self.evaluate = getattr(evaluate, "evaluate", evaluate)
        self.eps, self.max_depth = eps, max_depth
        self.cache: dict[tuple, Eval | None] = {}
        self._trees: list = []
        self.evals = 0

    # --- reading positions ---------------------------------------------------------------------------

    @staticmethod
    def key(board: chess.Board) -> str:
        """A position with its history, as text: the start position and every move since."""
        return board.root().fen() + "|" + " ".join(m.uci() for m in board.move_stack)

    def _fill(self, want: Iterable[tuple[tuple, chess.Board]]) -> None:
        """Evaluate, in one batch, the board states not evaluated yet. `want` yields ((key, white, black), board)."""
        todo: dict[tuple, chess.Board] = {}
        for k, b in want:
            if k not in self.cache and k not in todo:
                todo[k] = b
        if not todo:
            return
        ks = list(todo)
        boards = [todo[k] for k in ks]
        mover = [k[1] if b.turn == chess.WHITE else k[2] for k, b in zip(ks, boards)]
        other = [k[2] if b.turn == chess.WHITE else k[1] for k, b in zip(ks, boards)]
        self.cache.update(zip(ks, self.evaluate(boards, mover, other)))
        self.evals += len(ks)

    def _white_wdl(self, k: tuple, board: chess.Board) -> tuple[float, float, float]:
        t = terminal_wdl(board)
        if t is not None:
            return t
        e = self.cache[k]
        return (e.win, e.draw, e.loss) if board.turn == chess.WHITE else (e.loss, e.draw, e.win)

    # --- scoring -------------------------------------------------------------------------------------

    def value(self, board: chess.Board, white_elo: int, black_elo: int, *,
              also: Sequence[tuple[float, int]] = ()) -> Estimate:
        """The deep value of `board` (with its move stack) for players of these ratings.

        `also` lists looser settings, as (eps, max_depth) pairs with eps >= self.eps and max_depth <=
        self.max_depth, to be answered from the same tree at no extra cost."""
        return self.value_many([(board, white_elo, black_elo)], also=also)[0]

    def value_many(self, positions: Sequence[tuple[chess.Board, int, int]], *,
                   also: Sequence[tuple[float, int]] = ()) -> list[Estimate]:
        """`value` for several (board, white_elo, black_elo) at once. The trees grow level by level
        together, so each level is one batch through the model whatever the number of board states."""
        for e, d in also:
            if e < self.eps or d > self.max_depth:
                raise ValueError(f"also=({e}, {d}) is not inside the tree grown with eps={self.eps}, "
                                 f"max_depth={self.max_depth}")
        trees = []
        for board, we, be in positions:
            k = (self.key(board), int(we), int(be))
            # a node: board, key, q (path probability), d (depth), p (probability of its move), kids
            trees.append([{"b": board, "k": k, "q": 1.0, "d": 0, "p": 1.0, "kids": None}])
        level = [(t, 0) for t in range(len(trees))]
        while level:                                           # breadth first: one batch per level
            self._fill((trees[t][i]["k"], trees[t][i]["b"]) for t, i in level)
            nxt = []
            for t, i in level:
                nodes = trees[t]
                n = nodes[i]
                if terminal_wdl(n["b"]) is not None:
                    continue
                if n["d"] >= self.max_depth:
                    continue
                moves = [(u, p) for u, p in self.cache[n["k"]].policy if n["q"] * p >= self.eps]
                if not moves:
                    continue                                   # a leaf: no move is likely enough to follow
                n["kids"] = []
                for uci, p in moves:
                    nb = n["b"].copy()
                    nb.push(chess.Move.from_uci(uci))
                    sep = " " if n["b"].move_stack else ""
                    nodes.append({"b": nb, "k": (n["k"][0] + sep + uci, n["k"][1], n["k"][2]),
                                  "q": n["q"] * p, "d": n["d"] + 1, "p": p, "kids": None})
                    n["kids"].append(len(nodes) - 1)
                    nxt.append((t, len(nodes) - 1))
            level = nxt
        self._trees = trees
        return [self._back_up(nodes, list(also)) for nodes in trees]

    def explain(self, board: chess.Board, white_elo: int, black_elo: int) -> list[dict]:
        """The whole tree behind `value`, for drawing or debugging: one dict per node, the root
        first, with "parent" (index), "san", "p" (probability of the move), "q" (probability of the whole
        path), "depth", "wdl" (White's deep value, backed up), "head" (the value head on this board state alone,
        White's), "rest" (the policy weight of the moves it leaves out) and "kind": "inner", "leaf" (valued by
        the value head) or "over" (mate or a draw)."""
        self.value(board, white_elo, black_elo)
        nodes = self._trees[0]
        parent = {k: i for i, n in enumerate(nodes) for k in n["kids"] or ()}
        return [{"parent": parent.get(i), "san": nodes[parent[i]]["b"].san(n["b"].peek()) if i else None,
                 "p": n["p"], "q": n["q"], "depth": n["d"], "wdl": n["wdl"], "head": self._white_wdl(n["k"], n["b"]),
                 "rest": max(0.0, 1.0 - sum(nodes[k]["p"] for k in n["kids"])) if n["kids"] else 0.0,
                 "kind": "inner" if n["kids"] else "over" if terminal_wdl(n["b"]) is not None else "leaf"}
                for i, n in enumerate(nodes)]

    def _back_up(self, nodes: list[dict], also: list[tuple[float, int]]) -> Estimate:
        """Average the leaf values back to the root, once per setting: the tree's own, one-hop, and `also`."""
        root = nodes[0]
        zero = self._white_wdl(root["k"], root["b"])
        if root["kids"] is None:                               # the game is over here
            root["wdl"] = zero
            return Estimate(zero, zero, zero, also={s: zero for s in also}, also_nodes={s: 1 for s in also})
        settings = [(self.eps, self.max_depth), (0.0, 1)] + also      # the second is one-hop
        eps = np.array([s[0] for s in settings])
        cap = np.array([s[1] for s in settings])
        val = np.zeros((len(nodes), len(settings), 3))
        size = np.ones(len(settings), dtype=int)               # nodes in each setting's tree
        leaves = 0
        for i in range(len(nodes) - 1, -1, -1):                # children always come after their parent
            n = nodes[i]
            own = np.array(self._white_wdl(n["k"], n["b"]))
            if n["kids"] is None:
                val[i] = own
                leaves += 1
                continue
            ks = n["kids"]
            p = np.array([nodes[k]["p"] for k in ks])
            # per setting: the moves followed (a looser eps follows fewer) and the weight of the moves left out,
            # which count with this board state's own value head
            on = np.stack([nodes[k]["q"] >= eps for k in ks]) & (n["d"] < cap)
            down = sum(p[j] * on[j][:, None] * val[k] for j, k in enumerate(ks))
            rest = 1.0 - (on * p[:, None]).sum(axis=0)
            val[i] = down + rest[:, None] * own[None, :]
            size += on.sum(axis=0)
        for i, n in enumerate(nodes):
            n["wdl"] = tuple(val[i][0])
        moves = [{"san": root["b"].san(nodes[k]["b"].peek()), "p": round(nodes[k]["p"], 4), "wdl": tuple(val[k][0])}
                 for k in root["kids"][:ROOT_MOVES]]
        return Estimate(tuple(val[0][0]), zero, tuple(val[0][1]), nodes=len(nodes), leaves=leaves,
                     depth=max(n["d"] for n in nodes), moves=moves,
                     also={s: tuple(val[0][2 + j]) for j, s in enumerate(also)},
                     also_nodes={s: int(size[2 + j]) for j, s in enumerate(also)})

    def after(self, board: chess.Board, san: str, white_elo: int, black_elo: int) -> Estimate:
        """The deep value of the board state after `san`, whatever the policy thinks of that move."""
        nb = board.copy()
        nb.push_san(san)
        return self.value(nb, white_elo, black_elo)
