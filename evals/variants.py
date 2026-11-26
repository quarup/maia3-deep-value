"""The tree with every design option, kept for the experiments behind docs/design-notes.md.

The package (deep_value/tree.py) has one way to choose moves: follow a move when its path probability is at
least eps, and let the moves left out count with the node's value head. This module is the earlier, more
general tree that the comparisons were run with:

  default (no options)                 the likeliest moves up to `mass` of the policy, at most `max_children`,
                                       renormalised ("top 8")
  mass=1.0, max_children=None          every legal move
  renormalise=False                    the policy's own weights; moves left out count with the value head
  strict=True (with the three above)   "eps only", which is what the package now does
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import chess
import numpy as np

from deep_value.engine import Eval

EPS = 0.002          # follow a line while its probability from the root board state is at least this
MASS = 0.97          # share of the policy taken at a node, then renormalised
MAX_CHILDREN = 8
MAX_DEPTH = 40       # plies below the root
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

    def __init__(self, evaluate: Evaluate, *, eps: float = EPS, mass: float = MASS,
                 max_children: int | None = MAX_CHILDREN, max_depth: int = MAX_DEPTH,
                 renormalise: bool = True, strict: bool = False):
        self.evaluate = getattr(evaluate, "evaluate", evaluate)
        self.eps, self.mass, self.max_children, self.max_depth = eps, mass, max_children, max_depth
        self.renormalise, self.strict = renormalise, strict
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

    def _moves(self, e: Eval, q: float) -> tuple[list[tuple[str, float]], float]:
        """The moves followed from a node whose path probability is `q`, with their weights, and the weight
        left for the node's own value-head value. By default: the likeliest moves up to `mass` of the policy
        (at most `max_children`), renormalised, nothing left over. With `strict`, a move whose own path
        probability would be under `eps` is not followed either. With `renormalise` off, the weights stay
        the policy's and the moves left out count with the node's value-head value."""
        out, got = [], 0.0
        for uci, p in e.policy[: self.max_children]:
            if self.strict and q * p < self.eps:
                break
            out.append((uci, p))
            got += p
            if got >= self.mass:
                break
        if not out:
            return [], 1.0
        if self.renormalise:
            return [(u, p / got) for u, p in out], 0.0
        return out, max(0.0, 1.0 - got)

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
                if i and (n["q"] < self.eps or n["d"] >= self.max_depth):
                    continue                                   # a leaf: the value head gives its value
                moves, n["rest"] = self._moves(self.cache[n["k"]], n["q"])
                if not moves:
                    continue                                   # strict: no move is likely enough to follow
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
        White's) and "kind": "inner", "leaf" (valued by the value head)
        or "over" (mate or a draw)."""
        self.value(board, white_elo, black_elo)
        nodes = self._trees[0]
        parent = {k: i for i, n in enumerate(nodes) for k in n["kids"] or ()}
        return [{"parent": parent.get(i), "san": nodes[parent[i]]["b"].san(n["b"].peek()) if i else None,
                 "p": n["p"], "q": n["q"], "depth": n["d"], "wdl": n["wdl"], "head": self._white_wdl(n["k"], n["b"]),
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
            # under a looser setting (strict only) a move below that threshold is not followed either
            on = np.stack([nodes[k]["q"] >= eps for k in ks]) if self.strict else np.ones((len(ks), len(settings)), bool)
            down = sum(p[j] * on[j][:, None] * val[k] for j, k in enumerate(ks))
            rest = n["rest"] + ((~on) * p[:, None]).sum(axis=0)             # weight of the moves not followed
            tail = np.broadcast_to(own, (len(settings), 3))                # ... which count with this node's value head
            down = down + rest[:, None] * tail
            if i == 0:
                val[i] = down
                size += len(n["kids"])
            else:                                              # under a looser setting this node is a leaf
                followed = (n["q"] >= eps) & (n["d"] < cap)
                val[i] = np.where(followed[:, None], down, own[None, :])
                size += followed * len(n["kids"])
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
