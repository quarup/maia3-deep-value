"""Hand-made evaluators shared by the tests: no model needed."""
import zlib


from deep_value import Eval


def make_fake(policy_of, wdl_of):
    """An evaluator from two functions of the board: [(uci, p)], and the mover's value. The value is one
    number (win + loss only) or a (win, draw, loss) triple."""
    def evaluate(boards, self_elos, oppo_elos):
        out = []
        for b in boards:
            if b.is_game_over():
                out.append(None)
                continue
            v = wdl_of(b)
            w, d, l = v if isinstance(v, tuple) else (v, 0.0, 1.0 - v)
            out.append(Eval(policy_of(b), w, d, l))
        return out
    return evaluate


def _noise(text):
    return zlib.crc32(text.encode()) / 0xFFFFFFFF            # the same in every run, unlike hash()


def uniform_policy(b, n=2):
    ms = sorted(m.uci() for m in b.legal_moves)[:n]
    return [(u, 1.0 / len(ms)) for u in ms]


def peaked_policy(b):
    """Every legal move, with weights that differ a lot by position and move: likeliest first, sums to 1."""
    w = {m.uci(): _noise(b.fen() + m.uci()) ** 6 + 1e-6 for m in b.legal_moves}
    total = sum(w.values())
    return sorted(((u, x / total) for u, x in w.items()), key=lambda t: -t[1])


def noisy_wdl(b):
    """A (win, draw, loss) for the mover that differs by position and has a real draw share."""
    w, d = 0.2 + 0.5 * _noise("w" + b.fen()), 0.25 * _noise("d" + b.fen())
    return (w, d, 1.0 - w - d)

