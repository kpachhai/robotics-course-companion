"""Solution - Lesson 3.14: a reward function is a specification.

Self-contained. Run:  python reward_hacking.py

Measured on a 2018 4-core laptop CPU, whole run about fifteen seconds.
"""
import numpy as np

DT, HORIZON, DRAG, WALL, X0 = 0.05, 120, 0.2, 2.0, 1.0


def rollout(k1, k2):
    """Run one episode of a = clip(k1*x + k2*v). Returns the position trace."""
    x, v = X0, 0.0
    xs = np.empty(HORIZON)
    for t in range(HORIZON):
        a = float(np.clip(k1 * x + k2 * v, -1.0, 1.0))
        v += (a - DRAG * v) * DT
        x += v * DT
        if abs(x) > WALL:                    # soft bumper at the arena edge
            x, v = float(np.clip(x, -WALL, WALL)), 0.0
        xs[t] = x
    return xs


# ------------------------------------------------------------ the two scores
def reward_true(xs):
    """Be near the origin, at every step."""
    return float(np.sum(-np.abs(xs)))


def reward_progress(xs):
    """Pay for closing the gap; never charge for opening it.

    The `maximum(0, ...)` is the whole bug. Without it the sum telescopes to
    |x_0| - |x_T| and cannot be farmed. With it, every inward swing is income
    and every outward swing is free.
    """
    prev = np.concatenate([[X0], xs[:-1]])
    return float(np.sum(np.maximum(0.0, np.abs(prev) - np.abs(xs))))


# ----------------------------------------------------------------- the search
GRID = np.linspace(-4.0, 4.0, 81)


def search():
    """Score every controller in the grid under both rewards."""
    best = {"true": (-np.inf, None, None), "progress": (-np.inf, None, None)}
    for k1 in GRID:
        for k2 in GRID:
            xs = rollout(k1, k2)
            for name, score in (("true", reward_true(xs)),
                                ("progress", reward_progress(xs))):
                if score > best[name][0]:
                    best[name] = (score, (float(k1), float(k2)), xs)
    return best


def main():
    best = search()
    print(f"{'optimised for':<16}{'its own score':>14}{'true score':>12}"
          f"{'progress score':>16}{'mean |x|':>10}{'final |x|':>11}")
    for name in ("true", "progress"):
        score, (k1, k2), xs = best[name]
        print(f"{name:<16}{score:>14.3f}{reward_true(xs):>12.2f}"
              f"{reward_progress(xs):>16.3f}{np.mean(np.abs(xs)):>10.3f}"
              f"{abs(xs[-1]):>11.3f}   gains ({k1:+.1f}, {k2:+.1f})")

    farmed = best["progress"][0]
    print(f"\nThe target is {X0:.1f} units away, so honest progress tops out at {X0:.1f}.")
    print(f"The winner under the progress reward banked {farmed:.2f}.")

    # Behavioural assertions: this file is a claim in the lesson, so it checks
    # itself rather than trusting the printout.
    assert farmed > 2.5 * X0, "the progress reward should be farmable well past the real distance"
    assert abs(best["progress"][2][-1]) > 1.0, "the hacked policy should end far from the target"
    assert abs(best["true"][2][-1]) < 0.05, "the honest reward should land on the target"


if __name__ == "__main__":
    main()
