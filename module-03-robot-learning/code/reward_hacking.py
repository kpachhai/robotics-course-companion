"""Lesson 3.14 - a reward function is a specification, and the optimiser reads it
like an adversary.

One point on a line wants to reach the origin. A tiny linear controller drives
it. We search the whole controller space twice: once scored by a reward that
says what we mean, once by a reward that says something we did not mean.

The environment and the search are done. The two reward functions are yours.

Run:  python reward_hacking.py
About fifteen seconds on a laptop CPU.
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
    """What we actually want: be near the origin, at every step.

    TODO(you): return the sum over the episode of -|x_t|.
    One line with numpy. This is the reward a careful engineer writes.
    """
    raise NotImplementedError


def reward_progress(xs):
    """What a reasonable engineer writes instead, to "help" a sparse task:
    pay for every step that closes the gap, and never charge for one that opens it.

    TODO(you): let prev be the position one step earlier (x_0 = X0 before step 0).
    Return the sum over the episode of max(0, |prev| - |x_t|).

    Write it exactly as described. Do not add a penalty. The point of the
    exercise is to watch this specification get exploited.
    """
    raise NotImplementedError


# ----------------------------------------------------------------- the search
GRID = np.linspace(-4.0, 4.0, 81)


def search():
    """Score every controller in the grid under both rewards. Returns the winner
    under each, as (score, gains, position trace)."""
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

    total_available = X0            # the whole distance there is to close
    farmed = best["progress"][0]
    print(f"\nThe target is {total_available:.1f} units away, so honest progress "
          f"tops out at {total_available:.1f}.")
    print(f"The winner under the progress reward banked {farmed:.2f}.")


if __name__ == "__main__":
    main()
