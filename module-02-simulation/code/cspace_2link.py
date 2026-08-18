"""Lesson 2.13 - configuration space of the 2-link planar arm (starter).

The arm is Module 1's: links 1.00 and 0.70, base at the origin, both joints free
to turn all the way round. Obstacles are discs bolted to the bench. Every joint
pair is one point in a square of side 2*pi, and this file colours that square
free or forbidden, then counts what is connected to what.

Run:  python cspace_2link.py            occupancy, connectivity, writes cspace.png
      python cspace_2link.py --sweep    free fraction against grid resolution
      python cspace_2link.py --wrap     the square is a torus; the short way proves it
      python cspace_2link.py --split    one more disc, and a fifth of free space is lost
"""
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

L1, L2 = 1.0, 0.7
LINK_RADIUS = 0.04  # the arm is a capsule, not a mathematical line
TAU = 2.0 * np.pi

# (x, y, radius). One disc close enough for the upper arm to hit, three out at
# the rim that only the forearm can reach.
BENCH = ((0.15, -0.80, 0.22),
         (1.35, 0.45, 0.16),
         (0.10, 1.45, 0.18),
         (-1.30, 0.60, 0.17))

# The disc added by --split. It sits inside the upper arm's reach, which is the
# property that matters.
EXTRA_DISC = (0.95, 0.30, 0.18)

# The query used by the planner in the next lesson.
START = (-2.30, 1.05)
GOAL = (1.15, -1.60)


def elbow_and_tip(q):
    """Joint pair -> (elbow position, fingertip position). Module 1's forward kinematics."""
    t1, t2 = q
    elbow = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
    tip = elbow + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
    return elbow, tip


def segment_point_distance(a, b, p):
    """Shortest distance from point p to the segment a-b.

    Project p onto the infinite line through a and b, clamp the projection
    parameter to [0, 1] so it stays on the segment, and measure from there.
    """
    # TODO(you) 1
    raise NotImplementedError


def collides(q, obstacles=BENCH):
    """True if the arm at joint pair q overlaps any disc.

    Both links are capsules of radius LINK_RADIUS, so a link hits a disc of
    radius r once the distance from its centre line to the disc centre drops
    below r + LINK_RADIUS.
    """
    # TODO(you) 2: two segments, base-to-elbow and elbow-to-tip, against every disc.
    # Return True as soon as one distance drops below that disc's radius plus
    # LINK_RADIUS. Nothing else in this file needs to know how the arm is shaped.
    raise NotImplementedError


def occupancy(n=240, obstacles=BENCH):
    """Boolean grid over the whole configuration space. True means forbidden."""
    axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
    grid = np.zeros((n, n), dtype=bool)
    for i, t1 in enumerate(axis):
        for j, t2 in enumerate(axis):
            grid[i, j] = collides((t1, t2), obstacles)
    return axis, grid


def components(grid):
    """Label the free cells, wrapping at both edges because the space is a torus.

    Returns (labels, sizes sorted large to small).
    """
    n = grid.shape[0]
    free = ~grid
    labels = -np.ones((n, n), dtype=int)
    sizes = []
    nxt = 0
    for i in range(n):
        for j in range(n):
            if not free[i, j] or labels[i, j] >= 0:
                continue
            queue = deque([(i, j)])
            labels[i, j] = nxt
            count = 0
            while queue:
                a, b = queue.popleft()
                count += 1
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = (a + da) % n, (b + db) % n
                    if free[x, y] and labels[x, y] < 0:
                        labels[x, y] = nxt
                        queue.append((x, y))
            sizes.append(count)
            nxt += 1
    return labels, sorted(sizes, reverse=True)


def interpolate(q0, q1, steps):
    """Straight line in configuration space, taking the short way round each joint."""
    # TODO(you) 3: the naive q0 + (q1 - q0) * s is wrong here, because -3.0 and
    # +3.0 are 0.28 rad apart on a circle and 6.0 apart on the number line. Wrap
    # the difference into [-pi, pi) first, then walk it. The `--wrap` mode of this
    # file is the check: the two routes it prints must differ.
    raise NotImplementedError


def straight_line_survey(obstacles=BENCH, draws=2000, seed=0, n=240):
    """How often is the naive answer - drive straight there in joint space - wrong?"""
    axis, grid = occupancy(n, obstacles)
    labels, _ = components(grid)
    biggest = np.bincount(labels[labels >= 0].ravel()).argmax()
    rng = np.random.default_rng(seed)
    blocked = tried = 0
    while tried < draws:
        i0, j0, i1, j1 = rng.integers(0, n, 4)
        if labels[i0, j0] != biggest or labels[i1, j1] != biggest:
            continue
        tried += 1
        q0 = np.array([axis[i0], axis[j0]])
        q1 = np.array([axis[i1], axis[j1]])
        if any(collides(q, obstacles) for q in interpolate(q0, q1, 120)):
            blocked += 1
    return blocked, tried


def sweep_resolution():
    print("free fraction of configuration space, by grid resolution")
    print("  cells/axis      cells      free   seconds")
    for n in (60, 120, 240, 480):
        t0 = time.perf_counter()
        _, grid = occupancy(n)
        dt = time.perf_counter() - t0
        print(f"  {n:>10}  {n * n:>9,}   {1 - grid.mean():>7.4f}  {dt:>8.2f}")


def wrap_demo():
    """The square is a torus: its left edge is glued to its right edge."""
    q0 = np.array([2.90, 0.40])
    q1 = np.array([-2.90, 0.40])
    across = abs(q1[0] - q0[0])
    edge = abs((q1[0] - q0[0] + np.pi) % TAU - np.pi)
    print(f"shoulder from {q0[0]:+.2f} rad to {q1[0]:+.2f} rad, elbow held at {q0[1]:+.2f}")
    print(f"  straight across the picture: {across:.3f} rad of shoulder travel")
    print(f"  off one edge and back on:    {edge:.3f} rad")
    routes = (("across the picture", [q0 + (q1 - q0) * s for s in np.linspace(0, 1, 200)]),
              ("off the edge", interpolate(q0, q1, 200)))
    for label, route in routes:
        blocked = sum(collides(q) for q in route)
        print(f"  {label:>20}: {'free' if blocked == 0 else str(blocked) + '/200 blocked'}")


def split_demo(n=360):
    print(f"connectivity of free space, flood-filled on a {n} x {n} grid")
    for label, obstacles in (("bench", BENCH), ("bench + one more disc", BENCH + (EXTRA_DISC,))):
        _, grid = occupancy(n, obstacles)
        _, sizes = components(grid)
        total = sum(sizes)
        share = 100.0 * sizes[0] / total
        print(f"  {label:>22}: free {1 - grid.mean():.4f}   "
              f"components {len(sizes)}   largest holds {share:.1f}% of free space")


def plot(n=360):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis, grid = occupancy(n)
    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(11, 5.2))

    for ox, oy, orad in BENCH:
        ax_w.add_patch(plt.Circle((ox, oy), orad, color="#9c3b2e", alpha=0.35, lw=0))
    for q, colour in ((START, "#2d5d7c"), (GOAL, "#15614e")):
        elbow, tip = elbow_and_tip(q)
        ax_w.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], "-o", color=colour, lw=2.5, ms=4)
    ax_w.set_aspect("equal")
    ax_w.set_xlim(-1.9, 1.9)
    ax_w.set_ylim(-1.9, 1.9)
    ax_w.set_title("workspace: four discs, start and goal poses")

    ax_c.imshow(grid.T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                cmap="Reds", alpha=0.8, interpolation="nearest")
    line = np.array(interpolate(START, GOAL, 200))
    line = (line + np.pi) % TAU - np.pi  # the torus has no outside, so fold it back in
    cut = np.nonzero(np.abs(np.diff(line[:, 0])) > np.pi)[0] + 1
    for piece in np.split(line, cut):
        ax_c.plot(piece[:, 0], piece[:, 1], "--", color="#52514e", lw=1.4)
    ax_c.plot(*START, "o", color="#2d5d7c", ms=8)
    ax_c.plot(*GOAL, "o", color="#15614e", ms=8)
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-np.pi, np.pi)
    ax_c.set_ylim(-np.pi, np.pi)
    ax_c.set_xlabel("shoulder (rad)")
    ax_c.set_ylabel("elbow (rad)")
    ax_c.set_title("configuration space: shaded is forbidden")

    fig.tight_layout()
    # Next to this script, not the current directory: .gitignore covers
    # module-*/code/*.png, so running this from the repo root must not
    # drop an untracked file there.
    out = Path(__file__).with_name("cspace.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out.name}")


def main():
    if "--sweep" in sys.argv:
        return sweep_resolution()
    if "--wrap" in sys.argv:
        return wrap_demo()
    if "--split" in sys.argv:
        return split_demo()

    n = 360
    t0 = time.perf_counter()
    _, grid = occupancy(n)
    dt = time.perf_counter() - t0
    _, sizes = components(grid)
    print(f"grid {n} x {n} = {n * n:,} configurations in {dt:.1f} s")
    print(f"forbidden {grid.mean():.4f}   free {1 - grid.mean():.4f}   "
          f"connected components {len(sizes)}")

    line = interpolate(START, GOAL, 200)
    blocked = sum(collides(q) for q in line)
    print(f"straight line {START} -> {GOAL}: "
          f"{'free' if blocked == 0 else str(blocked) + '/200 samples blocked'}")

    hit, tried = straight_line_survey()
    print(f"over {tried} random reachable pairs, the straight line is blocked "
          f"{hit} times ({100 * hit / tried:.0f}%)")
    plot(n)


if __name__ == "__main__":
    main()
