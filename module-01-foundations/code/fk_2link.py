"""Lessons 1.11-1.12 - forward kinematics of the 2-link planar arm, two independent ways.

Run:  python fk_2link.py            (cross-check over 10k random poses)
      python fk_2link.py --workspace [--limits]   (draw the reachable set)
"""
import sys
from pathlib import Path

import numpy as np

L1, L2 = 1.0, 0.7
# Joint limits used with --limits (radians): a realistic asymmetric elbow.
LIMITS = ((-np.pi, np.pi), (-2.4, 2.4 * 0.1))


def fk_trig(t1, t2):
    """(x, y, phi) of the end-effector - from the three trig lines in the lesson."""
    # TODO(you)
    raise NotImplementedError


def fk_transforms(t1, t2):
    """Same result via the 4-factor transform product: Rz(t1)·Tx(L1)·Rz(t2)·Tx(L2).

    Import YOUR completed library (transforms_starter). Return (x, y, phi),
    with phi recovered from the final rotation block (hint: atan2 of R[1,0], R[0,0]).
    """
    # TODO(you)
    raise NotImplementedError


def crosscheck(n=10_000):
    rng = np.random.default_rng(7)
    for _ in range(n):
        t1, t2 = rng.uniform(-np.pi, np.pi, 2)
        a, b = np.array(fk_trig(t1, t2)), np.array(fk_transforms(t1, t2))
        # angles compare modulo 2π:
        da = (a[2] - b[2] + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(a[:2], b[:2], atol=1e-9) and abs(da) < 1e-9, (t1, t2, a, b)
    print(f"✅ two independent FK implementations agree over {n} random configurations.")
    print("   (This cross-check pattern is how kinematics gets tested for real.)")


def workspace(limits=False, n=250):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (lo1, hi1), (lo2, hi2) = LIMITS if limits else ((-np.pi, np.pi), (-np.pi, np.pi))
    t1s, t2s = np.linspace(lo1, hi1, n), np.linspace(lo2, hi2, n)
    pts = np.array([fk_trig(a, b)[:2] for a in t1s for b in t2s])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, color="#2a78d6", alpha=0.25, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    tag = "with joint limits" if limits else "no joint limits"
    ax.set_title(f"2-link workspace ({tag})")
    # Beside this script, not in the current directory: run from the repo root,
    # a bare filename would drop an untracked PNG there, and only
    # `module-*/code/*.png` is gitignored.
    out = Path(__file__).with_name(f"workspace_{'limits' if limits else 'full'}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    if "--workspace" in sys.argv:
        workspace(limits="--limits" in sys.argv)
    else:
        crosscheck()
