"""Lesson 4.8 - score a demonstration on consistency, not on success.

Four numbers plus a duration, all of them relative. There is no universal
threshold for "smooth enough", so this script never invents one: it ranks YOUR
episodes against each other and against your own median. An episode a long way
off your median is the one to watch back, and probably to re-record.

    python demo_smoothness.py             # synthetic smooth vs hesitant demo
    python demo_smoothness.py --seed 3    # different noise draw

NOT TESTED AGAINST A PHYSICAL ARM. The synthetic trajectories below are
hand-built to look like a rehearsed reach and a hesitant one; they are a way to
see what the metrics do, not evidence about any real robot. To score your own
recordings, fill in load_episode() for whatever LeRobot version you installed and
feed the result to score(). Check the import path against your own install
rather than trusting a snippet - the dataset API is one of the fastest-moving
surfaces in the library.
"""

import argparse

import numpy as np

FPS = 30.0
N_JOINTS = 6


def load_episode(repo_id, episode_index):
    """Return joint positions for one recorded episode as (T, n_joints) in degrees.

    This is the only hardware-facing function in the file, and it is the one
    thing here that cannot be written blind, because the dataset class and its
    column names move between LeRobot releases.

    Sketch of what you are looking for:
      - open the dataset for `repo_id` (local cache or Hub),
      - select the rows whose episode index equals `episode_index`,
      - pull the observation-state column (the follower's measured joint angles),
      - stack it into a (T, n_joints) float array, time along axis 0.

    Print one row and one column name before you trust it. A silently
    transposed array makes every metric below meaningless and none of them
    throw.
    """
    # TODO(you)
    raise NotImplementedError("wire this to your installed LeRobot version")


# --------------------------------------------------------------------------
# The metrics. Every one of them is unitless or self-referential on purpose.
# --------------------------------------------------------------------------


def _smooth(q, window=5):
    """Centred moving average, applied before anything is differentiated.

    This is not cosmetic. A raw third difference of an encoder signal measures
    the encoder, not your hand: quantisation noise of a few hundredths of a
    degree becomes hundreds of degrees per second cubed and drowns the thing you
    wanted to see. Smooth over about a sixth of a second first, then measure.
    """
    if window < 2:
        return np.asarray(q, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(q, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(q.shape[1])],
        axis=1,
    )[: len(q)]


def score(q, fps=FPS, window=5):
    """q: (T, n_joints) joint angles in degrees. Returns a dict of five numbers."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"expected (T, n_joints), got shape {q.shape}")
    dt = 1.0 / fps
    qs = _smooth(q, window)

    # Speed in joint space: how fast the whole configuration is changing.
    vel = np.diff(qs, axis=0) / dt                      # (T-1, n)
    speed = np.linalg.norm(vel, axis=1)                 # (T-1,)

    # 1. Detour. Distance actually travelled through joint space divided by the
    #    straight-line distance from the first pose to the last. 1.0 is a
    #    perfectly direct move. Backing off and coming in again inflates it.
    travelled = np.sum(np.linalg.norm(np.diff(qs, axis=0), axis=1))
    direct = np.linalg.norm(qs[-1] - qs[0])
    detour = travelled / direct if direct > 1e-9 else float("inf")

    # 2. Dwell. Fraction of the episode spent nearly still, measured against
    #    this episode's own busy speed rather than an absolute cutoff, so it
    #    survives you being a fast or a slow operator.
    busy = np.percentile(speed, 90)
    still = speed < 0.10 * busy if busy > 1e-9 else np.ones_like(speed, dtype=bool)
    dwell = float(np.mean(still))

    # 3. Reversals per second, summed over joints. A joint that changes
    #    direction is a joint that corrected itself. A deadband ignores the
    #    dither that any real encoder produces while a joint is essentially
    #    holding still.
    deadband = 0.05 * np.percentile(np.abs(vel), 90)
    sign = np.sign(np.where(np.abs(vel) < deadband, 0.0, vel))
    reversals = 0
    for j in range(sign.shape[1]):
        s = sign[:, j]
        s = s[s != 0]
        reversals += int(np.sum(np.abs(np.diff(s)) > 1e-9))
    reversals_per_s = reversals / (len(q) / fps)

    # 4. Interior stops. How many separate times the whole arm came to rest and
    #    then set off again, ignoring the rest at each end. Zero is a single
    #    committed motion. One or more is a decision you made mid-episode, and
    #    it is the most legible of these numbers when you watch the video back.
    padded = np.concatenate(([False], still, [False]))
    edges = np.diff(padded.astype(int))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)       # exclusive
    stops = int(np.sum((run_starts > 0) & (run_ends < len(still))))

    return {
        "seconds": len(q) / fps,
        "detour": detour,
        "dwell": dwell,
        "reversals/s": reversals_per_s,
        "stops": float(stops),
    }


def report(named_episodes, fps=FPS):
    """Print a table and flag anything far from the median. named_episodes: {name: q}."""
    rows = {name: score(q, fps) for name, q in named_episodes.items()}
    keys = ["seconds", "detour", "dwell", "reversals/s", "stops"]
    head = ["episode"] + keys
    width = max(len(n) for n in rows) + 2

    print(f"{head[0]:<{width}}" + "".join(f"{k:>17}" for k in keys))
    print("-" * (width + 17 * len(keys)))
    for name, r in rows.items():
        print(f"{name:<{width}}" + "".join(f"{r[k]:>17.2f}" for k in keys))

    if len(rows) >= 3:
        print("\noutliers against your own median:")
        flagged = False
        for k in ["detour", "dwell", "reversals/s", "stops"]:
            vals = np.array([r[k] for r in rows.values()])
            med = float(np.median(vals))
            for name, r in rows.items():
                # A metric whose median is zero has no scale, so any non-zero
                # value is the outlier. That case is the loudest one there is:
                # every other episode did this exactly never.
                off = r[k] > 1.5 * med if med > 1e-9 else r[k] > 1e-9
                if off:
                    print(f"  {name}: {k} = {r[k]:.2f} vs median {med:.2f}")
                    flagged = True
        if not flagged:
            print("  none. Your episodes resemble each other, which is the goal.")
    else:
        print("\n(Add three or more episodes to get outlier flagging.)")
    return rows


# --------------------------------------------------------------------------
# Synthetic demonstrations, so the metrics can be seen without hardware.
# --------------------------------------------------------------------------


def _minimum_jerk(t):
    """Standard smooth 0-to-1 profile: starts and ends at zero speed."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def rehearsed(seconds=6.0, fps=FPS, rng=None):
    """One deliberate reach: one acceleration, one deceleration, no second thoughts."""
    rng = rng or np.random.default_rng(0)
    n = int(seconds * fps)
    s = _minimum_jerk(np.linspace(0.0, 1.0, n))[:, None]
    start = np.array([0.0, -20.0, 40.0, 10.0, 0.0, 100.0])
    end = np.array([35.0, 15.0, -10.0, -25.0, 5.0, 8.0])
    q = start + s * (end - start)
    return q + rng.normal(0.0, 0.03, q.shape)   # encoder noise, not hesitation


def hesitant(seconds=6.0, fps=FPS, rng=None):
    """Same start, same end, same duration. The difference is entirely in the middle:
    a pause at 45% of the way, then a back-off and a second approach."""
    rng = rng or np.random.default_rng(0)
    n = int(seconds * fps)
    t = np.linspace(0.0, 1.0, n)

    s = np.empty_like(t)
    for i, x in enumerate(t):
        if x < 0.35:                       # first approach
            s[i] = 0.45 * _minimum_jerk(x / 0.35)
        elif x < 0.55:                     # freeze, deciding
            s[i] = 0.45
        elif x < 0.70:                     # back off
            s[i] = 0.45 - 0.18 * _minimum_jerk((x - 0.55) / 0.15)
        else:                              # commit, second approach
            s[i] = 0.27 + 0.73 * _minimum_jerk((x - 0.70) / 0.30)

    start = np.array([0.0, -20.0, 40.0, 10.0, 0.0, 100.0])
    end = np.array([35.0, 15.0, -10.0, -25.0, 5.0, 8.0])
    q = start + s[:, None] * (end - start)
    q += rng.normal(0.0, 0.06, q.shape)    # a less steady hand, too
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    episodes = {
        "rehearsed-a": rehearsed(rng=rng),
        "rehearsed-b": rehearsed(rng=rng),
        "rehearsed-c": rehearsed(rng=rng),
        "hesitant": hesitant(rng=rng),
    }
    print("Synthetic demonstrations. Same start pose, same end pose, same 6.0 s.\n")
    report(episodes)
    print(
        "\nAll four episodes 'succeed': every one of them ends at the target pose.\n"
        "Success is not the label. The path is the label."
    )


if __name__ == "__main__":
    main()
