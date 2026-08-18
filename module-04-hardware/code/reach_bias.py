"""Separate systematic reach error from random reach error (Lesson 4.16).

You measure where the gripper actually ended up on a handful of trials, against
where the object actually was. This splits that error into two numbers:

    bias     the average offset. A constant, repeatable lean.
    scatter  the spread around that average. Trial-to-trial randomness.

Which of the two dominates tells you which layer to go and look at, and the two
layers are nowhere near each other:

    bias >> scatter   geometry. Calibration, a moved camera, a shifted mount.
                      The arm is confidently going to the wrong place.
    scatter >> bias   data. Coverage, demonstration quality, visual drift.
                      The arm does not know where the place is.

Standard library only. No hardware needed: this reads numbers you wrote down.

Run:
    python reach_bias.py --demo systematic
    python reach_bias.py --demo scattered
    python reach_bias.py --csv trials.csv

The CSV wants a header and one row per trial, in millimetres, measured in the
same frame every time (a ruler taped to the table is fine, and better than a
guess from video):

    trial,target_x_mm,target_y_mm,landed_x_mm,landed_y_mm
    1,120.0,55.0,116.0,24.0
"""

import argparse
import csv
import math
import random
import sys


def summarise(errors):
    """errors: list of (dx, dy) in mm. Returns a dict of the derived numbers."""
    n = len(errors)
    if n < 3:
        raise SystemExit("need at least 3 trials before these numbers mean anything")

    mean_x = sum(dx for dx, _ in errors) / n
    mean_y = sum(dy for _, dy in errors) / n
    bias = math.hypot(mean_x, mean_y)

    # Scatter is the RMS distance from the mean error, not from zero: it is the
    # part of the error that changes trial to trial.
    var = sum((dx - mean_x) ** 2 + (dy - mean_y) ** 2 for dx, dy in errors) / (n - 1)
    scatter = math.sqrt(var)

    # Standard error of the mean, so you can say whether the bias is real or
    # just this sample of a wobbly process.
    sem = scatter / math.sqrt(n) if n else float("nan")

    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "bias": bias,
        "scatter": scatter,
        "sem": sem,
        "ratio": bias / scatter if scatter > 0 else float("inf"),
    }


def verdict(stats):
    """Plain-language reading of the two numbers. Deliberately blunt."""
    bias, scatter, sem, ratio = (stats["bias"], stats["scatter"],
                                 stats["sem"], stats["ratio"])
    lines = []

    if bias < 2 * sem:
        lines.append(
            "The mean offset is smaller than twice its own standard error, so "
            "there is no evidence of a systematic lean yet. Run more trials "
            "before chasing calibration."
        )
    elif ratio >= 2.0:
        lines.append(
            f"SYSTEMATIC. The arm leans {bias:.0f} mm the same way every time, "
            f"which is {ratio:.1f}x the trial-to-trial spread. This is geometry, "
            "not learning. Suspect, in order: a camera that moved since "
            "recording, calibration (wrong --robot.id, re-run since the last "
            "collision), a mount that shifted, the object start zone moved."
        )
    elif ratio <= 0.5:
        lines.append(
            f"RANDOM. Spread is {scatter:.0f} mm against only {bias:.0f} mm of "
            "lean, so the arm is aiming at roughly the right place and missing "
            "differently each time. This is a data problem. Suspect: too few "
            "episodes near these conditions, inconsistent demonstrations, or "
            "lighting that has drifted since recording."
        )
    else:
        lines.append(
            f"MIXED. Bias {bias:.0f} mm and scatter {scatter:.0f} mm are the "
            "same order. Fix the systematic part first: it is cheaper, and it "
            "will change the scatter estimate."
        )

    lines.append(
        f"Correcting the bias alone would move the mean miss from {bias:.0f} mm "
        f"to 0 mm and leave {scatter:.0f} mm of spread. If your grasp tolerance "
        f"is under {scatter:.0f} mm, that alone will not get you to a working "
        "policy."
    )
    return lines


def synthetic(kind, n=12, seed=0):
    """Two made-up datasets so the tool can be exercised with no robot present.

    These are illustrations of the two shapes, not measurements of any arm.
    """
    rng = random.Random(seed)
    if kind == "systematic":
        return [(rng.gauss(-30.0, 4.0), rng.gauss(2.0, 4.0)) for _ in range(n)]
    return [(rng.gauss(0.0, 22.0), rng.gauss(0.0, 22.0)) for _ in range(n)]


def read_trials(path):
    errors = []
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            errors.append((
                float(record["landed_x_mm"]) - float(record["target_x_mm"]),
                float(record["landed_y_mm"]) - float(record["target_y_mm"]),
            ))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="measured trials")
    parser.add_argument("--demo", choices=["systematic", "scattered"],
                        help="run on synthetic numbers instead of a file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.demo:
        errors = synthetic(args.demo, seed=args.seed)
        print(f"[synthetic '{args.demo}' data -- illustration, not a measurement]\n")
    elif args.csv:
        errors = read_trials(args.csv)
    else:
        parser.error("give --csv with your measurements or --demo to see the shape")

    stats = summarise(errors)
    print(f"trials            {stats['n']}")
    print(f"mean offset       x {stats['mean_x']:+7.1f} mm   "
          f"y {stats['mean_y']:+7.1f} mm")
    print(f"bias  |mean|      {stats['bias']:7.1f} mm  (+/- {stats['sem']:.1f} mm)")
    print(f"scatter  RMS      {stats['scatter']:7.1f} mm")
    print(f"bias / scatter    {stats['ratio']:7.2f}\n")
    for line in verdict(stats):
        print(line, file=sys.stdout)
        print()


if __name__ == "__main__":
    main()
