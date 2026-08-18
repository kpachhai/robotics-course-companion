"""Module 2 milestone - a hundred episodes and a dataset that replays (Lesson 2.18).

This is a scaffold, not a starter-with-blanks: the milestone is yours to build.
The structure below encodes the checkpoints from the lesson; replace the
NotImplementedError stubs with your own. No solution file exists for this one,
deliberately.

It imports YOUR completed lesson code from ../code:
  so101_pick.py      the scene and the arm            (Lesson 2.11)
  record_dataset.py  frames, schema and replay        (Lesson 2.17, TODOs done)

Run:  python collect.py --eval   -n 100 --seed 0
      python collect.py --record -n 50 --out pick50
      python collect.py --eval   -n 100 --width 0.10      # widen the box, watch it fall
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))
import record_dataset as rd                                    # noqa: E402
import so101_pick as sp                                        # noqa: E402

NOMINAL = np.array([0.22, 0.0])      # same nominal cube spot as Lesson 2.16
HALF_WIDTH = 0.08                    # the sampling box, in metres. Report it with every number.
TARGET = 0.90                        # the bar this milestone has to clear


def sample_cube(rng, half_width=HALF_WIDTH):
    """Uniform in a square around the nominal spot. Deliberately boring and reproducible."""
    return NOMINAL + rng.uniform(-half_width, half_width, size=2)


def classify(outcome, arm):
    """Give every failure a name. Counting them is not enough.

    Suggested labels, from the four things that actually go wrong:
      "success"        cube_in_bin says yes
      "never_lifted"   the cube never got above the table
      "dropped"        it was lifted and is not in the bin
      "not_settled"    it is inside the bin footprint but still moving or still held
      "off_target"     lifted, released, and landed outside the bin

    You have outcome["success"], outcome["lifted"], outcome["cube_end"], and the
    arm itself for anything else you want to test.

    TODO(you): return one label.
    """
    raise NotImplementedError


def evaluate(n, seed, half_width=HALF_WIDTH):
    """Run n randomised episodes with no rendering and report what happened.

    TODO(you):
      - one sp.Arm, reused across episodes (loading the model 100 times is slow)
      - arm.reset() and arm.place_cube(*sample_cube(rng)) each episode
      - rd.run_episode(arm) with no recorder and no renderer
      - tally classify() labels, and keep the (x, y) of every failure
      - return a dict with the rate, the label counts, the failure positions,
        and the seed, half_width and n you were called with

    Print as you go. A hundred episodes takes a couple of minutes and a silent
    script is indistinguishable from a hung one.
    """
    raise NotImplementedError


def record(n, out, seed, half_width=HALF_WIDTH):
    """Record n SUCCESSFUL episodes into a dataset directory.

    TODO(you):
      - build the recorder with rd.Recorder(out, rd.features(), fps=rd.FPS)
      - build a renderer with mujoco.Renderer(arm.model, rd.IMG_H, rd.IMG_W)
      - loop until you have kept n successes, discarding failures by clearing
        recorder.frames (a failed episode must not reach save_episode)
      - save_episode() with the metadata the format has no column for:
        the seed, the cube's PLACED pose, the success criterion, the attempt number
      - recorder.finalize(), then report kept, discarded and bytes on disk

    Then run rd.check(out) yourself before you believe any of it.
    """
    raise NotImplementedError


def report(result):
    """Print the evaluation as a markdown table, ready to paste into the writeup."""
    print(f"\n| metric | value |\n|---|---|")
    print(f"| episodes | {result['n']} |")
    print(f"| sampling box | {result['half_width'] * 100:.0f} cm half-width "
          f"around ({NOMINAL[0]:.2f}, {NOMINAL[1]:.2f}) |")
    print(f"| seed | {result['seed']} |")
    print(f"| success | {result['rate']:.0%} |")
    for label, count in sorted(result["labels"].items()):
        if label != "success":
            print(f"| {label} | {count} |")
    verdict = "PASS" if result["rate"] >= TARGET else "BELOW TARGET"
    print(f"\n{verdict}: {result['rate']:.0%} against a target of {TARGET:.0%}")
    if result["failures"]:
        radii = [float(np.hypot(x, y)) for x, y in result["failures"]]
        print(f"failure radii from the base: {min(radii):.3f} to {max(radii):.3f} m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="measure the success rate")
    parser.add_argument("--record", action="store_true", help="write the demonstration set")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=float, default=HALF_WIDTH, help="box half-width, metres")
    parser.add_argument("--out", default="pick50")
    args = parser.parse_args()

    if args.record:
        record(args.n, args.out, args.seed, args.width)
        return
    result = evaluate(args.n, args.seed, args.width)
    report(result)
    Path("eval.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
