"""Module 1 milestone - scripted 2D pick-and-place (Lesson 1.19).

This is a scaffold, not a starter-with-blanks: the milestone is yours to build.
The structure below encodes the checkpoints from the lesson; replace the
NotImplementedError stubs following your own designs. No solution file exists
for this one, deliberately.

Run:  python pick_and_place.py                  # animated demo
      python pick_and_place.py --headless -n 100 [--noise 0.005]
"""
import argparse
import numpy as np

# Bring in YOUR module-1 libraries (completed lesson code):
#   transforms (1.8-1.9), fk (1.11), ik_dls (1.15)
# e.g.: sys.path.append("../code") and import from there once yours pass tests.

L1, L2 = 1.0, 0.7
BIN_POS = np.array([-0.8, 0.35])
BLOCK_REGION = {"x": (0.5, 1.3), "y": (0.1, 0.9)}     # sampling box for the block
GRASP_TOL_POS, GRASP_TOL_ANG = 0.01, np.radians(10)


class Arm:
    """State: joint angles + gripper flag. Motion = joint-space interpolation."""

    def __init__(self):
        self.theta = np.array([0.6, 0.8])
        self.holding = False

    def goto(self, target_xy, duration_s, fps=60):
        """Yield interpolated joint configurations from here to IK(target).
        TODO: IK at target (seed from self.theta; branch continuity!),
        then linear θ(t); bonus: trapezoidal profile."""
        raise NotImplementedError


class PickAndPlace:
    """The state machine from the lesson: IDLE → HOVER → DESCEND → GRASP →
    LIFT → MOVE_TO_BIN → RELEASE → HOME. Each state returns the next state."""

    def __init__(self, arm, block_pos, log):
        self.arm, self.block, self.log = arm, block_pos, log
        self.state = "IDLE"

    def step(self):
        raise NotImplementedError


def run_episode(noise_sigma=0.0, render=False, rng=None):
    """One episode. Returns dict(success=…, place_error=…, ik_iters=…, …).
    noise_sigma: gaussian error added to the REPORTED block position;
    the perception-noise experiment from checkpoint 5."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0, help="perception noise σ (meters)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = [run_episode(args.noise, render=not args.headless, rng=rng)
               for _ in range(args.n)]
    succ = np.mean([r["success"] for r in results])
    errs = [r["place_error"] for r in results if r["success"]]
    print(f"episodes={args.n}  σ={args.noise*1000:.0f}mm  "
          f"success={succ:.1%}  mean place error={np.mean(errs)*1000 if errs else float('nan'):.1f}mm")


if __name__ == "__main__":
    main()
