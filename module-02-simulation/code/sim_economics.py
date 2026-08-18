"""Lesson 2.1 - the economics of a simulated trial, and where it lies.

A cube is carried sideways at VX, held H above the floor, and dropped into a bin.
You pick one number: the x at which you command the gripper to open. The world
adds noise, and the real gripper opens TRUE_DELAY seconds after you ask.

Three runs, one point:
  1. tune the release point in a sim where the gripper opens instantly - it looks perfect
  2. take that same number to a world with a 50 ms delay - it falls apart
  3. tune it again over a *randomised* delay - most of it comes back

Nothing here needs a physics engine. That is deliberate: this is what a simulator
buys you and what it costs you, before you install one.

Run:  python sim_economics.py
"""
import time

import numpy as np

G = 9.81
VX = 0.50            # m/s - the gripper carries the cube sideways at this speed
H = 0.25             # m   - release height above the floor
BIN_X = 0.60         # m   - centre of the bin
BIN_HALF = 0.020     # m   - half-width; a 4 cm bin for a 2 cm cube
TRUE_DELAY = 0.050   # s   - the real servo opens this late. Sim thinks it is 0.

VX_NOISE = 0.06      # 6% spread on carry speed, trial to trial
H_NOISE = 0.10       # 10% spread on release height


def landing_x(release_x, delay, rng):
    """Where the cube lands, for one trial.

    TODO(you):
      1. draw this trial's carry speed:  vx = VX * (1 + VX_NOISE * rng.standard_normal())
      2. draw this trial's height:       h  = H  * (1 + H_NOISE  * rng.standard_normal())
      3. the gripper keeps moving while it opens, so it actually lets go at
         x_open = release_x + vx * delay
      4. free fall from h takes sqrt(2h/g) seconds, during which the cube keeps
         its horizontal speed. Return where it lands.
    """
    raise NotImplementedError


def success_rate(release_x, delay_lo, delay_hi, n, rng):
    """Fraction of n trials landing inside the bin, delay drawn from [lo, hi].

    TODO(you): draw n delays with rng.uniform(delay_lo, delay_hi, n), run a trial
    for each, and count the ones with |landing_x - BIN_X| <= BIN_HALF.
    """
    raise NotImplementedError


def best_release_x(delay_lo, delay_hi, n, rng, grid=None):
    """Grid-search the release point that maximises success rate.

    TODO(you): score every x in `grid` with success_rate(), then return
    (best_x, best_score). np.argmax does the second half.
    """
    if grid is None:
        grid = np.linspace(0.40, 0.62, 111)
    raise NotImplementedError


def main():
    rng = np.random.default_rng(0)
    n = 400

    t0 = time.perf_counter()
    tuned_perfect, score_perfect = best_release_x(0.0, 0.0, n, rng)
    elapsed = time.perf_counter() - t0
    trials = 111 * n
    print(f"tuning pass: {trials:,} trials in {elapsed:.2f}s "
          f"= {trials / elapsed:,.0f} trials/second on one core")
    print(f"the same {trials:,} attempts on hardware, at 12s each plus a 20s reset, "
          f"would take {trials * 32 / 3600:,.0f} hours\n")

    tuned_random, _ = best_release_x(0.02, 0.08, n, rng)

    trials = 2000
    in_sim = success_rate(tuned_perfect, 0.0, 0.0, trials, rng)
    on_hardware = success_rate(tuned_perfect, TRUE_DELAY, TRUE_DELAY, trials, rng)
    randomised_on_hardware = success_rate(tuned_random, TRUE_DELAY, TRUE_DELAY, trials, rng)

    print(f"every row below is {trials:,} fresh trials")
    print(f"{'tuned no-delay, run no-delay':<38} release_x={tuned_perfect:.3f}  "
          f"success={in_sim:6.1%}")
    print(f"{'tuned no-delay, run 50 ms late':<38} release_x={tuned_perfect:.3f}  "
          f"success={on_hardware:6.1%}")
    print(f"{'tuned on 20-80 ms, run 50 ms late':<38} release_x={tuned_random:.3f}  "
          f"success={randomised_on_hardware:6.1%}")


if __name__ == "__main__":
    main()
