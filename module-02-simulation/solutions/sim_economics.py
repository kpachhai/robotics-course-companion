"""Solution - Lesson 2.1: the economics of a simulated trial, and where it lies.

Self-contained. Run:  python sim_economics.py
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
    """Where the cube lands, for one trial."""
    vx = VX * (1.0 + VX_NOISE * rng.standard_normal())
    h = H * (1.0 + H_NOISE * rng.standard_normal())
    x_open = release_x + vx * delay          # the gripper kept moving while it opened
    return x_open + vx * np.sqrt(2.0 * h / G)


def success_rate(release_x, delay_lo, delay_hi, n, rng):
    """Fraction of n trials landing inside the bin, delay drawn from [lo, hi]."""
    delays = rng.uniform(delay_lo, delay_hi, n)
    hits = 0
    for delay in delays:
        if abs(landing_x(release_x, delay, rng) - BIN_X) <= BIN_HALF:
            hits += 1
    return hits / n


def best_release_x(delay_lo, delay_hi, n, rng, grid=None):
    """Grid-search the release point that maximises success rate."""
    if grid is None:
        grid = np.linspace(0.40, 0.62, 111)
    scores = [success_rate(x, delay_lo, delay_hi, n, rng) for x in grid]
    return float(grid[int(np.argmax(scores))]), float(np.max(scores))


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

    assert score_perfect > 0.90 and in_sim > 0.90, "a delay-free sim should look great"
    assert on_hardware < 0.50, "the unmodelled delay should hurt badly"
    assert randomised_on_hardware > 0.80, "randomising the delay should recover most of it"
    print("\n✅ solution sim_economics: sim is cheap, and cheap is not the same as right.")


if __name__ == "__main__":
    main()
